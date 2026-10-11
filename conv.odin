package ann

import "base:intrinsics"
import tensor "../odin-tensor"

// Image layers work on [height, width, channels] samples, channels last. Any
// leading axes are batch axes, as for dense layers.
//
// A convolution gathers every input window into a row with im2col, giving
// [positions, kernel²·input_channels] once the batch and spatial axes are
// merged, and multiplies that by the weights viewed as
// [kernel²·input_channels, output_channels]. Training records the same
// im2col, product and bias add on the tape, so forward and train_step agree
// exactly.

// conv2d appends a convolution with square kernel_size kernels. The image is
// surrounded by padding rows and columns of zeros, and the kernels move stride
// pixels at a time, so a 3x3 kernel with padding 1 and stride 1 keeps the
// image's size and stride 2 halves it. Its input must be [height, width,
// input_channels] from set_input_shape or the preceding layer, with
// height+2·padding and width+2·padding at least kernel_size; the output is
// [(height+2·padding-kernel_size)/stride+1, likewise for width,
// output_channels], rounding down, so windows that would pass the padded
// image's edge are dropped. Weights are [kernel_size, kernel_size,
// input_channels, output_channels], drawn like dense weights with fan-in
// kernel_size²·input_channels and fan-out kernel_size²·output_channels;
// biases start at zero.
conv2d :: proc(network: ^Network($T), input_channels, output_channels, kernel_size: int, activation := Activation.Identity, stride := 1, padding := 0) -> Error where intrinsics.type_is_float(T) {
	if input_channels <= 0 || output_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0 {
		return .Invalid_Shape
	}
	input_shape := _current_shape(network^)
	if input_shape.rank != 3 || input_shape.dims[2] != input_channels {
		return .Shape_Mismatch
	}
	if padding > (max(int)-max(input_shape.dims[0], input_shape.dims[1]))/2 {
		return .Invalid_Shape
	}
	height, width := input_shape.dims[0]+2*padding, input_shape.dims[1]+2*padding
	if height < kernel_size || width < kernel_size {
		return .Shape_Mismatch
	}

	weights := tensor.new_tensor(T, []int{kernel_size, kernel_size, input_channels, output_channels}, network.allocator) or_return
	bias, err := tensor.new_tensor(T, []int{output_channels}, network.allocator)
	if err != .None {
		tensor.release(&weights)
		return err
	}
	area := kernel_size * kernel_size
	_fill_uniform(network, &weights, _init_limit(area*input_channels, area*output_channels, activation))

	output_shape := _sample_shape((height-kernel_size)/stride+1, (width-kernel_size)/stride+1, output_channels)
	layer := Layer(T){kind = .Conv2D, weights = weights, bias = bias, activation = activation, stride = stride, padding = padding, input_shape = input_shape, output_shape = output_shape}
	return _append_layer(network, layer)
}

// max_pool2d appends max pooling over non-overlapping pool_size × pool_size
// windows. Its input must be [height, width, channels] with height and width
// at least pool_size; the output is [height/pool_size, width/pool_size,
// channels], rounding down, so trailing rows and columns that do not fill a
// window are dropped. The gradient reaches one maximal element per window.
max_pool2d :: proc(network: ^Network($T), pool_size := 2) -> Error where intrinsics.type_is_float(T) {
	if pool_size <= 0 {
		return .Invalid_Shape
	}
	input_shape := _current_shape(network^)
	if input_shape.rank != 3 || input_shape.dims[0] < pool_size || input_shape.dims[1] < pool_size {
		return .Shape_Mismatch
	}
	output_shape := _sample_shape(input_shape.dims[0]/pool_size, input_shape.dims[1]/pool_size, input_shape.dims[2])
	return _append_layer(network, Layer(T){kind = .Max_Pool2D, pool_size = pool_size, input_shape = input_shape, output_shape = output_shape})
}

// upsample2d appends nearest-neighbour upsampling, which repeats each pixel
// factor times along height and width, so [height, width, channels] becomes
// [height·factor, width·factor, channels]. Followed by a padded convolution,
// it lets a decoder grow an image back to full size. Each output's gradient
// goes to the pixel it repeats.
upsample2d :: proc(network: ^Network($T), factor := 2) -> Error where intrinsics.type_is_float(T) {
	if factor <= 0 {
		return .Invalid_Shape
	}
	input_shape := _current_shape(network^)
	if input_shape.rank != 3 {
		return .Shape_Mismatch
	}
	height, width, channels := input_shape.dims[0], input_shape.dims[1], input_shape.dims[2]
	if height > max(int)/factor || width > max(int)/factor {
		return .Invalid_Shape
	}
	output_shape := _sample_shape(height*factor, width*factor, channels)
	return _append_layer(network, Layer(T){kind = .Upsample2D, factor = factor, input_shape = input_shape, output_shape = output_shape})
}

// flatten appends a layer that reshapes each sample into one vector in
// row-major order, so [h, w, c] becomes [h*w*c].
flatten :: proc(network: ^Network($T)) -> Error where intrinsics.type_is_float(T) {
	input_shape := _current_shape(network^)
	if input_shape.rank == 0 {
		return .Shape_Mismatch
	}
	features := 1
	for dim in _sample_dims(&input_shape) {
		features *= dim
	}
	return _append_layer(network, Layer(T){kind = .Flatten, input_shape = input_shape, output_shape = _sample_shape(features)})
}

// _conv2d_shapes writes the shapes a convolution of input_shape works in:
// the windows merged into [positions, kernel²·input_channels] rows, and the
// output with input's batch axes.
_conv2d_shapes :: proc(layer: Layer($T), input_shape: []int, rows_buffer: ^[2]int, output_buffer: []int) -> (rows, output: []int) {
	columns := len(layer.weights.data) / layer.weights.shape[3]
	rows_buffer^ = {-1, columns}
	return rows_buffer[:], _image_output_shape(layer, input_shape, output_buffer)
}

// _conv2d_forward computes a Conv2D layer's output before its activation.
_conv2d_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	kernel_size := layer.weights.shape[0]
	windows := tensor.im2col_views(input, kernel_size, kernel_size, layer.stride, layer.padding, allocator) or_return
	defer tensor.release(&windows)
	rows_buffer: [2]int
	output_buffer: [tensor.MAX_RANK]int
	rows, output_shape := _conv2d_shapes(layer, input.shape, &rows_buffer, output_buffer[:])
	tensor.reshape(&windows, rows) or_return

	kernel_shape := [2]int{rows[1], layer.weights.shape[3]}
	kernel_strides := [2]int{layer.weights.shape[3], 1}
	kernels := tensor.Tensor_View(T){data = layer.weights.data, shape = kernel_shape[:], strides = kernel_strides[:]}
	output = tensor.linear_views(tensor.as_view(windows), kernels, tensor.as_view(layer.bias), allocator) or_return
	defer if err != nil {
		tensor.release(&output)
	}
	tensor.reshape(&output, output_shape) or_return
	return output, nil
}

// _record_conv2d records a Conv2D layer before its activation, computing it
// exactly as _conv2d_forward does.
_record_conv2d :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input, weights, bias: tensor.Tape_Value(T)) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	kernel_size := layer.weights.shape[0]
	value := tensor.tape_value(tape, input) or_return
	rows_buffer: [2]int
	output_buffer: [tensor.MAX_RANK]int
	rows, output_shape := _conv2d_shapes(layer, value.shape, &rows_buffer, output_buffer[:])

	windows := tensor.tape_im2col(tape, input, kernel_size, kernel_size, layer.stride, layer.padding) or_return
	windows = tensor.tape_reshape(tape, windows, rows) or_return
	kernels := tensor.tape_reshape(tape, weights, []int{rows[1], -1}) or_return
	output = tensor.tape_linear(tape, windows, kernels, bias) or_return
	return tensor.tape_reshape(tape, output, output_shape)
}

_max_pool2d_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	return tensor.max_pool2d_views(input, layer.pool_size, layer.pool_size, allocator)
}

_record_max_pool2d :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input: tensor.Tape_Value(T)) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	return tensor.tape_max_pool2d(tape, input, layer.pool_size, layer.pool_size)
}

// _image_output_shape writes input's batch axes followed by an image layer's
// output sample shape into buffer.
_image_output_shape :: proc(layer: Layer($T), input_shape: []int, buffer: []int) -> []int {
	height_axis := len(input_shape) - 3
	copy(buffer, input_shape[:height_axis])
	sample := layer.output_shape
	copy(buffer[height_axis:], _sample_dims(&sample))
	return buffer[:height_axis+3]
}

// _upsample2d_forward writes each input row's pixels factor times each, then
// copies that output row factor-1 more times.
_upsample2d_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	source := tensor.copy_view(input, allocator) or_return
	defer tensor.release(&source)
	shape_buffer: [tensor.MAX_RANK]int
	output = tensor.new_tensor(T, _image_output_shape(layer, input.shape, shape_buffer[:]), allocator) or_return
	factor := layer.factor
	width, channels := layer.input_shape.dims[1], layer.input_shape.dims[2]
	input_row := width * channels
	output_row := input_row * factor
	next := 0
	for start := 0; start < len(source.data); start += input_row {
		row := output.data[next:][:output_row]
		for x in 0..<width {
			pixel := source.data[start + x*channels:][:channels]
			for copy_index in 0..<factor {
				copy(row[(x*factor + copy_index)*channels:][:channels], pixel)
			}
		}
		for copy_index in 1..<factor {
			copy(output.data[next + copy_index*output_row:][:output_row], row)
		}
		next += factor * output_row
	}
	return output, nil
}

// _record_upsample2d views input as [images, height, 1, width, 1, channels]
// and multiplies it by ones broadcast to [factor, 1, factor, 1], which
// repeats every pixel exactly; the product's gradient sums each pixel's
// copies.
_record_upsample2d :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input: tensor.Tape_Value(T)) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	value := tensor.tape_value(tape, input) or_return
	height, width, channels := layer.input_shape.dims[0], layer.input_shape.dims[1], layer.input_shape.dims[2]
	images := 1
	for dim in value.shape[:len(value.shape)-3] {
		images *= dim
	}
	factor := layer.factor
	ones := tensor.new_tensor(T, []int{factor, 1, factor, 1}, tape.allocator) or_return
	defer tensor.release(&ones)
	tensor.fill(&ones, 1)
	ones_leaf := tensor.tape_add_leaf_copy(tape, tensor.as_view(ones), false) or_return

	split := tensor.tape_reshape(tape, input, []int{images, height, 1, width, 1, channels}) or_return
	repeated := tensor.tape_mul(tape, split, ones_leaf) or_return
	shape_buffer: [tensor.MAX_RANK]int
	return tensor.tape_reshape(tape, repeated, _image_output_shape(layer, value.shape, shape_buffer[:]))
}

// _flatten_shape writes input's batch axes followed by the layer's flattened
// feature count into buffer.
_flatten_shape :: proc(layer: Layer($T), input_shape: []int, buffer: []int) -> []int {
	leading := len(input_shape) - layer.input_shape.rank
	copy(buffer, input_shape[:leading])
	buffer[leading] = layer.output_shape.dims[0]
	return buffer[:leading+1]
}

_flatten_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	output = tensor.copy_view(input, allocator) or_return
	shape_buffer: [tensor.MAX_RANK]int
	if reshape_err := tensor.reshape(&output, _flatten_shape(layer, input.shape, shape_buffer[:])); reshape_err != .None {
		tensor.release(&output)
		return output, reshape_err
	}
	return output, nil
}

_record_flatten :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input: tensor.Tape_Value(T)) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	value := tensor.tape_value(tape, input) or_return
	shape_buffer: [tensor.MAX_RANK]int
	return tensor.tape_reshape(tape, input, _flatten_shape(layer, value.shape, shape_buffer[:]))
}
