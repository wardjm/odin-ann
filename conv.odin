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

// conv2d appends a valid, stride-1 convolution with square kernel_size
// kernels. Its input must be [height, width, input_channels] with height and
// width at least kernel_size, from set_input_shape or the preceding layer; the
// output is [height-kernel_size+1, width-kernel_size+1, output_channels].
// Weights are [kernel_size, kernel_size, input_channels, output_channels],
// drawn like dense weights with fan-in kernel_size²·input_channels and fan-out
// kernel_size²·output_channels; biases start at zero.
conv2d :: proc(network: ^Network($T), input_channels, output_channels, kernel_size: int, activation := Activation.Identity) -> Error where intrinsics.type_is_float(T) {
	if input_channels <= 0 || output_channels <= 0 || kernel_size <= 0 {
		return .Invalid_Shape
	}
	input_shape := _current_shape(network^)
	if input_shape.rank != 3 || input_shape.dims[2] != input_channels || input_shape.dims[0] < kernel_size || input_shape.dims[1] < kernel_size {
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

	output_shape := _sample_shape(input_shape.dims[0]-kernel_size+1, input_shape.dims[1]-kernel_size+1, output_channels)
	layer := Layer(T){kind = .Conv2D, weights = weights, bias = bias, activation = activation, input_shape = input_shape, output_shape = output_shape}
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
	height_axis := len(input_shape) - 3
	columns := len(layer.weights.data) / layer.weights.shape[3]
	rows_buffer^ = {-1, columns}
	copy(output_buffer, input_shape[:height_axis])
	sample := layer.output_shape
	copy(output_buffer[height_axis:], _sample_dims(&sample))
	return rows_buffer[:], output_buffer[:height_axis+3]
}

// _conv2d_forward computes a Conv2D layer's output before its activation.
_conv2d_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	kernel_size := layer.weights.shape[0]
	windows := tensor.im2col_views(input, kernel_size, kernel_size, 1, allocator) or_return
	defer tensor.release(&windows)
	rows_buffer: [2]int
	output_buffer: [tensor.MAX_RANK]int
	rows, output_shape := _conv2d_shapes(layer, input.shape, &rows_buffer, output_buffer[:])
	tensor.reshape(&windows, rows) or_return

	kernel_shape := [2]int{rows[1], layer.weights.shape[3]}
	kernel_strides := [2]int{layer.weights.shape[3], 1}
	kernels := tensor.Tensor_View(T){data = layer.weights.data, shape = kernel_shape[:], strides = kernel_strides[:]}
	output = tensor.matmul_views(tensor.as_view(windows), kernels, allocator) or_return
	defer if err != nil {
		tensor.release(&output)
	}
	tensor.add_in_place(&output, layer.bias) or_return
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

	windows := tensor.tape_im2col(tape, input, kernel_size, kernel_size) or_return
	windows = tensor.tape_reshape(tape, windows, rows) or_return
	kernels := tensor.tape_reshape(tape, weights, []int{rows[1], -1}) or_return
	product := tensor.tape_matmul(tape, windows, kernels) or_return
	output = tensor.tape_add(tape, product, bias) or_return
	return tensor.tape_reshape(tape, output, output_shape)
}

// _pool_blocks_shape writes the shape that splits [..., height, width, c] into
// [..., out_h, pool, out_w, pool, c] blocks into buffer. Taking the max over
// the axes at height_axis+3 and then height_axis+1 pools each window.
_pool_blocks_shape :: proc(layer: Layer($T), leading: []int, buffer: []int) -> []int {
	copy(buffer, leading)
	pool := layer.pool_size
	blocks := [5]int{layer.output_shape.dims[0], pool, layer.output_shape.dims[1], pool, layer.output_shape.dims[2]}
	copy(buffer[len(leading):], blocks[:])
	return buffer[:len(leading)+5]
}

// _max_pool2d_forward copies the pooled region of input into contiguous
// blocks and takes the maximum of each window.
_max_pool2d_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	height_axis := len(input.shape) - 3
	pool := layer.pool_size
	rows_shape, rows_strides, region_shape, region_strides: [tensor.MAX_RANK]int
	rows := tensor.slice_view(input, height_axis, 0, layer.output_shape.dims[0]*pool, 1, rows_shape[:], rows_strides[:]) or_return
	region := tensor.slice_view(rows, height_axis+1, 0, layer.output_shape.dims[1]*pool, 1, region_shape[:], region_strides[:]) or_return
	blocks := tensor.copy_view(region, allocator) or_return
	defer tensor.release(&blocks)
	shape_buffer: [tensor.MAX_RANK]int
	tensor.reshape(&blocks, _pool_blocks_shape(layer, input.shape[:height_axis], shape_buffer[:])) or_return
	columns := tensor.max_axis(blocks, height_axis+3, allocator = allocator) or_return
	defer tensor.release(&columns)
	return tensor.max_axis(columns, height_axis+1, allocator = allocator)
}

_record_max_pool2d :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input: tensor.Tape_Value(T)) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	value := tensor.tape_value(tape, input) or_return
	height_axis := len(value.shape) - 3
	pool := layer.pool_size
	region := input
	if rows := layer.output_shape.dims[0] * pool; rows != value.shape[height_axis] {
		region = tensor.tape_slice(tape, region, height_axis, 0, rows) or_return
	}
	if columns := layer.output_shape.dims[1] * pool; columns != value.shape[height_axis+1] {
		region = tensor.tape_slice(tape, region, height_axis+1, 0, columns) or_return
	}
	shape_buffer: [tensor.MAX_RANK]int
	blocks := tensor.tape_reshape(tape, region, _pool_blocks_shape(layer, value.shape[:height_axis], shape_buffer[:])) or_return
	columns := tensor.tape_max_axis(tape, blocks, height_axis+3) or_return
	return tensor.tape_max_axis(tape, columns, height_axis+1)
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
