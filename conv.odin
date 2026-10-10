package ann

import "base:intrinsics"
import tensor "../odin-tensor"

// Image layers work on [height, width, channels] samples, channels last. Any
// leading axes are batch axes, as for dense layers.
//
// A convolution is computed as a sum over kernel offsets. For each (dy, dx),
// the input window [..., dy:dy+out_h, dx:dx+out_w, :] is a strided view, and
// multiplying it by that offset's [input_channels, output_channels] slice of
// the weights gives its contribution to every output position. Training
// records the same views, products and sums on the tape, so forward and
// train_step add the same terms in the same order and agree exactly.

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

// _conv2d_forward computes a Conv2D layer's output before its activation.
_conv2d_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	kernel_size, input_channels := layer.weights.shape[0], layer.weights.shape[2]
	output_height, output_width := layer.output_shape.dims[0], layer.output_shape.dims[1]
	height_axis := len(input.shape) - 3
	defer if err != nil {
		tensor.release(&output)
	}
	product: tensor.Tensor(T)
	defer tensor.release(&product)

	// The weights viewed as [kernel_size²·input_channels, output_channels],
	// whose rows for offset (dy, dx) start at (dy*kernel_size + dx)*input_channels.
	kernel_shape := [2]int{kernel_size * kernel_size * input_channels, layer.weights.shape[3]}
	kernel_strides := [2]int{layer.weights.shape[3], 1}
	kernels := tensor.Tensor_View(T){data = layer.weights.data, shape = kernel_shape[:], strides = kernel_strides[:]}

	rows_shape, rows_strides, window_shape, window_strides: [tensor.MAX_RANK]int
	offset_shape, offset_strides: [2]int
	for dy in 0..<kernel_size {
		rows := tensor.slice_view(input, height_axis, dy, output_height, 1, rows_shape[:], rows_strides[:]) or_return
		for dx in 0..<kernel_size {
			window := tensor.slice_view(rows, height_axis+1, dx, output_width, 1, window_shape[:], window_strides[:]) or_return
			kernel := tensor.slice_view(kernels, 0, (dy*kernel_size+dx)*input_channels, input_channels, 1, offset_shape[:], offset_strides[:]) or_return
			if dy == 0 && dx == 0 {
				output = tensor.matmul_views(window, kernel, allocator) or_return
				continue
			}
			if product.data == nil {
				product = tensor.new_tensor(T, output.shape, allocator) or_return
			}
			tensor.matmul_views_into(&product, window, kernel) or_return
			tensor.add_in_place(&output, product) or_return
		}
	}
	tensor.add_in_place(&output, layer.bias) or_return
	return output, nil
}

// _record_conv2d records a Conv2D layer before its activation, with the same
// terms in the same order as _conv2d_forward.
_record_conv2d :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input, weights, bias: tensor.Tape_Value(T)) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	kernel_size, input_channels := layer.weights.shape[0], layer.weights.shape[2]
	output_height, output_width := layer.output_shape.dims[0], layer.output_shape.dims[1]
	value := tensor.tape_value(tape, input) or_return
	height_axis := len(value.shape) - 3

	kernels := tensor.tape_reshape(tape, weights, []int{kernel_size * kernel_size * input_channels, -1}) or_return
	for dy in 0..<kernel_size {
		rows := tensor.tape_slice(tape, input, height_axis, dy, output_height) or_return
		for dx in 0..<kernel_size {
			window := tensor.tape_slice(tape, rows, height_axis+1, dx, output_width) or_return
			kernel := tensor.tape_slice(tape, kernels, 0, (dy*kernel_size+dx)*input_channels, input_channels) or_return
			product := tensor.tape_matmul(tape, window, kernel) or_return
			if dy == 0 && dx == 0 {
				output = product
			} else {
				output = tensor.tape_add(tape, output, product) or_return
			}
		}
	}
	return tensor.tape_add(tape, output, bias)
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
