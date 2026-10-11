package ann

import "base:intrinsics"
import tensor "../odin-tensor"

// forward runs the network in inference mode. input's trailing axes are the
// network's input sample shape, and any leading axes are batch axes, which
// the output keeps. The result is owned by the caller.
forward :: proc(network: Network($T), input: tensor.Tensor(T), allocator := context.allocator) -> (tensor.Tensor(T), Error) where intrinsics.type_is_float(T) {
	return _forward(network, tensor.as_view(input), false, allocator)
}

// forward_views is forward for a borrowed view, such as a slice of a dataset.
forward_views :: proc(network: Network($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (tensor.Tensor(T), Error) where intrinsics.type_is_float(T) {
	return _forward(network, input, false, allocator)
}

// _forward runs every layer, skipping the last activation when
// skip_last_activation is set so fused losses can work from logits.
_forward :: proc(network: Network($T), input: tensor.Tensor_View(T), skip_last_activation: bool, allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	_check_input(network, input.shape) or_return
	defer if err != nil {
		tensor.release(&output)
	}
	current := input
	for layer, i in network.layers {
		next := _layer_forward(layer, current, allocator) or_return
		tensor.release(&output)
		output = next
		if !(skip_last_activation && i == len(network.layers)-1) {
			activate_in_place(&output, layer.activation, allocator) or_return
		}
		current = tensor.as_view(output)
	}
	return output, nil
}

// _layer_forward computes one layer before its activation, as a new tensor.
// input's trailing axes are the layer's input sample shape.
_layer_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	switch layer.kind {
	case .Dense:
		return tensor.linear_views(input, tensor.as_view(layer.weights), tensor.as_view(layer.bias), allocator)
	case .Conv2D:
		return _conv2d_forward(layer, input, allocator)
	case .Max_Pool2D:
		return _max_pool2d_forward(layer, input, allocator)
	case .Flatten:
		return _flatten_forward(layer, input, allocator)
	case .Dropout:
		return _dropout_forward(input, allocator)
	case .Batch_Norm:
		return _batch_norm_forward(layer, input, allocator)
	case .Upsample2D:
		return _upsample2d_forward(layer, input, allocator)
	}
	return output, nil
}

// activate_in_place applies activation to every element of t. Softmax needs
// two small [..., 1] temporaries from allocator.
activate_in_place :: proc(t: ^tensor.Tensor($T), activation: Activation, allocator := context.allocator) -> Error where intrinsics.type_is_float(T) {
	switch activation {
	case .Identity:
	case .ReLU:
		tensor.relu_in_place(t) or_return
	case .Sigmoid:
		tensor.sigmoid_in_place(t) or_return
	case .Tanh:
		tensor.tanh_in_place(t) or_return
	case .Softmax:
		softmax_in_place(t, allocator) or_return
	}
	return nil
}

// softmax_in_place normalizes t over its last axis. Subtracting each row's
// maximum first keeps exp from overflowing.
softmax_in_place :: proc(t: ^tensor.Tensor($T), allocator := context.allocator) -> Error where intrinsics.type_is_float(T) {
	if len(t.shape) == 0 {
		return .Invalid_Shape
	}
	row_max := tensor.max_axis(t^, -1, keep_dims = true, allocator = allocator) or_return
	defer tensor.release(&row_max)
	tensor.sub_in_place(t, row_max) or_return
	tensor.exp_in_place(t) or_return
	row_sum := tensor.sum_axis(t^, -1, keep_dims = true, allocator = allocator) or_return
	defer tensor.release(&row_sum)
	tensor.div_in_place(t, row_sum) or_return
	return nil
}
