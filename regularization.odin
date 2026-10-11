package ann

import "base:intrinsics"
import "base:runtime"
import "core:math"
import "core:math/rand"
import tensor "../odin-tensor"

// Dropout and Batch_Norm compute different things while training. train_step
// and fit record the training behavior; forward, evaluate and the metrics use
// the inference behavior, so they do not match the training loss exactly for
// networks with these layers.

// dropout appends a layer that, while training, zeroes each value with
// probability rate and scales the rest by 1/(1-rate), so their expected value
// is unchanged. Outside training it passes values through. rate must be in
// [0, 1). The trainer's generator draws the masks.
dropout :: proc(network: ^Network($T), rate: f64) -> Error where intrinsics.type_is_float(T) {
	if !(rate >= 0 && rate < 1) {
		return .Invalid_Hyperparameter
	}
	shape := _current_shape(network^)
	if shape.rank == 0 {
		return .Shape_Mismatch
	}
	return _append_layer(network, Layer(T){kind = .Dropout, rate = rate, input_shape = shape, output_shape = shape})
}

// batch_norm appends batch normalization over the last axis, which is the
// features after a dense layer and the channels after a convolution. While
// training, each feature is normalized by its mean and variance over every
// other axis of the batch, then scaled and shifted by trainable parameters
// that start at 1 and 0. Each step also moves the running mean and the
// running, unbiased variance momentum of the way to the batch's values;
// outside training the layer normalizes with those instead. activation is
// applied afterwards, so a layer before it would usually use Identity.
batch_norm :: proc(network: ^Network($T), activation := Activation.Identity, momentum := 0.1, epsilon := 1e-5) -> (err: Error) where intrinsics.type_is_float(T) {
	if !(momentum >= 0 && momentum <= 1) || !(epsilon > 0) || math.is_inf(epsilon) {
		return .Invalid_Hyperparameter
	}
	shape := _current_shape(network^)
	if shape.rank == 0 {
		return .Shape_Mismatch
	}
	features := shape.dims[shape.rank-1]
	layer := Layer(T){kind = .Batch_Norm, activation = activation, momentum = momentum, epsilon = epsilon, input_shape = shape, output_shape = shape}
	defer if err != nil {
		_release_layer(&layer)
	}
	layer.weights = tensor.new_tensor(T, []int{features}, network.allocator) or_return
	layer.bias = tensor.new_tensor(T, []int{features}, network.allocator) or_return
	layer.running_mean = tensor.new_tensor(T, []int{features}, network.allocator) or_return
	layer.running_variance = tensor.new_tensor(T, []int{features}, network.allocator) or_return
	tensor.fill(&layer.weights, 1)
	tensor.fill(&layer.running_variance, 1)
	// _append_layer releases the layer itself if it fails.
	append_err := _append_layer(network, layer)
	layer = {}
	return append_err
}

_dropout_forward :: proc(input: tensor.Tensor_View($T), allocator := context.allocator) -> (tensor.Tensor(T), Error) where intrinsics.type_is_float(T) {
	return tensor.copy_view(input, allocator)
}

// _batch_norm_forward normalizes with the running statistics, folded into
// one scale and shift per feature.
_batch_norm_forward :: proc(layer: Layer($T), input: tensor.Tensor_View(T), allocator := context.allocator) -> (output: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	features := len(layer.weights.data)
	scale := tensor.new_tensor(T, []int{2, features}, allocator) or_return
	defer tensor.release(&scale)
	for f in 0..<features {
		scale.data[f] = layer.weights.data[f] / math.sqrt(layer.running_variance.data[f] + T(layer.epsilon))
		scale.data[features+f] = layer.bias.data[f] - layer.running_mean.data[f]*scale.data[f]
	}
	output = tensor.copy_view(input, allocator) or_return
	for start := 0; start < len(output.data); start += features {
		row := output.data[start:][:features]
		for &value, f in row {
			value = value*scale.data[f] + scale.data[features+f]
		}
	}
	return output, nil
}

// _record_dropout multiplies input by a constant mask of zeros and
// 1/(1-rate).
_record_dropout :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input: tensor.Tape_Value(T), training: _Training) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	if layer.rate == 0 {
		return input, nil
	}
	value := tensor.tape_value(tape, input) or_return
	mask := tensor.new_tensor(T, value.shape, tape.allocator) or_return
	defer tensor.release(&mask)
	kept := T(1 / (1 - layer.rate))
	generator := rand.default_random_generator(training.rng)
	for &m in mask.data {
		if rand.float64(generator) >= layer.rate {
			m = kept
		}
	}
	mask_leaf := tensor.tape_add_leaf_copy(tape, tensor.as_view(mask), false) or_return
	return tensor.tape_mul(tape, input, mask_leaf)
}

// _record_batch_norm records tensor.tape_batch_norm, which normalizes input
// with its own statistics over every axis but the last and applies the scale,
// then adds the shift. It updates the running statistics when training asks
// for it.
_record_batch_norm :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input, scale, shift: tensor.Tape_Value(T), training: _Training) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	features := len(layer.weights.data)
	statistics: []T
	if training.update_statistics {
		alloc_err: runtime.Allocator_Error
		if statistics, alloc_err = make([]T, 2*features, tape.allocator); alloc_err != nil {
			return output, .Allocation_Failed
		}
	}
	defer delete(statistics, tape.allocator)
	means, variances := statistics[:len(statistics)/2], statistics[len(statistics)/2:]
	normalized := tensor.tape_batch_norm(tape, input, scale, T(layer.epsilon), means if training.update_statistics else nil, variances if training.update_statistics else nil) or_return
	output = tensor.tape_add(tape, normalized, shift) or_return

	if training.update_statistics {
		value := tensor.tape_value(tape, input) or_return
		samples := 1
		for dim in value.shape {
			samples *= dim
		}
		samples /= features
		// The running variance is unbiased. One sample per feature has no
		// unbiased estimate, so its biased variance, zero, is used.
		correction := samples > 1 ? T(samples) / T(samples-1) : 1
		momentum := T(layer.momentum)
		for f in 0..<features {
			layer.running_mean.data[f] += momentum * (means[f] - layer.running_mean.data[f])
			layer.running_variance.data[f] += momentum * (variances[f]*correction - layer.running_variance.data[f])
		}
	}
	return output, nil
}
