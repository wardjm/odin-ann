package ann

import "core:math"
import "core:mem"
import "core:testing"
import tensor "../odin-tensor"

// pseudo_tensor builds a shape-sized tensor of pseudo_values.
pseudo_tensor :: proc(t: ^testing.T, shape: []int, salt: f64) -> tensor.Tensor(f64) {
	count := 1
	for dim in shape {
		count *= dim
	}
	result, err := tensor.from_slice(f64, pseudo_values(count, salt), shape)
	require_ok(t, err)
	return result
}

// expect_direct_convolution checks a 3x3 convolution of two 5x4x3 images
// against a direct sum over each window, skipping positions in the padding,
// and returns the network, inputs and output.
expect_direct_convolution :: proc(t: ^testing.T, stride, padding, out_height, out_width: int, loc := #caller_location) -> (Network(f64), tensor.Tensor(f64), tensor.Tensor(f64)) {
	network := new_network(f64)
	require_ok(t, set_input_shape(&network, []int{5, 4, 3}), loc)
	require_ok(t, conv2d(&network, 3, 2, 3, stride = stride, padding = padding), loc)
	layer := network.layers[0]
	expect_shape(t, _sample_dims(&layer.output_shape), []int{out_height, out_width, 2}, loc)
	copy(layer.weights.data, pseudo_values(len(layer.weights.data), 0.4))
	copy(layer.bias.data, []f64{0.25, -0.5})

	inputs := pseudo_tensor(t, []int{2, 5, 4, 3}, 1)
	output, err := forward(network, inputs)
	require_ok(t, err, loc)
	expect_shape(t, output.shape, []int{2, out_height, out_width, 2}, loc)

	for n in 0..<2 do for oy in 0..<out_height do for ox in 0..<out_width do for o in 0..<2 {
		expected := layer.bias.data[o]
		for dy in 0..<3 do for dx in 0..<3 do for c in 0..<3 {
			y, x := oy*stride+dy-padding, ox*stride+dx-padding
			if y < 0 || y >= 5 || x < 0 || x >= 4 {
				continue
			}
			input_value := inputs.data[((n*5 + y)*4 + x)*3 + c]
			weight := layer.weights.data[((dy*3 + dx)*3 + c)*2 + o]
			expected += input_value * weight
		}
		expect_close(t, output.data[((n*out_height + oy)*out_width + ox)*2 + o], expected, 1e-12, loc)
	}
	return network, inputs, output
}

@(test)
test_conv2d_matches_direct_convolution :: proc(t: ^testing.T) {
	network, inputs, output := expect_direct_convolution(t, 1, 0, 3, 2)
	defer release(&network)
	defer tensor.release(&inputs)
	defer tensor.release(&output)

	// Extra leading axes are batch axes.
	require_ok(t, tensor.reshape(&inputs, []int{1, 2, 5, 4, 3}))
	nested, nested_err := forward(network, inputs)
	require_ok(t, nested_err)
	defer tensor.release(&nested)
	expect_shape(t, nested.shape, []int{1, 2, 3, 2, 2})
	for value, i in nested.data {
		testing.expect_value(t, value, output.data[i])
	}
}

@(test)
test_conv2d_strides_and_pads :: proc(t: ^testing.T) {
	// Padding 1 keeps the size; with stride 2 the windows start in the
	// padding, and the last row's window is dropped.
	cases := [?][4]int{{1, 1, 5, 4}, {2, 1, 3, 2}, {2, 0, 2, 1}, {3, 2, 3, 2}, {1, 3, 9, 8}}
	for c in cases {
		network, inputs, output := expect_direct_convolution(t, c[0], c[1], c[2], c[3])
		release(&network)
		tensor.release(&inputs)
		tensor.release(&output)
	}

	network := new_network(f64)
	defer release(&network)
	require_ok(t, set_input_shape(&network, []int{4, 4, 1}))
	expect_error(t, conv2d(&network, 1, 2, 3, stride = 0), Tensor_Error.Invalid_Shape)
	expect_error(t, conv2d(&network, 1, 2, 3, padding = -1), Tensor_Error.Invalid_Shape)
	expect_error(t, conv2d(&network, 1, 2, 3, padding = max(int)/2), Tensor_Error.Invalid_Shape)
	expect_error(t, conv2d(&network, 1, 2, 7, padding = 1), Tensor_Error.Shape_Mismatch)
	// A kernel larger than the image fits once it is padded.
	require_ok(t, conv2d(&network, 1, 2, 6, padding = 1))
	expect_shape(t, _sample_dims(&network.layers[0].output_shape), []int{1, 1, 2})
}

@(test)
test_max_pool2d_takes_window_maxima_and_drops_remainders :: proc(t: ^testing.T) {
	network := new_network(f32)
	defer release(&network)
	require_ok(t, set_input_shape(&network, []int{5, 5, 1}))
	require_ok(t, max_pool2d(&network, 2))
	expect_shape(t, _sample_dims(&network.layers[0].output_shape), []int{2, 2, 1})
	testing.expect_value(t, parameter_count(network), 0)

	values: [25]f32
	for &value, i in values {
		value = f32(i)
	}
	// One negative maximum checks the pool doesn't clip at zero.
	values[0], values[1], values[5], values[6] = -4, -3, -2, -1
	inputs, input_err := tensor.from_slice(f32, values[:], []int{1, 5, 5, 1})
	require_ok(t, input_err)
	defer tensor.release(&inputs)
	output, err := forward(network, inputs)
	require_ok(t, err)
	defer tensor.release(&output)
	expect_shape(t, output.shape, []int{1, 2, 2, 1})
	testing.expect(t, output.data[0] == -1 && output.data[1] == 8 && output.data[2] == 16 && output.data[3] == 18, "unexpected pooled values")
}

@(test)
test_flatten_keeps_row_major_order :: proc(t: ^testing.T) {
	network := new_network(f32)
	defer release(&network)
	require_ok(t, set_input_shape(&network, []int{1, 2, 3}))
	require_ok(t, flatten(&network))
	testing.expect_value(t, output_features_of(network), 6)

	inputs, input_err := tensor.from_slice(f32, []f32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, []int{2, 1, 2, 3})
	require_ok(t, input_err)
	defer tensor.release(&inputs)
	output, err := forward(network, inputs)
	require_ok(t, err)
	defer tensor.release(&output)
	// The sample's leading 1 is flattened, not kept as a batch axis.
	expect_shape(t, output.shape, []int{2, 6})
	for value, i in output.data {
		testing.expect_value(t, value, f32(i))
	}
}

@(test)
test_image_layers_validate_shapes :: proc(t: ^testing.T) {
	network := new_network(f64)
	defer release(&network)
	expect_error(t, conv2d(&network, 1, 4, 3), Tensor_Error.Shape_Mismatch)
	expect_error(t, max_pool2d(&network), Tensor_Error.Shape_Mismatch)
	expect_error(t, flatten(&network), Tensor_Error.Shape_Mismatch)
	expect_error(t, set_input_shape(&network, []int{}), Tensor_Error.Invalid_Shape)
	expect_error(t, set_input_shape(&network, []int{8, 0, 1}), Tensor_Error.Invalid_Shape)
	expect_error(t, set_input_shape(&network, []int{1, 1, 1, 1, 1}), Tensor_Error.Invalid_Shape)

	require_ok(t, set_input_shape(&network, []int{8, 8, 1}))
	testing.expect_value(t, len(network.layers), 0)
	expect_error(t, dense(&network, 64, 4), Tensor_Error.Shape_Mismatch)
	expect_error(t, conv2d(&network, 2, 4, 3), Tensor_Error.Shape_Mismatch)
	expect_error(t, conv2d(&network, 1, 4, 9), Tensor_Error.Shape_Mismatch)
	expect_error(t, conv2d(&network, 1, 4, 0), Tensor_Error.Invalid_Shape)
	require_ok(t, conv2d(&network, 1, 4, 3, .ReLU))
	expect_error(t, set_input_shape(&network, []int{9, 9, 1}), Tensor_Error.Shape_Mismatch)
	require_ok(t, set_input_shape(&network, []int{8, 8, 1}))
	expect_error(t, dense(&network, 144, 4), Tensor_Error.Shape_Mismatch)
	expect_error(t, max_pool2d(&network, 7), Tensor_Error.Shape_Mismatch)
	expect_error(t, max_pool2d(&network, 0), Tensor_Error.Invalid_Shape)
	require_ok(t, max_pool2d(&network))
	require_ok(t, flatten(&network))
	require_ok(t, dense(&network, 36, 2, .Softmax))
	testing.expect_value(t, parameter_count(network), 3*3*1*4+4 + 36*2+2)
	testing.expect_value(t, _parameter_tensor_count(&network), 4)
	expect_shape(t, _sample_dims(&network.layers[1].output_shape), []int{3, 3, 4})

	// He scaling for ReLU bounds weights by sqrt(6 / fan_in), fan_in = 3*3*1.
	limit := math.sqrt(6.0 / 9.0)
	for weight in network.layers[0].weights.data {
		testing.expectf(t, abs(weight) <= limit, "weight %v exceeds the He limit %v", weight, limit)
	}

	wrong := pseudo_tensor(t, []int{2, 8, 8, 2}, 0)
	defer tensor.release(&wrong)
	_, wrong_err := forward(network, wrong)
	expect_error(t, wrong_err, Tensor_Error.Shape_Mismatch)
}

// build_image_classifier is a small network using every image layer: a 7x7
// input leaves a remainder row and column for the pool to drop, and the
// convolutions use padding and a stride.
build_image_classifier :: proc(t: ^testing.T, seed: u64) -> Network(f64) {
	network := new_network(f64, seed)
	require_ok(t, set_input_shape(&network, []int{7, 7, 2}))
	require_ok(t, conv2d(&network, 2, 3, 3, .Tanh, padding = 1))            // 7x7x3
	require_ok(t, max_pool2d(&network, 2))                                  // 3x3x3
	require_ok(t, conv2d(&network, 3, 2, 2, .Tanh, stride = 2, padding = 1)) // 2x2x2
	require_ok(t, flatten(&network))
	require_ok(t, dense(&network, 8, 3, .Softmax))
	for &layer in network.layers {
		copy(layer.bias.data, pseudo_values(len(layer.bias.data), 0.3))
	}
	return network
}

@(test)
test_image_layer_gradients_match_finite_differences :: proc(t: ^testing.T) {
	network := build_image_classifier(t, 4)
	defer release(&network)
	inputs := pseudo_tensor(t, []int{3, 7, 7, 2}, 1)
	defer tensor.release(&inputs)
	targets, target_err := one_hot(f64, []int{0, 2, 1}, 3)
	require_ok(t, target_err)
	defer tensor.release(&targets)
	check_gradients(t, &network, .Cross_Entropy, inputs, targets)

	// With leading batch axes and a loss that differentiates the softmax.
	require_ok(t, tensor.reshape(&inputs, []int{3, 1, 7, 7, 2}))
	require_ok(t, tensor.reshape(&targets, []int{3, 1, 3}))
	check_gradients(t, &network, .Mean_Squared_Error, inputs, targets)
}

@(test)
test_image_forward_matches_training_loss_exactly :: proc(t: ^testing.T) {
	network := build_image_classifier(t, 5)
	defer release(&network)
	inputs := pseudo_tensor(t, []int{4, 7, 7, 2}, 2)
	defer tensor.release(&inputs)
	targets, target_err := one_hot(f64, []int{1, 0, 2, 1}, 3)
	require_ok(t, target_err)
	defer tensor.release(&targets)

	evaluated, evaluate_err := evaluate(network, .Cross_Entropy, inputs, targets)
	require_ok(t, evaluate_err)
	trainer, trainer_err := trainer_init(f64, .Cross_Entropy, adam(0.01))
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)
	trained, step_err := train_step(&trainer, &network, inputs, targets)
	require_ok(t, step_err)
	testing.expect_value(t, trained, evaluated)
}

// stripes builds 6x6 single-channel images holding one vertical (class 0) or
// horizontal (class 1) bright line.
stripes :: proc(t: ^testing.T) -> (inputs, targets: tensor.Tensor(f32)) {
	SIDE :: 6
	values := make([]f32, 2*SIDE*SIDE*SIDE, context.temp_allocator)
	labels := make([]int, 2*SIDE, context.temp_allocator)
	for line in 0..<SIDE {
		for i in 0..<SIDE {
			values[(line*SIDE + i)*SIDE + line] = 1
			values[((SIDE+line)*SIDE + line)*SIDE + i] = 1
		}
		labels[SIDE+line] = 1
	}
	err: Error
	inputs, err = tensor.from_slice(f32, values, []int{2 * SIDE, SIDE, SIDE, 1})
	require_ok(t, err)
	targets, err = one_hot(f32, labels, 2)
	require_ok(t, err)
	return
}

@(test)
test_conv_network_learns_line_orientation :: proc(t: ^testing.T) {
	inputs, targets := stripes(t)
	defer tensor.release(&inputs)
	defer tensor.release(&targets)
	network := new_network(f32, 2)
	defer release(&network)
	require_ok(t, set_input_shape(&network, []int{6, 6, 1}))
	require_ok(t, conv2d(&network, 1, 4, 3, .ReLU))
	require_ok(t, max_pool2d(&network))
	require_ok(t, flatten(&network))
	require_ok(t, dense(&network, 16, 2, .Softmax))
	trainer, trainer_err := trainer_init(f32, .Cross_Entropy, adam(0.02))
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)

	_, fit_err := fit(&trainer, &network, inputs, targets, {epochs = 60, batch_size = 4, shuffle = true, seed = 1})
	require_ok(t, fit_err)
	predictions, forward_err := forward(network, inputs)
	require_ok(t, forward_err)
	defer tensor.release(&predictions)
	fraction, accuracy_err := accuracy(predictions, targets)
	require_ok(t, accuracy_err)
	testing.expect_value(t, fraction, 1.0)
}

image_allocation_scenario :: proc(allocator: mem.Allocator) -> Error {
	network := new_network(f64, 0, allocator)
	defer release(&network)
	set_input_shape(&network, []int{5, 5, 1}) or_return
	conv2d(&network, 1, 2, 2, .ReLU) or_return
	max_pool2d(&network) or_return
	flatten(&network) or_return
	dense(&network, 8, 2, .Softmax) or_return
	inputs := tensor.from_slice(f64, pseudo_values(3*25, 1), []int{3, 5, 5, 1}, allocator) or_return
	defer tensor.release(&inputs)
	targets := one_hot(f64, []int{0, 1, 1}, 2, allocator) or_return
	defer tensor.release(&targets)

	trainer := trainer_init(f64, .Cross_Entropy, adam(), allocator) or_return
	defer trainer_release(&trainer)
	fit(&trainer, &network, inputs, targets, {epochs = 2, batch_size = 2, shuffle = true}) or_return
	evaluate(network, .Cross_Entropy, inputs, targets, allocator) or_return
	predictions := forward(network, inputs, allocator) or_return
	defer tensor.release(&predictions)
	return nil
}

@(test)
test_image_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	for budget in 0..<8192 {
		state := Failing_Allocator{backing = context.allocator, remaining = budget}
		err := image_allocation_scenario(mem.Allocator{procedure = failing_allocator_proc, data = &state})
		if err == nil {
			return
		}
		testing.expect_value(t, err, Error(Tensor_Error.Allocation_Failed))
	}
	testing.fail_now(t, "scenario never succeeded")
}
