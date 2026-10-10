package ann

import "core:math"
import "core:mem"
import "core:testing"
import tensor "../odin-tensor"

@(test)
test_dropout_masks_while_training_and_passes_through_otherwise :: proc(t: ^testing.T) {
	network := new_network(f64)
	defer release(&network)
	require_ok(t, set_input_shape(&network, []int{4000}))
	require_ok(t, dropout(&network, 0.25))
	inputs := pseudo_tensor(t, []int{4000}, 0.5)
	defer tensor.release(&inputs)

	output, forward_err := forward(network, inputs)
	require_ok(t, forward_err)
	defer tensor.release(&output)
	for value, i in output.data {
		testing.expect_value(t, value, inputs.data[i])
	}

	// Training zeroes about a quarter of the values and scales the rest.
	trainer, trainer_err := trainer_init(f64, .Mean_Squared_Error, sgd(0))
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)
	recorded := record_training(t, &trainer, network, inputs)
	dropped := 0
	for value, i in recorded.data[recorded.offset:][:4000] {
		if value == 0 {
			dropped += 1
		} else {
			expect_close(t, value, inputs.data[i]/0.75, 1e-12)
		}
	}
	expect_close(t, f64(dropped)/4000, 0.25, 0.03)

	expect_error(t, dropout(&network, 1), Ann_Error.Invalid_Hyperparameter)
	expect_error(t, dropout(&network, -0.1), Ann_Error.Invalid_Hyperparameter)
	empty := new_network(f32)
	defer release(&empty)
	expect_error(t, dropout(&empty, 0.5), Tensor_Error.Shape_Mismatch)
}

// record_training records network on trainer's tape as a training step
// would, updating batch statistics, and returns the output.
record_training :: proc(t: ^testing.T, trainer: ^Trainer(f64), network: Network(f64), inputs: tensor.Tensor(f64)) -> tensor.Tensor_View(f64) {
	tensor.tape_reset(&trainer.tape)
	clear(&trainer.parameters)
	input, input_err := tensor.tape_add_leaf_view(&trainer.tape, tensor.as_view(inputs), false)
	require_ok(t, input_err)
	output, record_err := _record_forward(&trainer.tape, network, input, &trainer.parameters, false, _Training{rng = &trainer.rng, update_statistics = true})
	require_ok(t, record_err)
	value, value_err := tensor.tape_value(&trainer.tape, output)
	require_ok(t, value_err)
	return value
}

@(test)
test_batch_norm_normalizes_batches_and_tracks_running_statistics :: proc(t: ^testing.T) {
	network := new_network(f64)
	defer release(&network)
	require_ok(t, set_input_shape(&network, []int{2, 3}))
	require_ok(t, batch_norm(&network, momentum = 0.5))
	layer := network.layers[0]
	expect_shape(t, layer.weights.shape, []int{3})

	// Four samples of 2x3 give eight values of each of the three features.
	inputs := pseudo_tensor(t, []int{4, 2, 3}, 0.2)
	defer tensor.release(&inputs)
	for &value, i in inputs.data {
		value = value*f64(1 + i%3) + f64(i%3)
	}
	mean, variance: [3]f64
	for value, i in inputs.data {
		mean[i%3] += value / 8
	}
	for value, i in inputs.data {
		variance[i%3] += (value - mean[i%3]) * (value - mean[i%3]) / 8
	}

	trainer, trainer_err := trainer_init(f64, .Mean_Squared_Error, sgd(0))
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)
	recorded := record_training(t, &trainer, network, inputs)
	expect_shape(t, recorded.shape, []int{4, 2, 3})
	for value, i in recorded.data[recorded.offset:][:24] {
		f := i % 3
		expect_close(t, value, (inputs.data[i]-mean[f])/math.sqrt(variance[f]+1e-5), 1e-12)
	}
	for f in 0..<3 {
		expect_close(t, layer.running_mean.data[f], 0.5*mean[f], 1e-12)
		expect_close(t, layer.running_variance.data[f], 0.5 + 0.5*variance[f]*8/7, 1e-12)
	}

	// Outside training the running statistics, scale and shift apply.
	copy(layer.weights.data, []f64{2, -1, 0.5})
	copy(layer.bias.data, []f64{0.1, 0.2, 0.3})
	output, forward_err := forward(network, inputs)
	require_ok(t, forward_err)
	defer tensor.release(&output)
	for value, i in output.data {
		f := i % 3
		want := (inputs.data[i]-layer.running_mean.data[f])/math.sqrt(layer.running_variance.data[f]+1e-5)*layer.weights.data[f] + layer.bias.data[f]
		expect_close(t, value, want, 1e-12)
	}

	expect_error(t, batch_norm(&network, momentum = 1.5), Ann_Error.Invalid_Hyperparameter)
	expect_error(t, batch_norm(&network, epsilon = 0), Ann_Error.Invalid_Hyperparameter)
	empty := new_network(f32)
	defer release(&empty)
	expect_error(t, batch_norm(&empty), Tensor_Error.Shape_Mismatch)
}

// build_regularized_classifier uses batch normalization after a convolution
// and a dense layer, and dropout between them.
build_regularized_classifier :: proc(t: ^testing.T, seed: u64) -> Network(f64) {
	network := new_network(f64, seed)
	require_ok(t, set_input_shape(&network, []int{5, 5, 2}))
	require_ok(t, conv2d(&network, 2, 3, 2))         // 4x4x3
	require_ok(t, batch_norm(&network, .Tanh))
	require_ok(t, max_pool2d(&network))              // 2x2x3
	require_ok(t, flatten(&network))                 // 12
	require_ok(t, dropout(&network, 0.3))
	require_ok(t, dense(&network, 12, 4))
	require_ok(t, batch_norm(&network, .ReLU))
	require_ok(t, dense(&network, 4, 3, .Softmax))
	for &layer in network.layers {
		copy(layer.weights.data, pseudo_values(len(layer.weights.data), 0.7))
		copy(layer.bias.data, pseudo_values(len(layer.bias.data), 0.3))
		copy(layer.running_mean.data, pseudo_values(len(layer.running_mean.data), 1.1))
		for &value, i in layer.running_variance.data {
			value = 0.5 + f64(i)
		}
	}
	return network
}

@(test)
test_regularized_gradients_match_finite_differences :: proc(t: ^testing.T) {
	network := build_regularized_classifier(t, 3)
	defer release(&network)
	inputs := pseudo_tensor(t, []int{4, 5, 5, 2}, 1)
	defer tensor.release(&inputs)
	targets, target_err := one_hot(f64, []int{0, 2, 1, 1}, 3)
	require_ok(t, target_err)
	defer tensor.release(&targets)
	check_gradients(t, &network, .Cross_Entropy, inputs, targets)
	check_gradients(t, &network, .Mean_Squared_Error, inputs, targets)
}

@(test)
test_regularized_network_round_trips_through_marshal :: proc(t: ^testing.T) {
	network := build_regularized_classifier(t, 4)
	defer release(&network)
	data, marshal_err := marshal(network)
	require_ok(t, marshal_err)
	defer delete(data)
	restored, unmarshal_err := unmarshal(f64, data)
	require_ok(t, unmarshal_err)
	defer release(&restored)
	expect_same_network(t, restored, network)
	for layer, i in restored.layers {
		want := network.layers[i]
		testing.expect_value(t, layer.rate, want.rate)
		testing.expect_value(t, layer.momentum, want.momentum)
		testing.expect_value(t, layer.epsilon, want.epsilon)
		for value, j in layer.running_mean.data {
			testing.expect_value(t, value, want.running_mean.data[j])
		}
		for value, j in layer.running_variance.data {
			testing.expect_value(t, value, want.running_variance.data[j])
		}
	}

	inputs := pseudo_tensor(t, []int{2, 5, 5, 2}, 2)
	defer tensor.release(&inputs)
	want, want_err := forward(network, inputs)
	require_ok(t, want_err)
	defer tensor.release(&want)
	got, got_err := forward(restored, inputs)
	require_ok(t, got_err)
	defer tensor.release(&got)
	for value, i in got.data {
		testing.expect_value(t, value, want.data[i])
	}

	for length in 0..<len(data) {
		truncated, err := unmarshal(f64, data[:length])
		expect_error(t, err, Ann_Error.Invalid_File)
		release(&truncated)
	}
}

@(test)
test_regularized_network_learns_blobs :: proc(t: ^testing.T) {
	inputs, targets := blobs(t, 40)
	defer tensor.release(&inputs)
	defer tensor.release(&targets)
	network := new_network(f32, seed = 5)
	defer release(&network)
	require_ok(t, dense(&network, 2, 16))
	require_ok(t, batch_norm(&network, .ReLU))
	require_ok(t, dropout(&network, 0.2))
	require_ok(t, dense(&network, 16, 3, .Softmax))

	trainer, trainer_err := trainer_init(f32, .Cross_Entropy, adam(0.02), seed = 6)
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)
	_, fit_err := fit(&trainer, &network, inputs, targets, {epochs = 60, batch_size = 16, shuffle = true, seed = 7})
	require_ok(t, fit_err)

	predictions, forward_err := forward(network, inputs)
	require_ok(t, forward_err)
	defer tensor.release(&predictions)
	fraction, accuracy_err := accuracy(predictions, targets)
	require_ok(t, accuracy_err)
	testing.expectf(t, fraction >= 0.95, "accuracy %v", fraction)
}

regularized_allocation_scenario :: proc(allocator: mem.Allocator) -> Error {
	network := new_network(f64, 0, allocator)
	defer release(&network)
	set_input_shape(&network, []int{4, 4, 1}) or_return
	conv2d(&network, 1, 2, 2) or_return
	batch_norm(&network, .ReLU) or_return
	flatten(&network) or_return
	dropout(&network, 0.5) or_return
	dense(&network, 18, 2, .Softmax) or_return
	inputs := tensor.from_slice(f64, pseudo_values(3*16, 1), []int{3, 4, 4, 1}, allocator) or_return
	defer tensor.release(&inputs)
	targets := one_hot(f64, []int{0, 1, 1}, 2, allocator) or_return
	defer tensor.release(&targets)

	trainer := trainer_init(f64, .Cross_Entropy, adam(), allocator) or_return
	defer trainer_release(&trainer)
	fit(&trainer, &network, inputs, targets, {epochs = 2, batch_size = 2, shuffle = true}) or_return
	evaluate(network, .Cross_Entropy, inputs, targets, allocator) or_return
	data := marshal(network, allocator) or_return
	defer delete(data, allocator)
	restored := unmarshal(f64, data, allocator = allocator) or_return
	release(&restored)
	return nil
}

@(test)
test_regularized_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	for budget in 0..<8192 {
		state := Failing_Allocator{backing = context.allocator, remaining = budget}
		err := regularized_allocation_scenario(mem.Allocator{procedure = failing_allocator_proc, data = &state})
		if err == nil {
			return
		}
		testing.expect_value(t, err, Error(Tensor_Error.Allocation_Failed))
	}
	testing.fail_now(t, "scenario never succeeded")
}
