package ann

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:testing"
import tensor "../odin-tensor"

require_ok :: proc(t: ^testing.T, err: Error, loc := #caller_location) {
	if err != nil {
		testing.fail_now(t, fmt.tprintf("expected no error, got %v", err), loc)
	}
}

expect_error :: proc(t: ^testing.T, actual, expected: Error, loc := #caller_location) {
	testing.expectf(t, actual == expected, "got error %v, want %v", actual, expected, loc = loc)
}

expect_close :: proc(t: ^testing.T, actual, expected: $T, tolerance: T, loc := #caller_location) {
	testing.expectf(t, abs(actual-expected) <= tolerance, "got %v, want %v (tolerance %v)", actual, expected, tolerance, loc = loc)
}

expect_shape :: proc(t: ^testing.T, actual, expected: []int, loc := #caller_location) {
	testing.expectf(t, _same_shape(actual, expected), "shape %v, want %v", actual, expected, loc = loc)
}

// set_layer overwrites a layer's parameters in row-major order.
set_layer :: proc(network: ^Network($T), layer: int, weights, bias: []T) {
	copy(network.layers[layer].weights.data, weights)
	copy(network.layers[layer].bias.data, bias)
}

@(test)
test_forward_accepts_vectors_batches_and_leading_axes :: proc(t: ^testing.T) {
	network := new_network(f32)
	defer release(&network)
	require_ok(t, dense(&network, 2, 1))
	// y = 2*x0 - x1 + 0.5
	set_layer(&network, 0, []f32{2, -1}, []f32{0.5})

	vector, vector_err := tensor.from_slice(f32, []f32{3, 2}, []int{2})
	require_ok(t, vector_err)
	defer tensor.release(&vector)
	output, output_err := forward(network, vector)
	require_ok(t, output_err)
	defer tensor.release(&output)
	expect_shape(t, output.shape, []int{1})
	testing.expect_value(t, output.data[0], 4.5)

	batch, batch_err := tensor.from_slice(f32, []f32{3, 2, 0, 4, 1, 1, -1, 0}, []int{2, 2, 2})
	require_ok(t, batch_err)
	defer tensor.release(&batch)
	batch_output, batch_output_err := forward(network, batch)
	require_ok(t, batch_output_err)
	defer tensor.release(&batch_output)
	expect_shape(t, batch_output.shape, []int{2, 2, 1})
	testing.expect_value(t, batch_output.data[0], 4.5)
	testing.expect_value(t, batch_output.data[1], -3.5)
	testing.expect_value(t, batch_output.data[2], 1.5)
	testing.expect_value(t, batch_output.data[3], -1.5)
}

@(test)
test_dense_validates_layers_and_inputs :: proc(t: ^testing.T) {
	network := new_network(f32)
	defer release(&network)
	input, input_err := tensor.new_tensor(f32, []int{4})
	require_ok(t, input_err)
	defer tensor.release(&input)

	_, empty_err := forward(network, input)
	expect_error(t, empty_err, Ann_Error.Empty_Network)
	require_ok(t, dense(&network, 2, 3, .ReLU))
	require_ok(t, dense(&network, 3, 1, .Sigmoid))
	expect_error(t, dense(&network, 4, 1), Tensor_Error.Shape_Mismatch)
	expect_error(t, dense(&network, 0, 1), Tensor_Error.Invalid_Shape)
	testing.expect_value(t, len(network.layers), 2)
	testing.expect_value(t, parameter_count(network), 2*3+3+3*1+1)
	testing.expect_value(t, input_features_of(network), 2)
	testing.expect_value(t, output_features_of(network), 1)
	expect_shape(t, _sample_dims(&network.input_shape), []int{2})
	expect_shape(t, _sample_dims(&network.layers[0].output_shape), []int{3})
	expect_shape(t, _sample_dims(&network.layers[1].output_shape), []int{1})
	testing.expect_value(t, _parameter_tensor_count(&network), 4)

	_, mismatch_err := forward(network, input)
	expect_error(t, mismatch_err, Tensor_Error.Shape_Mismatch)
	scalar, scalar_err := tensor.new_tensor(f32, []int{})
	require_ok(t, scalar_err)
	defer tensor.release(&scalar)
	_, scalar_forward_err := forward(network, scalar)
	expect_error(t, scalar_forward_err, Tensor_Error.Invalid_Shape)
}

@(test)
test_format_shape_joins_dims :: proc(t: ^testing.T) {
	testing.expect_value(t, _format_shape([]int{4}), "4")
	testing.expect_value(t, _format_shape([]int{28, 28, 1}), "28x28x1")
}

@(test)
test_seed_fixes_initial_weights :: proc(t: ^testing.T) {
	build :: proc(seed: u64) -> Network(f64) {
		network := new_network(f64, seed)
		_ = dense(&network, 8, 8, .ReLU)
		_ = dense(&network, 8, 2)
		return network
	}
	a, b, c := build(7), build(7), build(8)
	defer release(&a)
	defer release(&b)
	defer release(&c)
	same, different := true, false
	for weight, i in a.layers[0].weights.data {
		same &&= weight == b.layers[0].weights.data[i]
		different ||= weight != c.layers[0].weights.data[i]
	}
	testing.expect(t, same, "the same seed should give the same weights")
	testing.expect(t, different, "different seeds should give different weights")

	// He scaling for ReLU bounds weights by sqrt(6 / fan_in).
	limit := math.sqrt(6.0 / 8.0)
	for weight in a.layers[0].weights.data {
		testing.expectf(t, abs(weight) <= limit, "weight %v exceeds the He limit %v", weight, limit)
	}
}

@(test)
test_softmax_rows_are_stable_distributions :: proc(t: ^testing.T) {
	values, err := tensor.from_slice(f64, []f64{1, 2, 3, 1000, 1000, 1000}, []int{2, 3})
	require_ok(t, err)
	defer tensor.release(&values)
	require_ok(t, softmax_in_place(&values))
	denominator := 1 + math.exp(1.0) + math.exp(2.0)
	expect_close(t, values.data[0], 1/denominator, 1e-12)
	expect_close(t, values.data[2], math.exp(2.0)/denominator, 1e-12)
	for i in 3..<6 {
		expect_close(t, values.data[i], 1.0/3, 1e-12)
	}
}

// check_gradients compares the gradients train_step would apply against
// central differences of the training loss. A second trainer with the same
// seed measures the loss, so every pass draws the same dropout masks, and
// neither updates batch normalization's running statistics.
check_gradients :: proc(t: ^testing.T, network: ^Network(f64), loss: Loss, inputs, targets: tensor.Tensor(f64), loc := #caller_location) {
	trainer, trainer_err := trainer_init(f64, loss, sgd(0), seed = 9)
	require_ok(t, trainer_err, loc)
	defer trainer_release(&trainer)
	_, backward_err := _backward(&trainer, network^, tensor.as_view(inputs), tensor.as_view(targets), false)
	require_ok(t, backward_err, loc)
	probe, probe_err := trainer_init(f64, loss, sgd(0))
	require_ok(t, probe_err, loc)
	defer trainer_release(&probe)
	training_loss :: proc(probe: ^Trainer(f64), network: Network(f64), inputs, targets: tensor.Tensor(f64)) -> (f64, Error) {
		probe.rng = rand.create(9)
		return _backward(probe, network, tensor.as_view(inputs), tensor.as_view(targets), false)
	}

	step :: 1e-6
	slot := 0
	for &layer, i in network.layers {
		buffer: [2]^tensor.Tensor(f64)
		for parameter in _layer_parameters(&layer, &buffer) {
			defer slot += 1
			gradient, gradient_err := tensor.tape_gradient(&trainer.tape, trainer.parameters[slot])
			require_ok(t, gradient_err, loc)
			for j in 0..<len(parameter.data) {
				saved := parameter.data[j]
				parameter.data[j] = saved + step
				plus, plus_err := training_loss(&probe, network^, inputs, targets)
				parameter.data[j] = saved - step
				minus, minus_err := training_loss(&probe, network^, inputs, targets)
				parameter.data[j] = saved
				require_ok(t, plus_err, loc)
				require_ok(t, minus_err, loc)
				numeric := (plus - minus) / (2*step)
				analytic := gradient.data[gradient.offset+j]
				testing.expectf(t, abs(analytic-numeric) <= 1e-6*(1+abs(numeric)), "layer %d parameter %d[%d]: analytic %v, numeric %v", i, slot, j, analytic, numeric, loc = loc)
			}
		}
	}
}

// pseudo_values fills count deterministic values in [-1, 1].
pseudo_values :: proc(count: int, salt: f64) -> []f64 {
	values := make([]f64, count, context.temp_allocator)
	for &value, i in values {
		value = math.sin(f64(i)*1.7 + salt)
	}
	return values
}

@(test)
test_gradients_match_finite_differences_for_every_loss :: proc(t: ^testing.T) {
	Case :: struct {
		loss:          Loss,
		widths:        []int,
		activations:   []Activation,
		input_shape:   []int,
		target_shape:  []int,
		targets:       []f64,
	}
	cases := []Case{
		{.Mean_Squared_Error, {3, 4, 2}, {.Tanh, .Sigmoid}, {5, 3}, {5, 2}, nil},
		// A hidden Softmax is recorded on the tape rather than fused.
		{.Mean_Squared_Error, {3, 4, 2}, {.Softmax, .Identity}, {5, 3}, {5, 2}, nil},
		{.Binary_Cross_Entropy, {3, 4, 1}, {.Tanh, .Sigmoid}, {4, 3}, {4, 1}, {0, 1, 1, 0}},
		{.Cross_Entropy, {3, 5, 3}, {.Tanh, .Softmax}, {2, 2, 3}, {2, 2, 3}, {1, 0, 0, 0, 0.5, 0.5, 0, 0, 1, 0.2, 0.3, 0.5}},
		{.Cross_Entropy, {2, 3}, {.Softmax}, {2}, {3}, {0, 1, 0}},
	}
	for c, case_index in cases {
		network := new_network(f64, u64(case_index))
		defer release(&network)
		for activation, i in c.activations {
			require_ok(t, dense(&network, c.widths[i], c.widths[i+1], activation))
		}
		// Nonzero biases keep every pre-activation away from zero.
		for &layer in network.layers {
			copy(layer.bias.data, pseudo_values(len(layer.bias.data), 0.3))
		}
		input_count, target_count := 1, 1
		for dim in c.input_shape {
			input_count *= dim
		}
		for dim in c.target_shape {
			target_count *= dim
		}
		inputs, input_err := tensor.from_slice(f64, pseudo_values(input_count, 1), c.input_shape)
		require_ok(t, input_err)
		defer tensor.release(&inputs)
		target_values := c.targets
		if target_values == nil {
			target_values = pseudo_values(target_count, 2)
		}
		targets, target_err := tensor.from_slice(f64, target_values, c.target_shape)
		require_ok(t, target_err)
		defer tensor.release(&targets)
		check_gradients(t, &network, c.loss, inputs, targets)
	}
}

@(test)
test_cross_entropy_losses_match_their_definitions :: proc(t: ^testing.T) {
	logits, logits_err := tensor.from_slice(f64, []f64{2, -1, 0.5, 40, -40, 0}, []int{2, 3})
	require_ok(t, logits_err)
	defer tensor.release(&logits)
	labels, labels_err := one_hot(f64, []int{0, 1}, 3)
	require_ok(t, labels_err)
	defer tensor.release(&labels)

	value, gradient, err := _loss_and_gradient(.Cross_Entropy, tensor.as_view(logits), tensor.as_view(labels), false)
	require_ok(t, err)
	testing.expect_value(t, len(gradient.data), 0)
	first := -math.ln(math.exp(2.0) / (math.exp(2.0) + math.exp(-1.0) + math.exp(0.5)))
	// The second row's target logit is 80 below the largest; the loss is
	// finite where a naive softmax would take log(0).
	second := 80 + math.ln(1 + math.exp(-40.0) + math.exp(-80.0))
	expect_close(t, value, (first+second)/2, 1e-9)

	binary_logits, binary_err := tensor.from_slice(f64, []f64{0, 50, -50}, []int{3, 1})
	require_ok(t, binary_err)
	defer tensor.release(&binary_logits)
	binary_targets, binary_targets_err := tensor.from_slice(f64, []f64{1, 1, 1}, []int{3, 1})
	require_ok(t, binary_targets_err)
	defer tensor.release(&binary_targets)
	binary_value, binary_gradient, binary_value_err := _loss_and_gradient(.Binary_Cross_Entropy, tensor.as_view(binary_logits), tensor.as_view(binary_targets), true)
	require_ok(t, binary_value_err)
	defer tensor.release(&binary_gradient)
	expect_close(t, binary_value, (math.ln(2.0) + 0 + 50)/3, 1e-9)
	expect_close(t, binary_gradient.data[0], -0.5/3, 1e-12)
	expect_close(t, binary_gradient.data[2], -1.0/3, 1e-12)
}

@(test)
test_sgd_learns_a_linear_mapping :: proc(t: ^testing.T) {
	for momentum in ([]f64{0, 0.9}) {
		network := new_network(f32)
		defer release(&network)
		require_ok(t, dense(&network, 1, 1))
		trainer, trainer_err := trainer_init(f32, .Mean_Squared_Error, sgd(0.05, momentum))
		require_ok(t, trainer_err)
		defer trainer_release(&trainer)

		inputs, input_err := tensor.from_slice(f32, []f32{-1, 0, 1, 2}, []int{4, 1})
		require_ok(t, input_err)
		defer tensor.release(&inputs)
		targets, target_err := tensor.from_slice(f32, []f32{-1, 1, 3, 5}, []int{4, 1})
		require_ok(t, target_err)
		defer tensor.release(&targets)

		initial_loss, initial_err := evaluate(network, .Mean_Squared_Error, inputs, targets)
		require_ok(t, initial_err)
		first_loss, first_err := train_step(&trainer, &network, inputs, targets)
		require_ok(t, first_err)
		expect_close(t, first_loss, initial_loss, 1e-6)
		for _ in 0..<199 {
			_, step_err := train_step(&trainer, &network, inputs, targets)
			require_ok(t, step_err)
		}
		expect_close(t, network.layers[0].weights.data[0], 2, 1e-3)
		expect_close(t, network.layers[0].bias.data[0], 1, 1e-3)
		testing.expect_value(t, trainer.steps, 200)
	}
}

@(test)
test_adam_learns_xor_with_binary_cross_entropy :: proc(t: ^testing.T) {
	network := new_network(f64, 3)
	defer release(&network)
	require_ok(t, dense(&network, 2, 8, .Tanh))
	require_ok(t, dense(&network, 8, 1, .Sigmoid))
	trainer, trainer_err := trainer_init(f64, .Binary_Cross_Entropy, adam(0.05))
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)

	inputs, input_err := tensor.from_slice(f64, []f64{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	require_ok(t, input_err)
	defer tensor.release(&inputs)
	targets, target_err := tensor.from_slice(f64, []f64{0, 1, 1, 0}, []int{4, 1})
	require_ok(t, target_err)
	defer tensor.release(&targets)

	loss: f64
	for _ in 0..<300 {
		step_err: Error
		loss, step_err = train_step(&trainer, &network, inputs, targets)
		require_ok(t, step_err)
	}
	testing.expectf(t, loss < 0.05, "XOR loss %v after training", loss)
	predictions, forward_err := forward(network, inputs)
	require_ok(t, forward_err)
	defer tensor.release(&predictions)
	fraction, accuracy_err := accuracy(predictions, targets)
	require_ok(t, accuracy_err)
	testing.expect_value(t, fraction, 1.0)
}

// blobs builds three well separated 2D clusters with one_hot labels.
blobs :: proc(t: ^testing.T, samples_per_class: int) -> (inputs, targets: tensor.Tensor(f32)) {
	centers := [3][2]f32{{-2, 0}, {2, 0}, {0, 2.5}}
	values := make([]f32, 3*samples_per_class*2, context.temp_allocator)
	labels := make([]int, 3*samples_per_class, context.temp_allocator)
	for i in 0..<len(labels) {
		class := i % 3
		labels[i] = class
		values[2*i] = centers[class][0] + 0.5*f32(math.sin(f64(i)*12.9898))
		values[2*i+1] = centers[class][1] + 0.5*f32(math.sin(f64(i)*78.233))
	}
	err: Error
	inputs, err = tensor.from_slice(f32, values, []int{len(labels), 2})
	require_ok(t, err)
	targets, err = one_hot(f32, labels, 3)
	require_ok(t, err)
	return
}

@(test)
test_fit_classifies_with_shuffled_minibatches :: proc(t: ^testing.T) {
	inputs, targets := blobs(t, 40)
	defer tensor.release(&inputs)
	defer tensor.release(&targets)
	network := new_network(f32, 1)
	defer release(&network)
	require_ok(t, dense(&network, 2, 16, .ReLU))
	require_ok(t, dense(&network, 16, 3, .Softmax))
	trainer, trainer_err := trainer_init(f32, .Cross_Entropy, adam(0.01))
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)

	initial_loss, initial_err := evaluate(network, .Cross_Entropy, inputs, targets)
	require_ok(t, initial_err)
	final_loss, fit_err := fit(&trainer, &network, inputs, targets, {epochs = 40, batch_size = 16, shuffle = true, seed = 5})
	require_ok(t, fit_err)
	testing.expectf(t, final_loss < f64(initial_loss)*0.1, "loss went from %v to %v", initial_loss, final_loss)
	testing.expect_value(t, trainer.steps, 40*8)

	predictions, forward_err := forward(network, inputs)
	require_ok(t, forward_err)
	defer tensor.release(&predictions)
	fraction, accuracy_err := accuracy(predictions, targets)
	require_ok(t, accuracy_err)
	testing.expectf(t, fraction >= 0.99, "accuracy %v", fraction)

	classes, classes_err := predict_classes(network, inputs)
	require_ok(t, classes_err)
	defer tensor.release(&classes)
	expect_shape(t, classes.shape, []int{120})
}

@(test)
test_ordered_fit_matches_manual_steps_and_stops_early :: proc(t: ^testing.T) {
	inputs, targets := blobs(t, 3)
	defer tensor.release(&inputs)
	defer tensor.release(&targets)
	build :: proc(t: ^testing.T) -> Network(f32) {
		network := new_network(f32, 9)
		require_ok(t, dense(&network, 2, 4, .Tanh))
		require_ok(t, dense(&network, 4, 3, .Softmax))
		return network
	}
	fitted, manual := build(t), build(t)
	defer release(&fitted)
	defer release(&manual)
	fit_trainer, fit_trainer_err := trainer_init(f32, .Cross_Entropy, sgd(0.1, 0.5))
	require_ok(t, fit_trainer_err)
	defer trainer_release(&fit_trainer)
	manual_trainer, manual_trainer_err := trainer_init(f32, .Cross_Entropy, sgd(0.1, 0.5))
	require_ok(t, manual_trainer_err)
	defer trainer_release(&manual_trainer)

	Progress :: struct {
		epochs: int,
		losses: [dynamic]f64,
	}
	progress: Progress
	progress.losses = make([dynamic]f64, context.temp_allocator)
	stop_after_two :: proc(epoch: int, loss: f64, user_data: rawptr) -> bool {
		progress := (^Progress)(user_data)
		progress.epochs += 1
		append(&progress.losses, loss)
		return epoch < 1
	}
	// Nine samples in batches of four: 4, 4, then 1.
	_, fit_err := fit(&fit_trainer, &fitted, inputs, targets, {epochs = 10, batch_size = 4, on_epoch = stop_after_two, user_data = &progress})
	require_ok(t, fit_err)
	testing.expect_value(t, progress.epochs, 2)

	for _ in 0..<2 {
		for start := 0; start < 9; start += 4 {
			count := min(4, 9-start)
			batch_inputs, batch_input_err := tensor.from_slice(f32, inputs.data[2*start:][:2*count], []int{count, 2})
			require_ok(t, batch_input_err)
			defer tensor.release(&batch_inputs)
			batch_targets, batch_target_err := tensor.from_slice(f32, targets.data[3*start:][:3*count], []int{count, 3})
			require_ok(t, batch_target_err)
			defer tensor.release(&batch_targets)
			_, step_err := train_step(&manual_trainer, &manual, batch_inputs, batch_targets)
			require_ok(t, step_err)
		}
	}
	for layer, i in fitted.layers {
		for weight, j in layer.weights.data {
			testing.expect_value(t, weight, manual.layers[i].weights.data[j])
		}
		for bias, j in layer.bias.data {
			testing.expect_value(t, bias, manual.layers[i].bias.data[j])
		}
	}
}

@(test)
test_training_rejects_invalid_setups :: proc(t: ^testing.T) {
	_, negative_err := trainer_init(f32, .Mean_Squared_Error, sgd(-0.1))
	expect_error(t, negative_err, Ann_Error.Invalid_Hyperparameter)
	_, momentum_err := trainer_init(f32, .Mean_Squared_Error, sgd(0.1, 1))
	expect_error(t, momentum_err, Ann_Error.Invalid_Hyperparameter)
	_, beta_err := trainer_init(f32, .Mean_Squared_Error, adam(beta1 = 1))
	expect_error(t, beta_err, Ann_Error.Invalid_Hyperparameter)
	_, missing_err := trainer_init(f32, .Mean_Squared_Error, nil)
	expect_error(t, missing_err, Ann_Error.Invalid_Hyperparameter)

	network := new_network(f32)
	defer release(&network)
	require_ok(t, dense(&network, 2, 2, .Tanh))
	inputs, input_err := tensor.from_slice(f32, []f32{1, 2}, []int{1, 2})
	require_ok(t, input_err)
	defer tensor.release(&inputs)
	targets, target_err := tensor.from_slice(f32, []f32{0, 1}, []int{1, 2})
	require_ok(t, target_err)
	defer tensor.release(&targets)
	wrong_targets, wrong_target_err := tensor.from_slice(f32, []f32{0, 1}, []int{2, 1})
	require_ok(t, wrong_target_err)
	defer tensor.release(&wrong_targets)

	cross_entropy, cross_entropy_err := trainer_init(f32, .Cross_Entropy, sgd(0.1))
	require_ok(t, cross_entropy_err)
	defer trainer_release(&cross_entropy)
	_, activation_err := train_step(&cross_entropy, &network, inputs, targets)
	expect_error(t, activation_err, Ann_Error.Loss_Activation_Mismatch)

	trainer, trainer_err := trainer_init(f32, .Mean_Squared_Error, adam())
	require_ok(t, trainer_err)
	defer trainer_release(&trainer)
	_, shape_err := train_step(&trainer, &network, inputs, wrong_targets)
	expect_error(t, shape_err, Tensor_Error.Shape_Mismatch)
	testing.expect_value(t, trainer.steps, 0)
	_, step_err := train_step(&trainer, &network, inputs, targets)
	require_ok(t, step_err)
	require_ok(t, set_learning_rate(&trainer, 0.5))
	expect_error(t, set_learning_rate(&trainer, -1), Ann_Error.Invalid_Hyperparameter)
	testing.expect_value(t, trainer.optimizer.(Adam).learning_rate, 0.5)

	// Adam's moments were sized for one layer.
	require_ok(t, dense(&network, 2, 2))
	_, changed_err := train_step(&trainer, &network, inputs, targets)
	expect_error(t, changed_err, Ann_Error.Network_Changed)

	_, epochs_err := fit(&trainer, &network, inputs, targets, {})
	expect_error(t, epochs_err, Ann_Error.Invalid_Hyperparameter)
}

@(test)
test_one_hot_and_accuracy :: proc(t: ^testing.T) {
	labels, err := one_hot(f32, []int{2, 0}, 3)
	require_ok(t, err)
	defer tensor.release(&labels)
	expect_shape(t, labels.shape, []int{2, 3})
	testing.expect(t, labels.data[2] == 1 && labels.data[3] == 1 && labels.data[0] == 0)
	_, bounds_err := one_hot(f32, []int{3}, 3)
	expect_error(t, bounds_err, Tensor_Error.Out_Of_Bounds)

	predictions, prediction_err := tensor.from_slice(f32, []f32{0.1, 0.2, 0.7, 0.3, 0.6, 0.1}, []int{2, 3})
	require_ok(t, prediction_err)
	defer tensor.release(&predictions)
	fraction, accuracy_err := accuracy(predictions, labels)
	require_ok(t, accuracy_err)
	testing.expect_value(t, fraction, 0.5)
}

@(test)
test_mean_squared_error :: proc(t: ^testing.T) {
	predictions, prediction_err := tensor.from_slice(f32, []f32{1, 3}, []int{2})
	require_ok(t, prediction_err)
	defer tensor.release(&predictions)
	targets, target_err := tensor.from_slice(f32, []f32{2, 1}, []int{2})
	require_ok(t, target_err)
	defer tensor.release(&targets)
	loss, loss_err := mean_squared_error(predictions, targets)
	require_ok(t, loss_err)
	testing.expect_value(t, loss, 2.5)
}

// Failing_Allocator forwards to backing until remaining allocations run out,
// then reports Out_Of_Memory. Frees always reach backing so leaks stay visible.
Failing_Allocator :: struct {
	backing:   mem.Allocator,
	remaining: int,
}

failing_allocator_proc :: proc(allocator_data: rawptr, mode: mem.Allocator_Mode, size, alignment: int, old_memory: rawptr, old_size: int, location := #caller_location) -> ([]byte, mem.Allocator_Error) {
	state := (^Failing_Allocator)(allocator_data)
	#partial switch mode {
	case .Alloc, .Alloc_Non_Zeroed, .Resize, .Resize_Non_Zeroed:
		if state.remaining <= 0 {
			return nil, .Out_Of_Memory
		}
		state.remaining -= 1
	}
	return state.backing.procedure(state.backing.data, mode, size, alignment, old_memory, old_size, location)
}

// allocation_scenario builds, trains and evaluates a classifier with every
// allocation drawn from allocator.
allocation_scenario :: proc(allocator: mem.Allocator) -> Error {
	network := new_network(f64, 0, allocator)
	defer release(&network)
	dense(&network, 2, 3, .Tanh) or_return
	dense(&network, 3, 2, .Softmax) or_return
	inputs := tensor.from_slice(f64, []f64{1, 0, 0, 1, 1, 1}, []int{3, 2}, allocator) or_return
	defer tensor.release(&inputs)
	targets := one_hot(f64, []int{0, 1, 1}, 2, allocator) or_return
	defer tensor.release(&targets)

	trainer := trainer_init(f64, .Cross_Entropy, adam(), allocator) or_return
	defer trainer_release(&trainer)
	fit(&trainer, &network, inputs, targets, {epochs = 2, batch_size = 2, shuffle = true}) or_return
	evaluate(network, .Cross_Entropy, inputs, targets, allocator) or_return
	evaluate(network, .Mean_Squared_Error, inputs, targets, allocator) or_return
	predictions := forward(network, inputs, allocator) or_return
	defer tensor.release(&predictions)
	accuracy(predictions, targets, allocator) or_return
	return nil
}

@(test)
test_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	for budget in 0..<2048 {
		state := Failing_Allocator{backing = context.allocator, remaining = budget}
		err := allocation_scenario(mem.Allocator{procedure = failing_allocator_proc, data = &state})
		if err == nil {
			return
		}
		testing.expect_value(t, err, Error(Tensor_Error.Allocation_Failed))
	}
	testing.fail_now(t, "scenario never succeeded")
}
