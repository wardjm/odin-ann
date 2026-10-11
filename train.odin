package ann

import "base:intrinsics"
import "core:math"
import "core:math/rand"
import "core:mem"
import tensor "../odin-tensor"

// SGD is stochastic gradient descent with optional heavy-ball momentum.
SGD :: struct {
	learning_rate: f64,
	momentum:      f64, // in [0, 1); 0 disables the velocity buffers
	weight_decay:  f64, // decoupled: each step scales parameters by 1 - learning_rate*weight_decay
}

// Adam is Adam with bias correction. A nonzero weight_decay makes it AdamW.
Adam :: struct {
	learning_rate: f64,
	beta1:         f64,
	beta2:         f64,
	epsilon:       f64,
	weight_decay:  f64, // decoupled, as for SGD
}

Optimizer :: union {
	SGD,
	Adam,
}

sgd :: proc(learning_rate: f64, momentum := 0.0, weight_decay := 0.0) -> Optimizer {
	return SGD{learning_rate = learning_rate, momentum = momentum, weight_decay = weight_decay}
}

adam :: proc(learning_rate := 0.001, beta1 := 0.9, beta2 := 0.999, epsilon := 1e-8, weight_decay := 0.0) -> Optimizer {
	return Adam{learning_rate = learning_rate, beta1 = beta1, beta2 = beta2, epsilon = epsilon, weight_decay = weight_decay}
}

// Trainer holds what persists between training steps: the loss, the
// optimizer and its per-parameter state, and one autodiff tape that is reset
// and rebuilt each step. Treat it as move-only and call trainer_release once.
Trainer :: struct($T: typeid) {
	loss:          Loss,
	optimizer:     Optimizer,
	steps:         int,
	tape:          tensor.Tape(T),
	// parameters holds every trainable tensor's leaf on the current tape, in
	// layer order and _layer_parameters order within a layer.
	parameters:    [dynamic]tensor.Tape_Value(T),
	// first_moment is the SGD velocity or Adam's mean; second_moment is
	// Adam's uncentered variance. Both are allocated on the first step.
	first_moment:  [dynamic]tensor.Tensor(T),
	second_moment: [dynamic]tensor.Tensor(T),
	// rng draws dropout masks.
	rng:           rand.Default_Random_State,
	allocator:     mem.Allocator,
}

// _Training is what recording a layer for training needs beyond the layer:
// the generator for dropout masks, and whether batch normalization updates
// its running statistics.
_Training :: struct {
	rng:               ^rand.Default_Random_State,
	update_statistics: bool,
}

// trainer_init creates a trainer for loss and optimizer. seed starts the
// generator that draws dropout masks.
trainer_init :: proc($T: typeid, loss: Loss, optimizer: Optimizer, allocator := context.allocator, seed: u64 = 0) -> (trainer: Trainer(T), err: Error) where intrinsics.type_is_float(T) {
	_check_optimizer(optimizer) or_return
	trainer.tape = tensor.tape_init(T, 64, allocator) or_return
	trainer.rng = rand.create(seed)
	trainer.loss = loss
	trainer.optimizer = optimizer
	trainer.parameters = make([dynamic]tensor.Tape_Value(T), allocator)
	trainer.first_moment = make([dynamic]tensor.Tensor(T), allocator)
	trainer.second_moment = make([dynamic]tensor.Tensor(T), allocator)
	trainer.allocator = allocator
	return trainer, nil
}

trainer_release :: proc(trainer: ^Trainer($T)) {
	tensor.tape_release(&trainer.tape)
	delete(trainer.parameters)
	_release_optimizer_state(trainer)
	delete(trainer.first_moment)
	delete(trainer.second_moment)
	trainer^ = {}
}

// set_learning_rate changes the optimizer's step size, for schedules. The
// optimizer state is kept.
set_learning_rate :: proc(trainer: ^Trainer($T), learning_rate: f64) -> Error {
	if !_is_finite_non_negative(learning_rate) {
		return .Invalid_Hyperparameter
	}
	switch &optimizer in trainer.optimizer {
	case SGD:
		optimizer.learning_rate = learning_rate
	case Adam:
		optimizer.learning_rate = learning_rate
	}
	return nil
}

// train_step runs one forward and backward pass over a batch and updates every
// parameter. inputs is [..., input_features] and targets must have the shape
// of the network's output. Returns the loss measured before the update.
train_step :: proc(trainer: ^Trainer($T), network: ^Network(T), inputs, targets: tensor.Tensor(T)) -> (T, Error) where intrinsics.type_is_float(T) {
	return train_step_views(trainer, network, tensor.as_view(inputs), tensor.as_view(targets))
}

// train_step_views is train_step for borrowed views, such as minibatch slices.
train_step_views :: proc(trainer: ^Trainer($T), network: ^Network(T), inputs, targets: tensor.Tensor_View(T)) -> (loss: T, err: Error) where intrinsics.type_is_float(T) {
	loss = _backward(trainer, network^, inputs, targets, true) or_return
	_apply_optimizer(trainer, network) or_return
	return loss, nil
}

// evaluate measures loss without training. It runs the network as forward
// does, so for networks without Dropout or Batch_Norm layers it computes the
// loss exactly as train_step does.
evaluate :: proc(network: Network($T), loss: Loss, inputs, targets: tensor.Tensor(T), allocator := context.allocator) -> (T, Error) where intrinsics.type_is_float(T) {
	return evaluate_views(network, loss, tensor.as_view(inputs), tensor.as_view(targets), allocator)
}

evaluate_views :: proc(network: Network($T), loss: Loss, inputs, targets: tensor.Tensor_View(T), allocator := context.allocator) -> (value: T, err: Error) where intrinsics.type_is_float(T) {
	_check_loss(network, loss) or_return
	prediction := _forward(network, inputs, _loss_uses_logits(loss), allocator) or_return
	defer tensor.release(&prediction)
	gradient: tensor.Tensor(T)
	value, gradient = _loss_and_gradient(loss, tensor.as_view(prediction), targets, false, allocator) or_return
	return value, nil
}

// _backward rebuilds the tape for one batch and leaves each parameter's
// gradient on it. The loss itself is not recorded: its gradient seeds the
// backward pass at the network output, or at the final logits for losses that
// absorb the last activation. update_statistics lets batch normalization
// update its running statistics.
_backward :: proc(trainer: ^Trainer($T), network: Network(T), inputs, targets: tensor.Tensor_View(T), update_statistics: bool) -> (loss: T, err: Error) where intrinsics.type_is_float(T) {
	_check_loss(network, trainer.loss) or_return
	_check_input(network, inputs.shape) or_return
	tape := &trainer.tape
	tensor.tape_reset(tape)
	clear(&trainer.parameters)

	input := tensor.tape_add_leaf_view(tape, inputs, false) or_return
	training := _Training{rng = &trainer.rng, update_statistics = update_statistics}
	output := _record_forward(tape, network, input, &trainer.parameters, _loss_uses_logits(trainer.loss), training) or_return
	prediction := tensor.tape_value(tape, output) or_return
	gradient: tensor.Tensor(T)
	loss, gradient = _loss_and_gradient(trainer.loss, prediction, targets, true, trainer.allocator) or_return
	defer tensor.release(&gradient)
	tensor.tape_backward_with_seed(tape, output, gradient) or_return
	return loss, nil
}

// _record_forward records every layer and its activation. Parameters are
// borrowed leaves, appended to parameters in _layer_parameters order.
_record_forward :: proc(tape: ^tensor.Tape($T), network: Network(T), input: tensor.Tape_Value(T), parameters: ^[dynamic]tensor.Tape_Value(T), skip_last_activation: bool, training: _Training) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	output = input
	for &layer, i in network.layers {
		buffer: [2]^tensor.Tensor(T)
		first := len(parameters)
		for parameter in _layer_parameters(&layer, &buffer) {
			leaf := tensor.tape_add_leaf_view(tape, tensor.as_view(parameter^)) or_return
			if _, alloc_err := append(parameters, leaf); alloc_err != nil {
				return output, .Allocation_Failed
			}
		}
		output = _record_layer(tape, layer, output, parameters[first:], training) or_return
		if !(skip_last_activation && i == len(network.layers)-1) {
			output = _record_activation(tape, output, layer.activation) or_return
		}
	}
	return output, nil
}

// _record_layer records one layer before its activation. parameters are the
// layer's leaves on tape, in _layer_parameters order.
_record_layer :: proc(tape: ^tensor.Tape($T), layer: Layer(T), input: tensor.Tape_Value(T), parameters: []tensor.Tape_Value(T), training: _Training) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	switch layer.kind {
	case .Dense:
		// As in _layer_forward: a batch of vectors takes the fused affine map,
		// other ranks a broadcast product and addition.
		value := tensor.tape_value(tape, input) or_return
		if len(value.shape) == 2 {
			output = tensor.tape_linear(tape, input, parameters[0], parameters[1]) or_return
		} else {
			product := tensor.tape_matmul(tape, input, parameters[0]) or_return
			output = tensor.tape_add(tape, product, parameters[1]) or_return
		}
	case .Conv2D:
		output = _record_conv2d(tape, layer, input, parameters[0], parameters[1]) or_return
	case .Max_Pool2D:
		output = _record_max_pool2d(tape, layer, input) or_return
	case .Flatten:
		output = _record_flatten(tape, layer, input) or_return
	case .Dropout:
		output = _record_dropout(tape, layer, input, training) or_return
	case .Batch_Norm:
		output = _record_batch_norm(tape, layer, input, parameters[0], parameters[1], training) or_return
	case .Upsample2D:
		output = _record_upsample2d(tape, layer, input) or_return
	}
	return output, nil
}

_record_activation :: proc(tape: ^tensor.Tape($T), x: tensor.Tape_Value(T), activation: Activation) -> (output: tensor.Tape_Value(T), err: Error) where intrinsics.type_is_float(T) {
	switch activation {
	case .Identity:
		output = x
	case .ReLU:
		output = tensor.tape_relu(tape, x) or_return
	case .Sigmoid:
		output = tensor.tape_sigmoid(tape, x) or_return
	case .Tanh:
		output = tensor.tape_tanh(tape, x) or_return
	case .Softmax:
		// Softmax is unchanged by subtracting a per-row constant, so the row
		// maximum enters as a constant leaf with no gradient.
		x_value := tensor.tape_value(tape, x) or_return
		row_max := tensor.max_axis_views(x_value, -1, keep_dims = true, allocator = tape.allocator) or_return
		defer tensor.release(&row_max)
		shift := tensor.tape_add_leaf_copy(tape, tensor.as_view(row_max), false) or_return
		shifted := tensor.tape_sub(tape, x, shift) or_return
		exponent := tensor.tape_exp(tape, shifted) or_return
		row_sum := tensor.tape_sum_axis(tape, exponent, -1, keep_dims = true) or_return
		output = tensor.tape_div(tape, exponent, row_sum) or_return
	}
	return output, nil
}

// _apply_optimizer updates every parameter from the gradients _backward left
// on the tape. The tape borrows the parameters, but nothing reads them again
// before the next step resets it.
_apply_optimizer :: proc(trainer: ^Trainer($T), network: ^Network(T)) -> Error where intrinsics.type_is_float(T) {
	if len(trainer.parameters) != _parameter_tensor_count(network) {
		return .Network_Changed
	}
	_ensure_optimizer_state(trainer, network) or_return
	trainer.steps += 1
	slot := 0
	for &layer in network.layers {
		buffer: [2]^tensor.Tensor(T)
		for parameter in _layer_parameters(&layer, &buffer) {
			_update_parameter(trainer, parameter, slot) or_return
			slot += 1
		}
	}
	return nil
}

_update_parameter :: proc(trainer: ^Trainer($T), parameter: ^tensor.Tensor(T), slot: int) -> Error where intrinsics.type_is_float(T) {
	gradient_view := tensor.tape_gradient(&trainer.tape, trainer.parameters[slot]) or_return
	// Gradients are contiguous tensors shaped like their parameter.
	gradient := gradient_view.data[gradient_view.offset:][:len(parameter.data)]
	values := parameter.data
	switch optimizer in trainer.optimizer {
	case SGD:
		learning_rate := T(optimizer.learning_rate)
		if optimizer.weight_decay != 0 {
			tensor.scale_in_place(parameter, T(1 - optimizer.learning_rate*optimizer.weight_decay))
		}
		if optimizer.momentum == 0 {
			for g, j in gradient {
				values[j] -= learning_rate * g
			}
			break
		}
		momentum := T(optimizer.momentum)
		velocity := trainer.first_moment[slot].data
		for g, j in gradient {
			velocity[j] = momentum*velocity[j] + g
			values[j] -= learning_rate * velocity[j]
		}
	case Adam:
		if optimizer.weight_decay != 0 {
			tensor.scale_in_place(parameter, T(1 - optimizer.learning_rate*optimizer.weight_decay))
		}
		beta1, beta2 := T(optimizer.beta1), T(optimizer.beta2)
		step := f64(trainer.steps)
		step_size := T(optimizer.learning_rate / (1 - math.pow(optimizer.beta1, step)))
		variance_correction := T(1 - math.pow(optimizer.beta2, step))
		epsilon := T(optimizer.epsilon)
		mean := trainer.first_moment[slot].data
		variance := trainer.second_moment[slot].data
		for g, j in gradient {
			mean[j] = beta1*mean[j] + (1-beta1)*g
			variance[j] = beta2*variance[j] + (1-beta2)*g*g
			values[j] -= step_size * mean[j] / (math.sqrt(variance[j]/variance_correction) + epsilon)
		}
	}
	return nil
}

// _ensure_optimizer_state allocates zeroed moment buffers on the first step
// and afterwards checks they still match the network's parameters, slot by
// slot in _layer_parameters order.
_ensure_optimizer_state :: proc(trainer: ^Trainer($T), network: ^Network(T)) -> Error where intrinsics.type_is_float(T) {
	moments := 0
	switch optimizer in trainer.optimizer {
	case SGD:
		if optimizer.momentum != 0 {
			moments = 1
		}
	case Adam:
		moments = 2
	}
	if moments == 0 {
		return nil
	}
	if len(trainer.first_moment) > 0 {
		if len(trainer.first_moment) != _parameter_tensor_count(network) {
			return .Network_Changed
		}
		slot := 0
		for &layer in network.layers {
			buffer: [2]^tensor.Tensor(T)
			for parameter in _layer_parameters(&layer, &buffer) {
				if !_same_shape(trainer.first_moment[slot].shape, parameter.shape) {
					return .Network_Changed
				}
				slot += 1
			}
		}
		return nil
	}

	err := _allocate_moments(trainer, network, moments)
	if err != nil {
		_release_optimizer_state(trainer)
	}
	return err
}

_allocate_moments :: proc(trainer: ^Trainer($T), network: ^Network(T), moments: int) -> Error where intrinsics.type_is_float(T) {
	for &layer in network.layers {
		buffer: [2]^tensor.Tensor(T)
		for parameter in _layer_parameters(&layer, &buffer) {
			shape := parameter.shape
			first := tensor.new_tensor(T, shape, trainer.allocator) or_return
			if _, alloc_err := append(&trainer.first_moment, first); alloc_err != nil {
				tensor.release(&first)
				return .Allocation_Failed
			}
			if moments < 2 {
				continue
			}
			second := tensor.new_tensor(T, shape, trainer.allocator) or_return
			if _, alloc_err := append(&trainer.second_moment, second); alloc_err != nil {
				tensor.release(&second)
				return .Allocation_Failed
			}
		}
	}
	return nil
}

_release_optimizer_state :: proc(trainer: ^Trainer($T)) {
	for &moment in trainer.first_moment {
		tensor.release(&moment)
	}
	for &moment in trainer.second_moment {
		tensor.release(&moment)
	}
	clear(&trainer.first_moment)
	clear(&trainer.second_moment)
}

_check_optimizer :: proc(optimizer: Optimizer) -> Error {
	switch o in optimizer {
	case SGD:
		if !_is_finite_non_negative(o.learning_rate) || !_is_finite_non_negative(o.weight_decay) || !(o.momentum >= 0 && o.momentum < 1) {
			return .Invalid_Hyperparameter
		}
	case Adam:
		if !_is_finite_non_negative(o.learning_rate) || !_is_finite_non_negative(o.weight_decay) || !(o.beta1 >= 0 && o.beta1 < 1) || !(o.beta2 >= 0 && o.beta2 < 1) || !(o.epsilon > 0) || math.is_inf(o.epsilon) {
			return .Invalid_Hyperparameter
		}
	case:
		return .Invalid_Hyperparameter
	}
	return nil
}

_is_finite_non_negative :: proc(value: f64) -> bool {
	return value >= 0 && !math.is_inf(value)
}

// Fit_Options configures fit. Leave batch_size at zero to train on the whole
// dataset each step.
Fit_Options :: struct {
	epochs:     int,
	batch_size: int,
	// shuffle draws a new sample order each epoch from seed; without it,
	// batches are zero-copy slices of the dataset in order.
	shuffle:    bool,
	seed:       u64,
	// on_epoch, when set, runs after each epoch with the sample-weighted mean
	// training loss of that epoch. Returning false stops training early. It
	// may change the values, but not the shapes, of the inputs and targets
	// passed to fit; the next epoch trains on the new values, which is how to
	// draw fresh noise or augmentation each epoch.
	on_epoch:   proc(epoch: int, loss: f64, user_data: rawptr) -> bool,
	user_data:  rawptr,
}

// fit trains on a dataset whose first axis indexes samples: inputs is
// [samples, ..., input_features] and targets [samples, ..., output_features].
// Returns the mean training loss of the last epoch run.
fit :: proc(trainer: ^Trainer($T), network: ^Network(T), inputs, targets: tensor.Tensor(T), options: Fit_Options) -> (epoch_loss: f64, err: Error) where intrinsics.type_is_float(T) {
	if options.epochs <= 0 || options.batch_size < 0 {
		return 0, .Invalid_Hyperparameter
	}
	if len(inputs.shape) < 2 || len(targets.shape) < 2 {
		return 0, .Invalid_Shape
	}
	samples := inputs.shape[0]
	if targets.shape[0] != samples {
		return 0, .Shape_Mismatch
	}
	if samples == 0 {
		return 0, .Empty_Reduction
	}
	batch_size := options.batch_size
	if batch_size == 0 || batch_size > samples {
		batch_size = samples
	}

	// Shuffled batches are gathered into these buffers; ordered batches view
	// the dataset directly.
	order: []int
	input_batch, target_batch: tensor.Tensor(T)
	defer {
		delete(order, trainer.allocator)
		tensor.release(&input_batch)
		tensor.release(&target_batch)
	}
	rng: rand.Default_Random_State
	if options.shuffle {
		rng = rand.create(options.seed)
		alloc_err: mem.Allocator_Error
		order, alloc_err = make([]int, samples, trainer.allocator)
		if alloc_err != nil {
			return 0, .Allocation_Failed
		}
		for &index, i in order {
			index = i
		}
		input_batch = _new_batch_buffer(inputs, batch_size, trainer.allocator) or_return
		target_batch = _new_batch_buffer(targets, batch_size, trainer.allocator) or_return
	}

	shape_buffer, strides_buffer: [tensor.MAX_RANK]int
	target_shape_buffer, target_strides_buffer: [tensor.MAX_RANK]int
	for epoch in 0..<options.epochs {
		input_source, target_source := tensor.as_view(inputs), tensor.as_view(targets)
		if options.shuffle {
			rand.shuffle(order, rand.default_random_generator(&rng))
			input_source, target_source = tensor.as_view(input_batch), tensor.as_view(target_batch)
		}
		total: f64
		for start := 0; start < samples; start += batch_size {
			count := min(batch_size, samples-start)
			first := start
			if options.shuffle {
				_gather_rows(&input_batch, inputs, order[start:][:count])
				_gather_rows(&target_batch, targets, order[start:][:count])
				first = 0
			}
			input_view := tensor.slice_view(input_source, 0, first, count, 1, shape_buffer[:], strides_buffer[:]) or_return
			target_view := tensor.slice_view(target_source, 0, first, count, 1, target_shape_buffer[:], target_strides_buffer[:]) or_return
			batch_loss := train_step_views(trainer, network, input_view, target_view) or_return
			total += f64(batch_loss) * f64(count)
		}
		epoch_loss = total / f64(samples)
		if options.on_epoch != nil && !options.on_epoch(epoch, epoch_loss, options.user_data) {
			break
		}
	}
	return epoch_loss, nil
}

// _new_batch_buffer allocates room for batch_size samples shaped like the
// samples of dataset.
_new_batch_buffer :: proc(dataset: tensor.Tensor($T), batch_size: int, allocator: mem.Allocator) -> (tensor.Tensor(T), Tensor_Error) {
	shape: [tensor.MAX_RANK]int
	copy(shape[:], dataset.shape)
	shape[0] = batch_size
	return tensor.new_tensor(T, shape[:len(dataset.shape)], allocator)
}

// _gather_rows copies the samples named by rows into the front of batch.
_gather_rows :: proc(batch: ^tensor.Tensor($T), dataset: tensor.Tensor(T), rows: []int) {
	width := dataset.strides[0]
	for row, i in rows {
		copy(batch.data[i*width:][:width], dataset.data[row*width:][:width])
	}
}
