package ann

import "base:intrinsics"
import "core:math"
import tensor "../odin-tensor"

// Loss selects the training objective. Every loss averages over the batch.
Loss :: enum {
	// Mean over every element of (prediction - target)^2. Works with any
	// final activation.
	Mean_Squared_Error,
	// Binary cross-entropy for targets in [0, 1]. The final layer must use
	// Sigmoid; the loss is computed from its logits, so saturated outputs
	// still give finite losses and gradients.
	Binary_Cross_Entropy,
	// Categorical cross-entropy for target rows that are distributions over
	// the last axis, such as one_hot labels. The final layer must use Softmax;
	// the loss is computed from its logits with a log-sum-exp.
	Cross_Entropy,
}

// _loss_uses_logits reports whether loss absorbs the final activation and
// works from the last layer's pre-activation values.
_loss_uses_logits :: proc(loss: Loss) -> bool {
	return loss != .Mean_Squared_Error
}

// _check_loss rejects a cross-entropy loss whose matching activation is not
// on the final layer.
_check_loss :: proc(network: Network($T), loss: Loss) -> Error {
	if len(network.layers) == 0 {
		return .Empty_Network
	}
	last := network.layers[len(network.layers)-1].activation
	switch loss {
	case .Mean_Squared_Error:
	case .Binary_Cross_Entropy:
		if last != .Sigmoid {
			return .Loss_Activation_Mismatch
		}
	case .Cross_Entropy:
		if last != .Softmax {
			return .Loss_Activation_Mismatch
		}
	}
	return nil
}

// mean_squared_error computes the mean of the squared element-wise
// difference. Predictions and targets must have identical shapes.
mean_squared_error :: proc(predictions, targets: tensor.Tensor($T), allocator := context.allocator) -> (loss: T, err: Error) where intrinsics.type_is_float(T) {
	gradient: tensor.Tensor(T)
	loss, gradient = _loss_and_gradient(.Mean_Squared_Error, tensor.as_view(predictions), tensor.as_view(targets), false, allocator) or_return
	return loss, nil
}

// _loss_and_gradient computes loss for one batch and, when want_gradient is
// set, its gradient with respect to prediction. prediction holds the network
// output for Mean_Squared_Error and the final logits for the cross-entropy
// losses. Without want_gradient the returned gradient is empty.
_loss_and_gradient :: proc(loss: Loss, prediction, targets: tensor.Tensor_View($T), want_gradient: bool, allocator := context.allocator) -> (value: T, gradient: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	if !_same_shape(prediction.shape, targets.shape) {
		return value, gradient, .Shape_Mismatch
	}
	count := 1
	for dim in prediction.shape {
		count *= dim
	}
	if count == 0 {
		return value, gradient, .Empty_Reduction
	}
	defer if err != nil {
		tensor.release(&gradient)
	}

	switch loss {
	case .Mean_Squared_Error:
		gradient = tensor.sub_views(prediction, targets, allocator) or_return
		squared := tensor.mul(gradient, gradient, allocator) or_return
		defer tensor.release(&squared)
		mean := tensor.mean_all(squared, allocator) or_return
		value = mean.data[0]
		tensor.release(&mean)
		tensor.scale_in_place(&gradient, 2/T(count))

	case .Binary_Cross_Entropy:
		// Per element, max(z, 0) - z*y + log(1 + exp(-|z|)) equals
		// -y*log(sigmoid(z)) - (1-y)*log(1-sigmoid(z)) without overflowing,
		// and its derivative is sigmoid(z) - y.
		gradient = tensor.copy_view(prediction, allocator) or_return
		labels := tensor.copy_view(targets, allocator) or_return
		defer tensor.release(&labels)
		total: f64
		for logit, i in gradient.data {
			z, y := f64(logit), f64(labels.data[i])
			total += max(z, 0) - z*y + math.log1p(math.exp(-abs(z)))
		}
		value = T(total / f64(count))
		tensor.sigmoid_in_place(&gradient) or_return
		tensor.sub_in_place(&gradient, labels) or_return
		tensor.scale_in_place(&gradient, 1/T(count))

	case .Cross_Entropy:
		// log_softmax(z) = (z - max) - log(sum(exp(z - max))) per row, and the
		// gradient of -sum(y * log_softmax(z)) is softmax(z)*sum(y) - y.
		rows := count / prediction.shape[len(prediction.shape)-1]
		row_max := tensor.max_axis_views(prediction, -1, keep_dims = true, allocator = allocator) or_return
		defer tensor.release(&row_max)
		gradient = tensor.sub_views(prediction, tensor.as_view(row_max), allocator) or_return
		probabilities := tensor.exp(gradient, allocator) or_return
		defer tensor.release(&probabilities)
		row_sum := tensor.sum_axis(probabilities, -1, keep_dims = true, allocator = allocator) or_return
		defer tensor.release(&row_sum)
		log_sum := tensor.log(row_sum, allocator) or_return
		defer tensor.release(&log_sum)
		tensor.sub_in_place(&gradient, log_sum) or_return // log-probabilities

		weighted := tensor.mul_views(targets, tensor.as_view(gradient), allocator) or_return
		defer tensor.release(&weighted)
		total := tensor.sum_all(weighted, allocator = allocator) or_return
		value = -total.data[0] / T(rows)
		tensor.release(&total)
		if !want_gradient {
			break
		}

		tensor.div_in_place(&probabilities, row_sum) or_return
		target_mass := tensor.sum_axis_views(targets, -1, keep_dims = true, allocator = allocator) or_return
		defer tensor.release(&target_mass)
		tensor.mul_into(&gradient, probabilities, target_mass) or_return
		tensor.sub_views_into(&gradient, tensor.as_view(gradient), targets) or_return
		tensor.scale_in_place(&gradient, 1/T(rows))
	}
	if !want_gradient {
		tensor.release(&gradient)
	}
	return value, gradient, nil
}

_same_shape :: proc(a, b: []int) -> bool {
	if len(a) != len(b) {
		return false
	}
	for i in 0..<len(a) {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
