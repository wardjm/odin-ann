package ann

import "base:intrinsics"
import tensor "../odin-tensor"

// one_hot builds a [len(labels), classes] target tensor with a one in each
// row's label column. A label outside [0, classes) returns Out_Of_Bounds.
one_hot :: proc($T: typeid, labels: []int, classes: int, allocator := context.allocator) -> (result: tensor.Tensor(T), err: Error) where intrinsics.type_is_float(T) {
	if classes <= 0 {
		return result, .Invalid_Shape
	}
	for label in labels {
		if label < 0 || label >= classes {
			return result, .Out_Of_Bounds
		}
	}
	result = tensor.new_tensor(T, []int{len(labels), classes}, allocator) or_return
	for label, row in labels {
		result.data[row*classes+label] = 1
	}
	return result, nil
}

// accuracy is the fraction of samples whose predicted class matches the
// target's. With one output per sample, predictions and targets are
// thresholded at 0.5; otherwise the class is the argmax over the last axis.
accuracy :: proc(predictions, targets: tensor.Tensor($T), allocator := context.allocator) -> (fraction: f64, err: Error) where intrinsics.type_is_float(T) {
	if !_same_shape(predictions.shape, targets.shape) {
		return 0, .Shape_Mismatch
	}
	if len(predictions.shape) == 0 || len(predictions.data) == 0 {
		return 0, .Empty_Reduction
	}
	if predictions.shape[len(predictions.shape)-1] == 1 {
		correct := 0
		for prediction, i in predictions.data {
			if (prediction >= 0.5) == (targets.data[i] >= 0.5) {
				correct += 1
			}
		}
		return f64(correct) / f64(len(predictions.data)), nil
	}

	predicted := tensor.argmax_axis(predictions, -1, allocator = allocator) or_return
	defer tensor.release(&predicted)
	expected := tensor.argmax_axis(targets, -1, allocator = allocator) or_return
	defer tensor.release(&expected)
	correct := 0
	for class, i in predicted.data {
		if class == expected.data[i] {
			correct += 1
		}
	}
	return f64(correct) / f64(len(predicted.data)), nil
}

// predict_classes returns the argmax over the last axis of the network output
// for each sample, as a Tensor(int) of shape [...].
predict_classes :: proc(network: Network($T), inputs: tensor.Tensor(T), allocator := context.allocator) -> (classes: tensor.Tensor(int), err: Error) where intrinsics.type_is_float(T) {
	output := forward(network, inputs, allocator) or_return
	defer tensor.release(&output)
	classes = tensor.argmax_axis(output, -1, allocator = allocator) or_return
	return classes, nil
}
