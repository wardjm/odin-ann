package ann

import "base:intrinsics"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import tensor "../odin-tensor"

Tensor_Error :: tensor.Tensor_Error

// Ann_Error covers failures that are about the model or training setup rather
// than tensor shapes or memory.
Ann_Error :: enum {
	None,
	// The network has no layers.
	Empty_Network,
	// A learning rate, momentum, decay, epoch count or similar is out of range.
	Invalid_Hyperparameter,
	// Binary_Cross_Entropy needs a final Sigmoid layer and Cross_Entropy a
	// final Softmax layer.
	Loss_Activation_Mismatch,
	// The network's parameter shapes no longer match the optimizer state the
	// trainer built on its first step.
	Network_Changed,
}

// Error is nil on success. Shape, bounds and allocation failures come from
// the tensor library as Tensor_Error values.
Error :: union #shared_nil {
	Tensor_Error,
	Ann_Error,
}

// Activation is applied after a dense layer's affine transform. Softmax
// normalizes over the last axis.
Activation :: enum {
	Identity,
	ReLU,
	Sigmoid,
	Tanh,
	Softmax,
}

Layer :: struct($T: typeid) {
	weights:    tensor.Tensor(T), // [input_features, output_features]
	bias:       tensor.Tensor(T), // [output_features]
	activation: Activation,
}

// Network is an ordered sequence of dense layers. It owns its parameters;
// treat Network values as move-only and call release once when finished.
Network :: struct($T: typeid) {
	layers:    [dynamic]Layer(T),
	// rng draws initial weights, so the same seed and layer sequence always
	// build the same network.
	rng:       rand.Default_Random_State,
	allocator: mem.Allocator,
}

// new_network creates an empty sequential network. seed fixes the initial
// weights drawn by dense.
new_network :: proc($T: typeid, seed: u64 = 0, allocator := context.allocator) -> Network(T) where intrinsics.type_is_float(T) {
	return Network(T){layers = make([dynamic]Layer(T), allocator), rng = rand.create(seed), allocator = allocator}
}

// release frees all layer parameters and the layer list.
release :: proc(network: ^Network($T)) {
	for &layer in network.layers {
		tensor.release(&layer.weights)
		tensor.release(&layer.bias)
	}
	delete(network.layers)
	network^ = {}
}

// dense appends a fully connected layer. Each layer's input size must equal
// the preceding layer's output size. Weights are drawn uniformly with He
// scaling for ReLU and Glorot scaling otherwise; biases start at zero. The
// parameters are ordinary tensors, so callers may overwrite them.
dense :: proc(network: ^Network($T), input_features, output_features: int, activation := Activation.Identity) -> Error where intrinsics.type_is_float(T) {
	if input_features <= 0 || output_features <= 0 {
		return .Invalid_Shape
	}
	if len(network.layers) > 0 && output_features_of(network^) != input_features {
		return .Shape_Mismatch
	}

	weights := tensor.new_tensor(T, []int{input_features, output_features}, network.allocator) or_return
	bias, err := tensor.new_tensor(T, []int{output_features}, network.allocator)
	if err != .None {
		tensor.release(&weights)
		return err
	}

	fan := f64(input_features + output_features) / 2
	if activation == .ReLU {
		fan = f64(input_features) / 2
	}
	limit := math.sqrt(3 / fan)
	generator := rand.default_random_generator(&network.rng)
	for &weight in weights.data {
		weight = T(rand.float64_range(-limit, limit, generator))
	}

	if _, alloc_err := append(&network.layers, Layer(T){weights = weights, bias = bias, activation = activation}); alloc_err != nil {
		tensor.release(&weights)
		tensor.release(&bias)
		return .Allocation_Failed
	}
	return nil
}

// input_features_of is the feature count the first layer expects, or zero for
// an empty network.
input_features_of :: proc(network: Network($T)) -> int {
	if len(network.layers) == 0 {
		return 0
	}
	return network.layers[0].weights.shape[0]
}

// output_features_of is the width of the last layer, or zero for an empty
// network.
output_features_of :: proc(network: Network($T)) -> int {
	if len(network.layers) == 0 {
		return 0
	}
	return network.layers[len(network.layers)-1].bias.shape[0]
}

// parameter_count is the total number of weights and biases.
parameter_count :: proc(network: Network($T)) -> int {
	count := 0
	for layer in network.layers {
		count += len(layer.weights.data) + len(layer.bias.data)
	}
	return count
}

// print_summary writes one line per layer and the parameter total to stdout.
print_summary :: proc(network: Network($T)) {
	_print_summary_header()
	for layer, i in network.layers {
		_print_summary_row(i, layer.weights.shape[0], layer.weights.shape[1], layer.activation)
	}
	_print_summary_total(parameter_count(network))
}

// The printing helpers are not generic, so core:fmt counts as used even when
// a program never instantiates print_summary.
_print_summary_header :: proc() {
	fmt.printf("%-6s %-14s %-10s %s\n", "layer", "shape", "activation", "parameters")
}

_print_summary_row :: proc(index, inputs, outputs: int, activation: Activation) {
	shape := fmt.tprintf("%d -> %d", inputs, outputs)
	fmt.printf("%-6d %-14s %-10v %d\n", index, shape, activation, inputs*outputs+outputs)
}

_print_summary_total :: proc(parameters: int) {
	fmt.printf("total parameters: %d\n", parameters)
}

// _check_input validates a [..., input_features] tensor for network.
_check_input :: proc(network: Network($T), shape: []int) -> Error {
	if len(network.layers) == 0 {
		return .Empty_Network
	}
	if len(shape) == 0 {
		return .Invalid_Shape
	}
	if shape[len(shape)-1] != input_features_of(network) {
		return .Shape_Mismatch
	}
	return nil
}
