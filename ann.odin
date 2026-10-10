package ann

import "base:intrinsics"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:strings"
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

// Layer_Kind is what a layer computes. Only Dense exists so far.
Layer_Kind :: enum {
	Dense,
}

// MAX_SAMPLE_RANK bounds the rank of one sample's shape, such as [features]
// or [height, width, channels].
MAX_SAMPLE_RANK :: 4

// Sample_Shape is the shape of one sample, without batch axes.
Sample_Shape :: struct {
	dims: [MAX_SAMPLE_RANK]int,
	rank: int,
}

Layer :: struct($T: typeid) {
	kind:         Layer_Kind,
	// weights and bias are empty for a kind without parameters.
	weights:      tensor.Tensor(T), // Dense: [input_features, output_features]
	bias:         tensor.Tensor(T), // Dense: [output_features]
	activation:   Activation,
	// output_shape is one sample's shape after this layer.
	output_shape: Sample_Shape,
}

// Network is an ordered sequence of layers. It owns its parameters; treat
// Network values as move-only and call release once when finished.
Network :: struct($T: typeid) {
	layers:      [dynamic]Layer(T),
	// input_shape is one sample's shape before the first layer, set when the
	// first layer is added.
	input_shape: Sample_Shape,
	// rng draws initial weights, so the same seed and layer sequence always
	// build the same network.
	rng:         rand.Default_Random_State,
	allocator:   mem.Allocator,
}

// new_network creates an empty sequential network. seed fixes the initial
// weights drawn by the layer builders.
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

// dense appends a fully connected layer. Its input must be a vector of
// input_features: either it is the first layer, or the preceding layer's
// output is [input_features]. Weights are drawn uniformly with He scaling for
// ReLU and Glorot scaling otherwise; biases start at zero. The parameters are
// ordinary tensors, so callers may overwrite them.
dense :: proc(network: ^Network($T), input_features, output_features: int, activation := Activation.Identity) -> Error where intrinsics.type_is_float(T) {
	if input_features <= 0 || output_features <= 0 {
		return .Invalid_Shape
	}
	input_shape := _sample_shape(input_features)
	if len(network.layers) > 0 && !_same_sample_shape(_output_shape(network^), input_shape) {
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
	_fill_uniform(network, &weights, math.sqrt(3 / fan))

	layer := Layer(T){kind = .Dense, weights = weights, bias = bias, activation = activation, output_shape = _sample_shape(output_features)}
	return _append_layer(network, layer, input_shape)
}

// _fill_uniform draws every element of t from [-limit, limit] with the
// network's generator.
_fill_uniform :: proc(network: ^Network($T), t: ^tensor.Tensor(T), limit: f64) {
	generator := rand.default_random_generator(&network.rng)
	for &value in t.data {
		value = T(rand.float64_range(-limit, limit, generator))
	}
}

// _append_layer takes ownership of layer's parameters, releasing them if the
// append fails. input_shape becomes the network's input shape when layer is
// the first.
_append_layer :: proc(network: ^Network($T), layer: Layer(T), input_shape: Sample_Shape) -> Error {
	layer := layer
	if _, alloc_err := append(&network.layers, layer); alloc_err != nil {
		tensor.release(&layer.weights)
		tensor.release(&layer.bias)
		return .Allocation_Failed
	}
	if len(network.layers) == 1 {
		network.input_shape = input_shape
	}
	return nil
}

// _layer_parameters lists layer's trainable tensors in the order the trainer
// records and updates them, using buffer for storage.
_layer_parameters :: proc(layer: ^Layer($T), buffer: ^[2]^tensor.Tensor(T)) -> []^tensor.Tensor(T) {
	switch layer.kind {
	case .Dense:
		buffer^ = {&layer.weights, &layer.bias}
		return buffer[:]
	}
	return nil
}

// _parameter_tensor_count is the number of trainable tensors in network.
_parameter_tensor_count :: proc(network: ^Network($T)) -> int {
	count := 0
	buffer: [2]^tensor.Tensor(T)
	for &layer in network.layers {
		count += len(_layer_parameters(&layer, &buffer))
	}
	return count
}

_sample_shape :: proc(dims: ..int) -> (shape: Sample_Shape) {
	shape.rank = len(dims)
	copy(shape.dims[:], dims)
	return shape
}

_sample_dims :: proc(shape: ^Sample_Shape) -> []int {
	return shape.dims[:shape.rank]
}

_same_sample_shape :: proc(a, b: Sample_Shape) -> bool {
	a, b := a, b
	return _same_shape(_sample_dims(&a), _sample_dims(&b))
}

// _layer_input_shape is one sample's shape as layer index receives it.
_layer_input_shape :: proc(network: Network($T), index: int) -> Sample_Shape {
	if index == 0 {
		return network.input_shape
	}
	return network.layers[index-1].output_shape
}

// _output_shape is one sample's shape after the last layer. The network must
// not be empty.
_output_shape :: proc(network: Network($T)) -> Sample_Shape {
	return network.layers[len(network.layers)-1].output_shape
}

// input_features_of is the size of the last axis of one input sample, or zero
// for an empty network.
input_features_of :: proc(network: Network($T)) -> int {
	if len(network.layers) == 0 {
		return 0
	}
	return network.input_shape.dims[network.input_shape.rank-1]
}

// output_features_of is the size of the last axis of one output sample, or
// zero for an empty network.
output_features_of :: proc(network: Network($T)) -> int {
	if len(network.layers) == 0 {
		return 0
	}
	shape := _output_shape(network)
	return shape.dims[shape.rank-1]
}

// parameter_count is the total number of trainable values.
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
		_print_summary_row(i, layer.kind, _layer_input_shape(network, i), layer.output_shape, layer.activation, len(layer.weights.data)+len(layer.bias.data))
	}
	_print_summary_total(parameter_count(network))
}

// The printing helpers are not generic, so core:fmt counts as used even when
// a program never instantiates print_summary.
_print_summary_header :: proc() {
	fmt.printf("%-6s %-8s %-22s %-10s %s\n", "layer", "kind", "shape", "activation", "parameters")
}

_print_summary_row :: proc(index: int, kind: Layer_Kind, input, output: Sample_Shape, activation: Activation, parameters: int) {
	input, output := input, output
	shape := fmt.tprintf("%s -> %s", _format_shape(_sample_dims(&input)), _format_shape(_sample_dims(&output)))
	fmt.printf("%-6d %-8v %-22s %-10v %d\n", index, kind, shape, activation, parameters)
}

_print_summary_total :: proc(parameters: int) {
	fmt.printf("total parameters: %d\n", parameters)
}

// _format_shape joins dims with "x", so [28, 28, 1] prints as 28x28x1.
_format_shape :: proc(dims: []int) -> string {
	builder := strings.builder_make(context.temp_allocator)
	for dim, i in dims {
		if i > 0 {
			strings.write_byte(&builder, 'x')
		}
		strings.write_int(&builder, dim)
	}
	return strings.to_string(builder)
}

// _check_input validates an input tensor for network: its trailing axes must
// be the network's input sample shape, and any leading axes are batch axes.
_check_input :: proc(network: Network($T), shape: []int) -> Error {
	if len(network.layers) == 0 {
		return .Empty_Network
	}
	if len(shape) == 0 {
		return .Invalid_Shape
	}
	sample := network.input_shape
	if len(shape) < sample.rank || !_same_shape(shape[len(shape)-sample.rank:], _sample_dims(&sample)) {
		return .Shape_Mismatch
	}
	return nil
}
