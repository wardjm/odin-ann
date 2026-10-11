package ann

import "base:intrinsics"
import "base:runtime"
import "core:encoding/endian"
import "core:os"

// A saved network holds its architecture and parameters, so load rebuilds it
// without the code that built it. Everything is little-endian:
//
//     magic           8 bytes, "ODINANN" and a zero byte
//     version         u32, FILE_VERSION
//     float size      u32, 4 for f32 parameters or 8 for f64
//     input rank      u32, then one u32 per dimension; 0 if never set
//     layer count     u32, then for each layer:
//         kind        u32, a Layer_Kind value
//         activation  u32, an Activation value
//         Dense       u32 input_features, output_features
//         Conv2D      u32 input_channels, output_channels, kernel_size,
//                     stride, padding
//         Max_Pool2D  u32 pool_size
//         Flatten     nothing more
//         Dropout     f64 rate
//         Batch_Norm  u32 features, f64 momentum, f64 epsilon
//         Upsample2D  u32 factor
//         then the weights, bias, running mean and running variance that the
//         layer has, each in row-major order
//
// Layer_Kind and Activation are stored by value, so new members go at the end
// of each enum. Optimizer state, such as Adam's moments, is not saved.
FILE_MAGIC :: "ODINANN\x00"
FILE_VERSION :: 1

// save writes network to path, replacing any existing file.
save :: proc(network: Network($T), path: string) -> Error where T == f32 || T == f64 {
	data := marshal(network) or_return
	defer delete(data)
	return _write_file(path, data)
}

// load reads a network written by save. Parameters saved as f32 or f64 are
// converted to T. seed starts the generator that draws weights for any layers
// added afterwards. The caller releases the result.
load :: proc($T: typeid, path: string, seed: u64 = 0, allocator := context.allocator) -> (network: Network(T), err: Error) where T == f32 || T == f64 {
	data := _read_file(path, allocator) or_return
	defer delete(data, allocator)
	return unmarshal(T, data, seed, allocator)
}

// marshal encodes network as save writes it, into a new slice owned by the
// caller.
marshal :: proc(network: Network($T), allocator := context.allocator) -> (data: []byte, err: Error) where T == f32 || T == f64 {
	size := len(FILE_MAGIC) + 4*(4+network.input_shape.rank)
	for layer in network.layers {
		size += 8 + _layer_settings_size(layer.kind) + size_of(T)*_layer_value_count(layer)
	}
	data = _make_bytes(size, allocator) or_return

	writer := _Cursor{data = data}
	copy(data, FILE_MAGIC)
	writer.offset = len(FILE_MAGIC)
	_put_u32(&writer, FILE_VERSION)
	_put_u32(&writer, size_of(T))
	input_shape := network.input_shape
	_put_u32(&writer, input_shape.rank)
	for dim in _sample_dims(&input_shape) {
		_put_u32(&writer, dim)
	}
	_put_u32(&writer, len(network.layers))
	for layer in network.layers {
		_put_u32(&writer, int(layer.kind))
		_put_u32(&writer, int(layer.activation))
		switch layer.kind {
		case .Dense:
			_put_u32(&writer, layer.weights.shape[0])
			_put_u32(&writer, layer.weights.shape[1])
		case .Conv2D:
			_put_u32(&writer, layer.weights.shape[2])
			_put_u32(&writer, layer.weights.shape[3])
			_put_u32(&writer, layer.weights.shape[0])
			_put_u32(&writer, layer.stride)
			_put_u32(&writer, layer.padding)
		case .Max_Pool2D:
			_put_u32(&writer, layer.pool_size)
		case .Flatten:
		case .Dropout:
			_put_f64(&writer, layer.rate)
		case .Batch_Norm:
			_put_u32(&writer, len(layer.weights.data))
			_put_f64(&writer, layer.momentum)
			_put_f64(&writer, layer.epsilon)
		case .Upsample2D:
			_put_u32(&writer, layer.factor)
		}
		for values in _layer_values(layer) {
			_put_floats(&writer, values)
		}
	}
	assert(writer.offset == len(data))
	return data, nil
}

// unmarshal decodes a network that marshal or save produced. Malformed or
// truncated data returns Invalid_File, and data from a newer format returns
// Unsupported_Version. The caller releases the result.
unmarshal :: proc($T: typeid, data: []byte, seed: u64 = 0, allocator := context.allocator) -> (Network(T), Error) where T == f32 || T == f64 {
	network := new_network(T, seed, allocator)
	if err := _unmarshal_into(&network, data); err != nil {
		release(&network)
		return {}, err
	}
	return network, nil
}

// _unmarshal_into appends the saved layers to an empty network.
_unmarshal_into :: proc(network: ^Network($T), data: []byte) -> Error {
	if len(data) < len(FILE_MAGIC) || string(data[:len(FILE_MAGIC)]) != FILE_MAGIC {
		return .Invalid_File
	}
	reader := _Cursor{data = data, offset = len(FILE_MAGIC)}
	version := _get_u32(&reader) or_return
	if version != FILE_VERSION {
		return .Unsupported_Version
	}
	float_size := _get_u32(&reader) or_return
	if float_size != 4 && float_size != 8 {
		return .Invalid_File
	}
	rank := _get_u32(&reader) or_return
	if rank > MAX_SAMPLE_RANK {
		return .Invalid_File
	}
	dims: [MAX_SAMPLE_RANK]int
	for i in 0..<rank {
		dims[i] = _get_u32(&reader) or_return
	}
	if rank > 0 {
		_file_error(set_input_shape(network, dims[:rank])) or_return
	}
	layer_count := _get_u32(&reader) or_return
	for _ in 0..<layer_count {
		kind_value := _get_u32(&reader) or_return
		activation_value := _get_u32(&reader) or_return
		if kind_value >= len(Layer_Kind) || activation_value >= len(Activation) {
			return .Invalid_File
		}
		kind, activation := Layer_Kind(kind_value), Activation(activation_value)
		if kind != .Dense && kind != .Conv2D && kind != .Batch_Norm && activation != .Identity {
			return .Invalid_File
		}

		// Check the values are all there before the builder allocates them,
		// so a corrupt size cannot request a huge allocation. The count is in
		// f64 so that no u32 sizes can overflow it.
		build_err: Error
		switch kind {
		case .Dense:
			inputs := _get_u32(&reader) or_return
			outputs := _get_u32(&reader) or_return
			_check_values_present(reader, (f64(inputs)+1)*f64(outputs), float_size) or_return
			build_err = dense(network, inputs, outputs, activation)
		case .Conv2D:
			input_channels := _get_u32(&reader) or_return
			output_channels := _get_u32(&reader) or_return
			kernel_size := _get_u32(&reader) or_return
			stride := _get_u32(&reader) or_return
			padding := _get_u32(&reader) or_return
			_check_values_present(reader, (f64(kernel_size)*f64(kernel_size)*f64(input_channels)+1)*f64(output_channels), float_size) or_return
			build_err = conv2d(network, input_channels, output_channels, kernel_size, activation, stride, padding)
		case .Max_Pool2D:
			build_err = max_pool2d(network, _get_u32(&reader) or_return)
		case .Flatten:
			build_err = flatten(network)
		case .Dropout:
			build_err = dropout(network, _get_f64(&reader) or_return)
		case .Batch_Norm:
			features := _get_u32(&reader) or_return
			momentum := _get_f64(&reader) or_return
			epsilon := _get_f64(&reader) or_return
			_check_values_present(reader, 4*f64(features), float_size) or_return
			build_err = batch_norm(network, activation, momentum, epsilon)
			if build_err == nil && len(network.layers[len(network.layers)-1].weights.data) != features {
				return .Invalid_File
			}
		case .Upsample2D:
			build_err = upsample2d(network, _get_u32(&reader) or_return)
		}
		_file_error(build_err) or_return
		for stored in _layer_values(network.layers[len(network.layers)-1]) {
			_get_floats(&reader, stored, float_size)
		}
	}
	if reader.offset != len(data) {
		return .Invalid_File
	}
	return nil
}

// The file helpers are not generic, so core:os and base:runtime count as used
// even when a program never instantiates save or load.
_write_file :: proc(path: string, data: []byte) -> Error {
	if os.write_entire_file(path, data) != nil {
		return .Write_Failed
	}
	return nil
}

_read_file :: proc(path: string, allocator: runtime.Allocator) -> ([]byte, Error) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		if _, ok := err.(runtime.Allocator_Error); ok {
			return nil, .Allocation_Failed
		}
		return nil, .Read_Failed
	}
	return data, nil
}

_make_bytes :: proc(size: int, allocator: runtime.Allocator) -> ([]byte, Error) {
	data, err := make([]byte, size, allocator)
	if err != nil {
		return nil, .Allocation_Failed
	}
	return data, nil
}

// _layer_settings_size is the number of bytes between a layer's activation
// and its values.
_layer_settings_size :: proc(kind: Layer_Kind) -> int {
	switch kind {
	case .Dense:      return 8
	case .Conv2D:     return 20
	case .Max_Pool2D: return 4
	case .Flatten:    return 0
	case .Dropout:    return 8
	case .Batch_Norm: return 20
	case .Upsample2D: return 4
	}
	return 0
}

// _layer_values lists the stored tensors' elements in file order. Tensors a
// layer kind does not use are empty.
_layer_values :: proc(layer: Layer($T)) -> [4][]T {
	return {layer.weights.data, layer.bias.data, layer.running_mean.data, layer.running_variance.data}
}

_layer_value_count :: proc(layer: Layer($T)) -> int {
	count := 0
	for values in _layer_values(layer) {
		count += len(values)
	}
	return count
}

// _check_values_present returns Invalid_File unless count values of
// float_size bytes remain after the reader.
_check_values_present :: proc(reader: _Cursor, count: f64, float_size: int) -> Error {
	if count*f64(float_size) > f64(len(reader.data)-reader.offset) {
		return .Invalid_File
	}
	return nil
}

// _file_error reports a layer the builders reject as Invalid_File, keeping
// allocation failures as they are.
_file_error :: proc(err: Error) -> Error {
	if err == nil || err == Error(Tensor_Error.Allocation_Failed) {
		return err
	}
	return .Invalid_File
}

_Cursor :: struct {
	data:   []byte,
	offset: int,
}

_put_u32 :: proc(writer: ^_Cursor, value: int) {
	endian.put_u32(writer.data[writer.offset:], .Little, u32(value))
	writer.offset += 4
}

_put_f64 :: proc(writer: ^_Cursor, value: f64) {
	endian.put_f64(writer.data[writer.offset:], .Little, value)
	writer.offset += 8
}

_put_floats :: proc(writer: ^_Cursor, values: []$T) {
	for value in values {
		when T == f32 {
			endian.put_f32(writer.data[writer.offset:], .Little, value)
		} else {
			endian.put_f64(writer.data[writer.offset:], .Little, value)
		}
		writer.offset += size_of(T)
	}
}

_get_u32 :: proc(reader: ^_Cursor) -> (int, Error) {
	value, ok := endian.get_u32(reader.data[reader.offset:], .Little)
	if !ok {
		return 0, .Invalid_File
	}
	reader.offset += 4
	return int(value), nil
}

_get_f64 :: proc(reader: ^_Cursor) -> (f64, Error) {
	value, ok := endian.get_f64(reader.data[reader.offset:], .Little)
	if !ok {
		return 0, .Invalid_File
	}
	reader.offset += 8
	return value, nil
}

// _get_floats fills values from float_size-byte floats. The caller has
// checked they are all present.
_get_floats :: proc(reader: ^_Cursor, values: []$T, float_size: int) {
	for &value in values {
		if float_size == 4 {
			stored, _ := endian.get_f32(reader.data[reader.offset:], .Little)
			value = T(stored)
		} else {
			stored, _ := endian.get_f64(reader.data[reader.offset:], .Little)
			value = T(stored)
		}
		reader.offset += float_size
	}
}
