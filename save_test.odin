package ann

import "core:encoding/endian"
import "core:mem"
import "core:os"
import "core:testing"
import tensor "../odin-tensor"

expect_same_network :: proc(t: ^testing.T, actual: Network($A), expected: Network($E), loc := #caller_location) {
	testing.expect(t, _same_sample_shape(actual.input_shape, expected.input_shape), "input shapes differ", loc = loc)
	if !testing.expect_value(t, len(actual.layers), len(expected.layers), loc) {
		return
	}
	for layer, i in actual.layers {
		want := expected.layers[i]
		testing.expect_value(t, layer.kind, want.kind, loc)
		testing.expect_value(t, layer.activation, want.activation, loc)
		testing.expect_value(t, layer.pool_size, want.pool_size, loc)
		testing.expect_value(t, layer.stride, want.stride, loc)
		testing.expect_value(t, layer.padding, want.padding, loc)
		testing.expect(t, _same_sample_shape(layer.output_shape, want.output_shape), "output shapes differ", loc = loc)
		expect_shape(t, layer.weights.shape, want.weights.shape, loc)
		expect_shape(t, layer.bias.shape, want.bias.shape, loc)
		for value, j in layer.weights.data {
			testing.expect_value(t, value, A(want.weights.data[j]), loc)
		}
		for value, j in layer.bias.data {
			testing.expect_value(t, value, A(want.bias.data[j]), loc)
		}
	}
}

@(test)
test_unmarshal_restores_every_layer_kind :: proc(t: ^testing.T) {
	network := build_image_classifier(t, 6)
	defer release(&network)
	data, marshal_err := marshal(network)
	require_ok(t, marshal_err)
	defer delete(data)

	restored, unmarshal_err := unmarshal(f64, data)
	require_ok(t, unmarshal_err)
	defer release(&restored)
	expect_same_network(t, restored, network)

	inputs := pseudo_tensor(t, []int{3, 7, 7, 2}, 1)
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

	// An empty network, with or without an input shape, also round-trips.
	empty := new_network(f32)
	defer release(&empty)
	for _ in 0..<2 {
		empty_data, empty_err := marshal(empty)
		require_ok(t, empty_err)
		defer delete(empty_data)
		restored_empty, restored_err := unmarshal(f32, empty_data)
		require_ok(t, restored_err)
		defer release(&restored_empty)
		expect_same_network(t, restored_empty, empty)
		require_ok(t, set_input_shape(&empty, []int{4, 4, 1}))
	}
}

@(test)
test_unmarshal_converts_float_types :: proc(t: ^testing.T) {
	network := new_network(f32, seed = 2)
	defer release(&network)
	require_ok(t, dense(&network, 3, 4, .ReLU))
	require_ok(t, dense(&network, 4, 2, .Sigmoid))
	data, marshal_err := marshal(network)
	require_ok(t, marshal_err)
	defer delete(data)

	widened, widen_err := unmarshal(f64, data)
	require_ok(t, widen_err)
	defer release(&widened)
	expect_same_network(t, widened, network)

	// f32 -> f64 -> f32 is exact.
	wide_data, wide_err := marshal(widened)
	require_ok(t, wide_err)
	defer delete(wide_data)
	narrowed, narrow_err := unmarshal(f32, wide_data)
	require_ok(t, narrow_err)
	defer release(&narrowed)
	expect_same_network(t, narrowed, network)
}

@(test)
test_unmarshal_rejects_malformed_data :: proc(t: ^testing.T) {
	network := build_image_classifier(t, 7)
	defer release(&network)
	data, marshal_err := marshal(network)
	require_ok(t, marshal_err)
	defer delete(data)

	// Every proper prefix is truncated.
	for length in 0..<len(data) {
		restored, err := unmarshal(f64, data[:length])
		expect_error(t, err, Ann_Error.Invalid_File)
		release(&restored)
	}

	corrupt := make([]byte, len(data)+1)
	defer delete(corrupt)
	copy(corrupt, data)
	expect_corrupt :: proc(t: ^testing.T, corrupt: []byte, expected: Error, loc := #caller_location) {
		restored, err := unmarshal(f64, corrupt)
		expect_error(t, err, expected, loc)
		release(&restored)
	}
	expect_corrupt(t, corrupt, Ann_Error.Invalid_File) // a trailing byte

	corrupt = corrupt[:len(data)]
	corrupt[0] = 'X'
	expect_corrupt(t, corrupt, Ann_Error.Invalid_File)
	corrupt[0] = data[0]

	// Header fields follow the 8-byte magic: version, float size, input rank,
	// the input dims 7, 7 and 2, the layer count and the first layer's kind.
	set_u32 :: proc(data: []byte, field: int, value: u32) {
		endian.put_u32(data[8+4*field:], .Little, value)
	}
	set_u32(corrupt, 0, FILE_VERSION+1)
	expect_corrupt(t, corrupt, Ann_Error.Unsupported_Version)
	set_u32(corrupt, 0, FILE_VERSION)
	set_u32(corrupt, 1, 2)
	expect_corrupt(t, corrupt, Ann_Error.Invalid_File)
	set_u32(corrupt, 1, 8)
	set_u32(corrupt, 7, u32(len(Layer_Kind)))
	expect_corrupt(t, corrupt, Ann_Error.Invalid_File)
	set_u32(corrupt, 7, u32(Layer_Kind.Conv2D))
	// The input no longer has the first convolution's two channels.
	set_u32(corrupt, 5, 3)
	expect_corrupt(t, corrupt, Ann_Error.Invalid_File)
	set_u32(corrupt, 5, 2)
	expect_corrupt(t, corrupt, nil)

	// A dense layer far larger than the data is rejected before allocating.
	huge := new_network(f32)
	defer release(&huge)
	require_ok(t, dense(&huge, 1, 1))
	huge_data, huge_err := marshal(huge)
	require_ok(t, huge_err)
	defer delete(huge_data)
	set_u32(huge_data, 1, 4)
	set_u32(huge_data, 2, 1)
	set_u32(huge_data, 3, 0xFFFF_FFFF)
	set_u32(huge_data, 4, 1)
	set_u32(huge_data, 7, 0xFFFF_FFFF)
	set_u32(huge_data, 8, 0xFFFF_FFFF)
	restored, err := unmarshal(f32, huge_data, allocator = mem.panic_allocator())
	expect_error(t, err, Ann_Error.Invalid_File)
	release(&restored)
}

@(test)
test_load_reads_what_save_wrote :: proc(t: ^testing.T) {
	network := build_image_classifier(t, 8)
	defer release(&network)
	dir, dir_err := os.make_directory_temp("", "odin-ann-*", context.allocator)
	if !testing.expect_value(t, dir_err, nil) {
		return
	}
	defer {
		os.remove_all(dir)
		delete(dir)
	}
	path, path_err := os.join_path({dir, "network.bin"}, context.allocator)
	testing.expect_value(t, path_err, nil)
	defer delete(path)

	require_ok(t, save(network, path))
	restored, load_err := load(f64, path)
	require_ok(t, load_err)
	defer release(&restored)
	expect_same_network(t, restored, network)

	missing, missing_err := load(f64, "does/not/exist.bin")
	expect_error(t, missing_err, Ann_Error.Read_Failed)
	release(&missing)
	expect_error(t, save(network, "does/not/exist.bin"), Ann_Error.Write_Failed)
}

save_allocation_scenario :: proc(allocator: mem.Allocator) -> Error {
	network := new_network(f64, 0, allocator)
	defer release(&network)
	set_input_shape(&network, []int{5, 5, 1}) or_return
	conv2d(&network, 1, 2, 2, .ReLU) or_return
	max_pool2d(&network) or_return
	flatten(&network) or_return
	dense(&network, 8, 2, .Softmax) or_return
	data := marshal(network, allocator) or_return
	defer delete(data, allocator)
	restored := unmarshal(f32, data, allocator = allocator) or_return
	release(&restored)
	return nil
}

@(test)
test_save_allocation_failures_release_partial_results :: proc(t: ^testing.T) {
	for budget in 0..<256 {
		state := Failing_Allocator{backing = context.allocator, remaining = budget}
		err := save_allocation_scenario(mem.Allocator{procedure = failing_allocator_proc, data = &state})
		if err == nil {
			return
		}
		testing.expect_value(t, err, Error(Tensor_Error.Allocation_Failed))
	}
	testing.fail_now(t, "scenario never succeeded")
}
