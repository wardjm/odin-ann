// The Odin half of the PyTorch comparison. bench/torch_parity.py writes a
// case directory (spec.txt plus raw little-endian arrays), builds this program
// with -o:speed, and runs it once per case:
//
//     torch_parity <case_dir>
//
// spec.txt describes the network as builder calls: an optional "input h w c"
// line, then "dense in out activation", "conv2d in_channels out_channels
// kernel activation", "max_pool2d size", "flatten" and "batch_norm features
// activation" lines in order. Weights and biases, a batch norm's scale and
// shift, are read and written as w<i>.bin and b<i>.bin, where i is the
// layer's index; layers without parameters have none. A batch norm's running
// mean and variance are written as m<i>.bin and v<i>.bin after training.
//
// The program loads the network and data, writes the forward output, the loss
// from every training step and the final parameters, then times forward and
// train_step and writes the medians to timing.txt.
package torch_parity

import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strconv"
import "core:strings"
import "core:time"
import ann "../.."
import tensor "../../../odin-tensor"

Spec :: struct {
	dtype:       string,
	threads:     int,
	// input_shape is one sample's shape; the first layer's input if empty.
	input_shape: [dynamic]int,
	layers:      [dynamic]Layer_Spec,
	loss:        ann.Loss,
	optimizer:   ann.Optimizer,
	batch:       int,
	steps:       int,
	fwd_reps:    int,
	train_reps:  int,
}

// Layer_Spec holds one builder call. sizes are dense's inputs and outputs,
// conv2d's input channels, output channels and kernel size, max_pool2d's
// pool size, or batch_norm's features.
Layer_Spec :: struct {
	kind:       ann.Layer_Kind,
	sizes:      [3]int,
	activation: ann.Activation,
}

main :: proc() {
	if len(os.args) != 2 {
		fmt.eprintln("usage: torch_parity <case_dir>")
		os.exit(2)
	}
	dir := os.args[1]
	spec, ok := parse_spec(dir)
	if !ok {
		os.exit(2)
	}
	if spec.threads > 1 {
		if err := tensor.set_thread_count(spec.threads); err != .None {
			fmt.eprintln("threads:", err)
			os.exit(2)
		}
	}
	defer tensor.set_thread_count(1)
	err: ann.Error
	switch spec.dtype {
	case "f32":
		err = run(f32, dir, spec)
	case "f64":
		err = run(f64, dir, spec)
	case:
		fmt.eprintln("unknown dtype", spec.dtype)
		os.exit(2)
	}
	if err != nil {
		fmt.eprintln("error:", err)
		os.exit(1)
	}
}

run :: proc($T: typeid, dir: string, spec: Spec) -> ann.Error {
	network := ann.new_network(T)
	defer ann.release(&network)
	if len(spec.input_shape) > 0 {
		ann.set_input_shape(&network, spec.input_shape[:]) or_return
	}
	for layer, i in spec.layers {
		switch layer.kind {
		case .Dense:
			ann.dense(&network, layer.sizes[0], layer.sizes[1], layer.activation) or_return
		case .Conv2D:
			ann.conv2d(&network, layer.sizes[0], layer.sizes[1], layer.sizes[2], layer.activation) or_return
		case .Max_Pool2D:
			ann.max_pool2d(&network, layer.sizes[0]) or_return
		case .Flatten:
			ann.flatten(&network) or_return
		case .Batch_Norm:
			ann.batch_norm(&network, layer.activation) or_return
			if len(network.layers[i].weights.data) != layer.sizes[0] {
				return .Shape_Mismatch
			}
		case .Dropout:
			return .Invalid_Hyperparameter // masks cannot match PyTorch's
		}
		if len(network.layers[i].weights.data) > 0 {
			load_into(dir, fmt.tprintf("w%d.bin", i), network.layers[i].weights.data) or_return
			load_into(dir, fmt.tprintf("b%d.bin", i), network.layers[i].bias.data) or_return
		}
	}

	input_shape := make([dynamic]int, context.temp_allocator)
	append(&input_shape, spec.batch)
	sample := network.input_shape
	append(&input_shape, ..sample.dims[:sample.rank])
	inputs := tensor.new_tensor(T, input_shape[:]) or_return
	defer tensor.release(&inputs)
	load_into(dir, "x.bin", inputs.data) or_return
	targets := tensor.new_tensor(T, []int{spec.batch, ann.output_features_of(network)}) or_return
	defer tensor.release(&targets)
	load_into(dir, "y.bin", targets.data) or_return

	// Correctness: forward output, then the loss of each step and the
	// parameters after the last one.
	output := ann.forward(network, inputs) or_return
	save(dir, "out_forward.bin", output.data)
	tensor.release(&output)

	trainer := ann.trainer_init(T, spec.loss, spec.optimizer) or_return
	defer ann.trainer_release(&trainer)
	losses := make([]f64, spec.steps, context.temp_allocator)
	for &loss in losses {
		value := ann.train_step(&trainer, &network, inputs, targets) or_return
		loss = f64(value)
	}
	save(dir, "out_losses.bin", losses)
	for layer, i in network.layers {
		if len(layer.weights.data) > 0 {
			save(dir, fmt.tprintf("out_w%d.bin", i), layer.weights.data)
			save(dir, fmt.tprintf("out_b%d.bin", i), layer.bias.data)
		}
		if layer.kind == .Batch_Norm {
			save(dir, fmt.tprintf("out_m%d.bin", i), layer.running_mean.data)
			save(dir, fmt.tprintf("out_v%d.bin", i), layer.running_variance.data)
		}
	}

	// Timing. Training continues on the same network; the values no longer
	// matter, only the cost of each call.
	forward_samples := make([]f64, spec.fwd_reps, context.temp_allocator)
	for &sample in forward_samples {
		start := time.tick_now()
		result := ann.forward(network, inputs) or_return
		tensor.release(&result)
		sample = time.duration_milliseconds(time.tick_since(start))
	}
	train_samples := make([]f64, spec.train_reps, context.temp_allocator)
	for &sample in train_samples {
		start := time.tick_now()
		ann.train_step(&trainer, &network, inputs, targets) or_return
		sample = time.duration_milliseconds(time.tick_since(start))
	}
	timing := fmt.tprintf("forward_ms %.6f\ntrain_step_ms %.6f\n", median(forward_samples), median(train_samples))
	_ = os.write_entire_file(path(dir, "timing.txt"), timing)
	return nil
}

median :: proc(samples: []f64) -> f64 {
	slice.sort(samples)
	return samples[len(samples)/2]
}

path :: proc(dir, name: string) -> string {
	return strings.concatenate({dir, "/", name}, context.temp_allocator)
}

load_into :: proc(dir, name: string, destination: []$T) -> ann.Error {
	bytes, err := os.read_entire_file(path(dir, name), context.temp_allocator)
	if err != nil || len(bytes) != len(destination)*size_of(T) {
		fmt.eprintfln("%s: read error %v or %d bytes, wanted %d", name, err, len(bytes), len(destination)*size_of(T))
		return .Shape_Mismatch
	}
	mem.copy(raw_data(destination), raw_data(bytes), len(bytes))
	return nil
}

save :: proc(dir, name: string, values: []$T) {
	if err := os.write_entire_file(path(dir, name), slice.to_bytes(values)); err != nil {
		fmt.eprintln(name, err)
	}
}

// parse_spec reads spec.txt: one "key value..." entry per line.
parse_spec :: proc(dir: string) -> (spec: Spec, ok: bool) {
	text, err := os.read_entire_file(path(dir, "spec.txt"), context.temp_allocator)
	if err != nil {
		fmt.eprintln("spec.txt:", err)
		return
	}
	for line in strings.split_lines(string(text), context.temp_allocator) {
		f := strings.fields(line, context.temp_allocator)
		if len(f) == 0 {
			continue
		}
		switch f[0] {
		case "dtype":
			spec.dtype = f[1]
		case "threads":
			spec.threads = atoi(f[1])
		case "input":
			for field in f[1:] {
				append(&spec.input_shape, atoi(field))
			}
		case "dense", "conv2d", "batch_norm":
			// The activation follows the sizes.
			layer := Layer_Spec{kind = .Dense if f[0] == "dense" else .Conv2D if f[0] == "conv2d" else .Batch_Norm}
			for field, i in f[1:len(f)-1] {
				layer.sizes[i] = atoi(field)
			}
			activation, found := fmt.string_to_enum_value(ann.Activation, f[len(f)-1])
			if !found {
				fmt.eprintln("unknown activation", f[len(f)-1])
				return
			}
			layer.activation = activation
			append(&spec.layers, layer)
		case "max_pool2d":
			append(&spec.layers, Layer_Spec{kind = .Max_Pool2D, sizes = {atoi(f[1]), 0, 0}})
		case "flatten":
			append(&spec.layers, Layer_Spec{kind = .Flatten})
		case "loss":
			loss, found := fmt.string_to_enum_value(ann.Loss, f[1])
			if !found {
				fmt.eprintln("unknown loss", f[1])
				return
			}
			spec.loss = loss
		case "sgd":
			spec.optimizer = ann.sgd(atof(f[1]), momentum = atof(f[2]), weight_decay = atof(f[3]))
		case "adam":
			spec.optimizer = ann.adam(atof(f[1]), atof(f[2]), atof(f[3]), atof(f[4]), atof(f[5]))
		case "batch":
			spec.batch = atoi(f[1])
		case "steps":
			spec.steps = atoi(f[1])
		case "fwd_reps":
			spec.fwd_reps = atoi(f[1])
		case "train_reps":
			spec.train_reps = atoi(f[1])
		case:
			fmt.eprintln("unknown spec key", f[0])
			return
		}
	}
	if len(spec.layers) == 0 {
		fmt.eprintln("spec has no layers")
		return
	}
	return spec, true
}

atoi :: proc(s: string) -> int {
	value, _ := strconv.parse_int(s)
	return value
}

atof :: proc(s: string) -> f64 {
	value, _ := strconv.parse_f64(s)
	return value
}
