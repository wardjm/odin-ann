// Package digits loads the MNIST handwritten digits and measures networks on
// them, for the examples that train on MNIST.
package digits

import "core:bytes"
import "core:compress/gzip"
import "core:encoding/endian"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:sys/info"
import "core:time"
import ann "../../.."
import tensor "../../../../odin-tensor"

SIDE :: 28
CLASSES :: 10
// Test images are classified this many at a time, which bounds the memory the
// convolution windows take.
EVAL_BATCH :: 500

Dataset :: struct {
	images: tensor.Tensor(f32), // [count, 28, 28, 1], pixels scaled to [0, 1]
	labels: []int,
}

dataset_release :: proc(dataset: ^Dataset) {
	tensor.release(&dataset.images)
	delete(dataset.labels)
}

// Arguments are the examples' optional command-line arguments: the data
// directory, the number of epochs and the number of threads.
Arguments :: struct {
	dir:     string,
	epochs:  int,
	threads: int,
}

// parse_arguments reads the arguments, defaulting to data/mnist, epochs and
// every physical core, and sets odin-tensor's thread count. It prints what was
// wrong and returns false for invalid arguments. min_epochs is 0 or 1.
parse_arguments :: proc(epochs, min_epochs: int) -> (arguments: Arguments, ok: bool) {
	arguments.dir = len(os.args) > 1 ? os.args[1] : "data/mnist"
	arguments.epochs = epochs
	if len(os.args) > 2 {
		parsed: bool
		if arguments.epochs, parsed = strconv.parse_int(os.args[2]); !parsed || arguments.epochs < min_epochs {
			fmt.eprintf("epochs must be an integer of at least %d, got %s\n", min_epochs, os.args[2])
			return
		}
	}
	arguments.threads, _, _ = info.cpu_core_count()
	if len(os.args) > 3 {
		parsed: bool
		if arguments.threads, parsed = strconv.parse_int(os.args[3]); !parsed || arguments.threads <= 0 {
			fmt.eprintln("threads must be a positive integer, got", os.args[3])
			return
		}
	}
	if err := tensor.set_thread_count(max(arguments.threads, 1)); err != .None {
		fmt.eprintln("threads:", err)
		return
	}
	return arguments, true
}

Progress :: struct {
	network: ^ann.Network(f32),
	test:    ^Dataset,
	start:   time.Time,
	err:     ann.Error,
}

// report_epoch prints the epoch's training loss and the current test
// accuracy. It stops training if the evaluation fails.
report_epoch :: proc(epoch: int, loss: f64, user_data: rawptr) -> bool {
	progress := (^Progress)(user_data)
	correct, err := count_correct(progress.network^, progress.test^)
	if err != nil {
		progress.err = err
		return false
	}
	elapsed := time.duration_seconds(time.since(progress.start))
	fmt.printf("epoch %d  loss %.4f  test accuracy %.2f%%  (%.0fs)\n", epoch+1, loss, 100*f64(correct)/f64(len(progress.test.labels)), elapsed)
	return true
}

// count_correct classifies the test set EVAL_BATCH images at a time, each
// batch a zero-copy slice of the dataset.
count_correct :: proc(network: ann.Network(f32), test: Dataset) -> (correct: int, err: ann.Error) {
	shape, strides: [tensor.MAX_RANK]int
	count := len(test.labels)
	for start := 0; start < count; start += EVAL_BATCH {
		batch := min(EVAL_BATCH, count-start)
		images := tensor.slice_view(tensor.as_view(test.images), 0, start, batch, 1, shape[:], strides[:]) or_return
		output := ann.forward_views(network, images) or_return
		defer tensor.release(&output)
		classes := tensor.argmax_axis(output, -1) or_return
		defer tensor.release(&classes)
		for class, i in classes.data {
			if class == test.labels[start+i] {
				correct += 1
			}
		}
	}
	return correct, nil
}

// show_mistake draws the first test digit the network misclassifies.
show_mistake :: proc(network: ann.Network(f32), test: Dataset) {
	shape, strides: [tensor.MAX_RANK]int
	for start := 0; start < len(test.labels); start += EVAL_BATCH {
		batch := min(EVAL_BATCH, len(test.labels)-start)
		images, slice_err := tensor.slice_view(tensor.as_view(test.images), 0, start, batch, 1, shape[:], strides[:])
		if slice_err != .None do return
		output, err := ann.forward_views(network, images)
		if err != nil do return
		defer tensor.release(&output)
		for i in 0..<batch {
			scores := output.data[i*CLASSES:][:CLASSES]
			predicted := 0
			for score, class in scores {
				if score > scores[predicted] {
					predicted = class
				}
			}
			label := test.labels[start+i]
			if predicted == label {
				continue
			}
			fmt.printf("\ntest image %d is a %d; the network says %d (%.0f%% sure):\n", start+i, label, predicted, 100*scores[predicted])
			pixels := test.images.data[(start+i)*SIDE*SIDE:][:SIDE*SIDE]
			for row in 0..<SIDE {
				for pixel in pixels[row*SIDE:][:SIDE] {
					fmt.print(pixel > 0.6 ? "##" : pixel > 0.2 ? "++" : "  ")
				}
				fmt.println()
			}
			return
		}
	}
	fmt.println("\nthe network classifies every test image correctly")
}

// load_dataset reads <prefix>-images-idx3-ubyte.gz and
// <prefix>-labels-idx1-ubyte.gz from dir, printing what went wrong on failure.
load_dataset :: proc(dir, prefix: string) -> (dataset: Dataset, ok: bool) {
	image_path := fmt.tprintf("%s/%s-images-idx3-ubyte.gz", dir, prefix)
	label_path := fmt.tprintf("%s/%s-labels-idx1-ubyte.gz", dir, prefix)
	pixels, image_dims := read_idx(image_path, 3) or_return
	defer delete(pixels)
	labels, label_dims := read_idx(label_path, 1) or_return
	defer delete(labels)
	count := image_dims[0]
	if image_dims[1] != SIDE || image_dims[2] != SIDE || label_dims[0] != count {
		fmt.eprintf("%s: expected %dx%d images with one label each\n", dir, SIDE, SIDE)
		return dataset, false
	}

	err: tensor.Tensor_Error
	dataset.images, err = tensor.new_tensor(f32, []int{count, SIDE, SIDE, 1})
	if err != .None {
		fmt.eprintln("images:", err)
		return dataset, false
	}
	for pixel, i in pixels {
		dataset.images.data[i] = f32(pixel) / 255
	}
	dataset.labels = make([]int, count)
	for label, i in labels {
		dataset.labels[i] = int(label)
	}
	return dataset, true
}

// read_idx decompresses a gzipped IDX file holding unsigned bytes with rank
// dimensions and returns its elements and dimensions. The header is a
// big-endian magic number, 0x0800 plus the rank, then one u32 per dimension.
read_idx :: proc(path: string, $RANK: int) -> (elements: []byte, dims: [RANK]int, ok: bool) {
	buffer: bytes.Buffer
	defer bytes.buffer_destroy(&buffer)
	if err := gzip.load(path, &buffer); err != nil {
		fmt.eprintf("%s: %v (run scripts/fetch_mnist.sh to download MNIST)\n", path, err)
		return
	}
	contents := bytes.buffer_to_bytes(&buffer)
	header := 4 * (1 + RANK)
	magic, magic_ok := endian.get_u32(contents, .Big)
	if !magic_ok || magic != u32(0x0800 + RANK) || len(contents) < header {
		fmt.eprintf("%s: not an IDX file of rank %d\n", path, RANK)
		return
	}
	count := 1
	for &dim, i in dims {
		value, _ := endian.get_u32(contents[4*(i+1):], .Big)
		dim = int(value)
		count *= dim
	}
	if len(contents) != header+count {
		fmt.eprintf("%s: expected %d elements after the header, found %d\n", path, count, len(contents)-header)
		return
	}
	elements = make([]byte, count)
	copy(elements, contents[header:])
	return elements, dims, true
}
