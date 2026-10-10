// Trains a small convolutional network on the MNIST handwritten digits with
// shuffled minibatches and Adam, reporting test accuracy after every epoch,
// then saves it as cnn.bin in the data directory and draws a test digit the
// network gets wrong. Fetch the data first:
//
//     scripts/fetch_mnist.sh
//     odin run examples/mnist -o:speed
//     odin run examples/mnist -o:speed -- path/to/mnist 5 4
//
// The optional arguments are the data directory, the number of epochs and the
// number of threads, which defaults to the physical core count. The results
// are the same for any thread count. With 0 epochs, the example loads the
// saved network instead of training one, and tests that.
package mnist

import "core:bytes"
import "core:compress/gzip"
import "core:encoding/endian"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:sys/info"
import "core:time"
import ann "../.."
import tensor "../../../odin-tensor"

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

main :: proc() {
	dir := len(os.args) > 1 ? os.args[1] : "data/mnist"
	epochs := 3
	if len(os.args) > 2 {
		ok: bool
		if epochs, ok = strconv.parse_int(os.args[2]); !ok || epochs < 0 {
			fmt.eprintln("epochs must be a non-negative integer, got", os.args[2])
			return
		}
	}
	threads, _, _ := info.cpu_core_count()
	if len(os.args) > 3 {
		ok: bool
		if threads, ok = strconv.parse_int(os.args[3]); !ok || threads <= 0 {
			fmt.eprintln("threads must be a positive integer, got", os.args[3])
			return
		}
	}
	if err := tensor.set_thread_count(max(threads, 1)); err != .None {
		fmt.eprintln("threads:", err)
		return
	}
	defer tensor.set_thread_count(1)

	test, test_ok := load_dataset(dir, "t10k")
	if !test_ok do return
	defer dataset_release(&test)
	saved_path := fmt.tprintf("%s/cnn.bin", dir)

	network: ann.Network(f32)
	defer ann.release(&network)
	if epochs == 0 {
		err: ann.Error
		if network, err = ann.load(f32, saved_path); err != nil {
			fmt.eprintf("%s: %v (train with a positive epoch count to save one)\n", saved_path, err)
			return
		}
		correct, eval_err := count_correct(network, test)
		if eval_err != nil {
			fmt.eprintln("evaluation:", eval_err)
			return
		}
		fmt.printf("loaded %s\n\n", saved_path)
		ann.print_summary(network)
		fmt.printf("\ntest accuracy %.2f%%\n", 100*f64(correct)/f64(len(test.labels)))
	} else {
		train, train_ok := load_dataset(dir, "train")
		if !train_ok do return
		defer dataset_release(&train)
		if !train_network(&network, train, &test, epochs) do return
		if err := ann.save(network, saved_path); err != nil {
			fmt.eprintf("%s: %v\n", saved_path, err)
			return
		}
		fmt.printf("\nsaved the network to %s; pass 0 epochs to load it\n", saved_path)
	}
	show_mistake(network, test)
}

// train_network builds the network and trains it on train for epochs,
// reporting test accuracy after each one. It prints any error and returns
// false.
train_network :: proc(network: ^ann.Network(f32), train: Dataset, test: ^Dataset, epochs: int) -> bool {
	targets, targets_err := ann.one_hot(f32, train.labels, CLASSES)
	if targets_err != nil {
		fmt.eprintln("targets:", targets_err)
		return false
	}
	defer tensor.release(&targets)
	fmt.printf("%d training and %d test images, %d threads\n\n", len(train.labels), len(test.labels), tensor.thread_count())

	// 28x28x1 -> 24x24x8 -> 12x12x8 -> 8x8x16 -> 4x4x16 -> 256 -> 10
	network^ = ann.new_network(f32, seed = 1)
	if err := build(network); err != nil {
		fmt.eprintln("network:", err)
		return false
	}
	ann.print_summary(network^)
	fmt.println()

	trainer, trainer_err := ann.trainer_init(f32, .Cross_Entropy, ann.adam(learning_rate = 0.002))
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return false
	}
	defer ann.trainer_release(&trainer)

	progress := Progress{network = network, test = test, start = time.now()}
	options := ann.Fit_Options{epochs = epochs, batch_size = 64, shuffle = true, seed = 2, on_epoch = report_epoch, user_data = &progress}
	if _, err := ann.fit(&trainer, network, train.images, targets, options); err != nil {
		fmt.eprintln("fit:", err)
		return false
	}
	if progress.err != nil {
		fmt.eprintln("evaluation:", progress.err)
		return false
	}
	return true
}

build :: proc(network: ^ann.Network(f32)) -> ann.Error {
	ann.set_input_shape(network, []int{SIDE, SIDE, 1}) or_return
	ann.conv2d(network, 1, 8, 5, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.conv2d(network, 8, 16, 5, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.flatten(network) or_return
	return ann.dense(network, 4*4*16, CLASSES, .Softmax)
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
