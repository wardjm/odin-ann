// Trains a convolutional network on CIFAR-10, 32x32 colour photos of ten
// kinds of object. Every epoch trains on a new augmented copy of the training
// images, each shifted by up to SHIFT pixels and mirrored half the time,
// written from on_epoch, which also lowers the learning rate along a cosine.
// It reports test accuracy after each epoch, saves the network as cifar10.bin
// in the data directory, and prints the test set's confusion matrix. Fetch
// the data first:
//
//     scripts/fetch_cifar10.sh
//     odin run examples/cifar10 -o:speed
//     odin run examples/cifar10 -o:speed -- path/to/cifar10 20 4
//
// The optional arguments are the data directory, the number of epochs and the
// number of threads, which defaults to the physical core count. With 0
// epochs, the example loads the saved network instead of training one.
package cifar10

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:strconv"
import "core:sys/info"
import "core:time"
import ann "../.."
import tensor "../../../odin-tensor"

SIDE :: 32
CHANNELS :: 3
PIXELS :: SIDE * SIDE * CHANNELS
CLASSES :: 10
CLASS_NAMES :: [CLASSES]string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}
// Images are normalized with the training set's per-channel mean and
// standard deviation.
MEAN :: [CHANNELS]f32{0.4914, 0.4822, 0.4465}
STD :: [CHANNELS]f32{0.2470, 0.2435, 0.2616}
// SHIFT is the most pixels augmentation moves an image in each direction.
SHIFT :: 4
// Test images are classified this many at a time.
EVAL_BATCH :: 500
LEARNING_RATE :: 0.002

main :: proc() {
	dir := len(os.args) > 1 ? os.args[1] : "data/cifar10"
	epochs := 15
	if len(os.args) > 2 {
		parsed: bool
		if epochs, parsed = strconv.parse_int(os.args[2]); !parsed || epochs < 0 {
			fmt.eprintln("epochs must be a non-negative integer, got", os.args[2])
			return
		}
	}
	threads, _, _ := info.cpu_core_count()
	if len(os.args) > 3 {
		parsed: bool
		if threads, parsed = strconv.parse_int(os.args[3]); !parsed || threads <= 0 {
			fmt.eprintln("threads must be a positive integer, got", os.args[3])
			return
		}
	}
	if err := tensor.set_thread_count(max(threads, 1)); err != .None {
		fmt.eprintln("threads:", err)
		return
	}
	defer tensor.set_thread_count(1)

	test, test_ok := load_dataset(dir, {"test_batch.bin"})
	if !test_ok do return
	defer dataset_release(&test)
	test_images, images_err := tensor.new_tensor(f32, []int{len(test.labels), SIDE, SIDE, CHANNELS})
	if images_err != .None {
		fmt.eprintln("test images:", images_err)
		return
	}
	defer tensor.release(&test_images)
	for i in 0..<len(test.labels) {
		write_image(test_images.data[i*PIXELS:][:PIXELS], test.pixels[i*PIXELS:][:PIXELS], 0, 0, false)
	}
	saved_path := fmt.tprintf("%s/cifar10.bin", dir)

	network: ann.Network(f32)
	defer ann.release(&network)
	if epochs == 0 {
		err: ann.Error
		if network, err = ann.load(f32, saved_path); err != nil {
			fmt.eprintf("%s: %v (train with a positive epoch count to save one)\n", saved_path, err)
			return
		}
		fmt.printf("loaded %s\n\n", saved_path)
		ann.print_summary(network)
	} else {
		train, train_ok := load_dataset(dir, {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"})
		if !train_ok do return
		defer dataset_release(&train)
		if !train_network(&network, train, test, test_images, epochs) do return
		if err := ann.save(network, saved_path); err != nil {
			fmt.eprintf("%s: %v\n", saved_path, err)
			return
		}
		fmt.printf("\nsaved the network to %s; pass 0 epochs to load it\n", saved_path)
	}

	confusion, err := confusion_matrix(network, test_images, test.labels)
	if err != nil {
		fmt.eprintln("evaluation:", err)
		return
	}
	print_confusion(confusion)
}

// build makes 32x32x3 -> 32x32x32 -> 16x16x32 -> 16x16x64 -> 8x8x64 ->
// 8x8x128 -> 4x4x128 -> 2048 -> 10. Padding keeps each convolution's output
// the size of its input, so only the pools shrink it.
build :: proc(network: ^ann.Network(f32)) -> ann.Error {
	ann.set_input_shape(network, []int{SIDE, SIDE, CHANNELS}) or_return
	ann.conv2d(network, CHANNELS, 32, 3, padding = 1) or_return
	ann.batch_norm(network, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.conv2d(network, 32, 64, 3, padding = 1) or_return
	ann.batch_norm(network, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.conv2d(network, 64, 128, 3, padding = 1) or_return
	ann.batch_norm(network, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.flatten(network) or_return
	ann.dropout(network, 0.3) or_return
	return ann.dense(network, 4*4*128, CLASSES, .Softmax)
}

// train_network builds the network and trains it for epochs, reporting test
// accuracy after each one. It prints any error and returns false.
train_network :: proc(network: ^ann.Network(f32), train, test: Dataset, test_images: tensor.Tensor(f32), epochs: int) -> bool {
	targets, targets_err := ann.one_hot(f32, train.labels, CLASSES)
	if targets_err != nil {
		fmt.eprintln("targets:", targets_err)
		return false
	}
	defer tensor.release(&targets)
	images, images_err := tensor.new_tensor(f32, []int{len(train.labels), SIDE, SIDE, CHANNELS})
	if images_err != .None {
		fmt.eprintln("training images:", images_err)
		return false
	}
	defer tensor.release(&images)
	fmt.printf("%d training and %d test images, %d threads\n\n", len(train.labels), len(test.labels), tensor.thread_count())

	network^ = ann.new_network(f32, seed = 1)
	if err := build(network); err != nil {
		fmt.eprintln("network:", err)
		return false
	}
	ann.print_summary(network^)
	fmt.println()

	trainer, trainer_err := ann.trainer_init(f32, .Cross_Entropy, ann.adam(learning_rate = LEARNING_RATE), seed = 3)
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return false
	}
	defer ann.trainer_release(&trainer)

	progress := Progress{
		network     = network,
		trainer     = &trainer,
		train       = train,
		images      = &images,
		test        = test_images,
		test_labels = test.labels,
		epochs      = epochs,
		rng         = rand.create(4),
		start       = time.now(),
	}
	augment(&images, train, &progress.rng)
	options := ann.Fit_Options{epochs = epochs, batch_size = 64, shuffle = true, seed = 2, on_epoch = report_epoch, user_data = &progress}
	if _, err := ann.fit(&trainer, network, images, targets, options); err != nil {
		fmt.eprintln("fit:", err)
		return false
	}
	if progress.err != nil {
		fmt.eprintln("evaluation:", progress.err)
		return false
	}
	return true
}

Progress :: struct {
	network:     ^ann.Network(f32),
	trainer:     ^ann.Trainer(f32),
	train:       Dataset,
	images:      ^tensor.Tensor(f32),
	test:        tensor.Tensor(f32),
	test_labels: []int,
	epochs:      int,
	rng:         rand.Default_Random_State,
	start:       time.Time,
	err:         ann.Error,
}

// report_epoch prints the epoch's training loss and test accuracy, sets the
// next epoch's learning rate along a cosine from LEARNING_RATE down to zero
// at the end of training, and writes the next epoch's augmented images.
report_epoch :: proc(epoch: int, loss: f64, user_data: rawptr) -> bool {
	progress := (^Progress)(user_data)
	confusion, err := confusion_matrix(progress.network^, progress.test, progress.test_labels)
	if err != nil {
		progress.err = err
		return false
	}
	correct := 0
	for class in 0..<CLASSES {
		correct += confusion[class][class]
	}
	elapsed := time.duration_seconds(time.since(progress.start))
	fmt.printf("epoch %d  loss %.4f  test accuracy %.2f%%  (%.0fs)\n", epoch+1, loss, 100*f64(correct)/f64(len(progress.test_labels)), elapsed)

	done := f64(epoch+1) / f64(progress.epochs)
	if progress.err = ann.set_learning_rate(progress.trainer, LEARNING_RATE*0.5*(1+math.cos(math.PI*done))); progress.err != nil {
		return false
	}
	augment(progress.images, progress.train, &progress.rng)
	return true
}

// Dataset holds images as bytes, [count, 32, 32, 3] channels last, so the
// training set takes a quarter of the memory floats would.
Dataset :: struct {
	pixels: []u8,
	labels: []int,
}

dataset_release :: proc(dataset: ^Dataset) {
	delete(dataset.pixels)
	delete(dataset.labels)
}

// load_dataset reads CIFAR-10 binary batch files from dir, printing what went
// wrong on failure. Each image is a label byte, then 1024 red, 1024 green and
// 1024 blue bytes in row-major order, which are reordered channels last.
load_dataset :: proc(dir: string, names: []string) -> (dataset: Dataset, ok: bool) {
	RECORD :: 1 + PIXELS
	pixels: [dynamic]u8
	labels: [dynamic]int
	for name in names {
		path := fmt.tprintf("%s/%s", dir, name)
		contents, err := os.read_entire_file(path, context.allocator)
		if err != nil {
			fmt.eprintf("%s: %v (run scripts/fetch_cifar10.sh to download CIFAR-10)\n", path, err)
			delete(pixels)
			delete(labels)
			return
		}
		defer delete(contents)
		if len(contents) == 0 || len(contents) % RECORD != 0 {
			fmt.eprintf("%s: not a CIFAR-10 batch of %d-byte records\n", path, RECORD)
			delete(pixels)
			delete(labels)
			return
		}
		for start := 0; start < len(contents); start += RECORD {
			record := contents[start:][:RECORD]
			append(&labels, int(record[0]))
			first := len(pixels)
			resize(&pixels, first + PIXELS)
			image := pixels[first:]
			for c in 0..<CHANNELS {
				plane := record[1 + c*SIDE*SIDE:][:SIDE*SIDE]
				for value, i in plane {
					image[i*CHANNELS + c] = value
				}
			}
		}
	}
	return Dataset{pixels = pixels[:], labels = labels[:]}, true
}

// augment writes every training image into images, normalized, shifted by up
// to SHIFT pixels each way and mirrored half the time.
augment :: proc(images: ^tensor.Tensor(f32), dataset: Dataset, rng: ^rand.Default_Random_State) {
	generator := rand.default_random_generator(rng)
	for i in 0..<len(dataset.labels) {
		dy := rand.int_max(2*SHIFT+1, generator) - SHIFT
		dx := rand.int_max(2*SHIFT+1, generator) - SHIFT
		mirror := rand.int_max(2, generator) == 1
		write_image(images.data[i*PIXELS:][:PIXELS], dataset.pixels[i*PIXELS:][:PIXELS], dy, dx, mirror)
	}
}

// write_image normalizes source into target, mirrored left to right when
// mirror is set and then moved down dy and right dx pixels. Pixels moved in
// from outside the image are zero, the mean.
write_image :: proc(target: []f32, source: []u8, dy, dx: int, mirror: bool) {
	mean, std := MEAN, STD
	for y in 0..<SIDE {
		for x in 0..<SIDE {
			pixel := target[(y*SIDE + x)*CHANNELS:][:CHANNELS]
			sy, sx := y-dy, x-dx
			if sy < 0 || sy >= SIDE || sx < 0 || sx >= SIDE {
				pixel[0], pixel[1], pixel[2] = 0, 0, 0
				continue
			}
			if mirror {
				sx = SIDE-1 - sx
			}
			for c in 0..<CHANNELS {
				pixel[c] = (f32(source[(sy*SIDE + sx)*CHANNELS + c])/255 - mean[c]) / std[c]
			}
		}
	}
}

// confusion_matrix classifies images EVAL_BATCH at a time and counts, for
// each true class, how often each class was predicted.
confusion_matrix :: proc(network: ann.Network(f32), images: tensor.Tensor(f32), labels: []int) -> (confusion: [CLASSES][CLASSES]int, err: ann.Error) {
	shape, strides: [tensor.MAX_RANK]int
	for start := 0; start < len(labels); start += EVAL_BATCH {
		batch := min(EVAL_BATCH, len(labels)-start)
		view := tensor.slice_view(tensor.as_view(images), 0, start, batch, 1, shape[:], strides[:]) or_return
		output := ann.forward_views(network, view) or_return
		defer tensor.release(&output)
		classes := tensor.argmax_axis(output, -1) or_return
		defer tensor.release(&classes)
		for class, i in classes.data {
			confusion[labels[start+i]][class] += 1
		}
	}
	return confusion, nil
}

// print_confusion prints each class's accuracy and the counts it was
// predicted as, one row per true class.
print_confusion :: proc(confusion: [CLASSES][CLASSES]int) {
	names := CLASS_NAMES
	fmt.printf("\n%-10s %8s   predicted as\n", "class", "accuracy")
	fmt.printf("%-10s %8s  ", "", "")
	for name in names {
		fmt.printf(" %5s", name[:min(5, len(name))])
	}
	fmt.println()
	correct, total := 0, 0
	for row, class in confusion {
		count := 0
		for value in row {
			count += value
		}
		correct += row[class]
		total += count
		fmt.printf("%-10s %8s  ", names[class], fmt.tprintf("%.1f%%", 100*f64(row[class])/f64(max(count, 1))))
		for value in row {
			fmt.printf(" %5s", fmt.tprint(value))
		}
		fmt.println()
	}
	fmt.printf("\ntest accuracy %.2f%%\n", 100*f64(correct)/f64(max(total, 1)))
}
