// Trains a deeper convolutional network on the MNIST digits than
// examples/mnist does: four 3x3 convolutions, each followed by batch
// normalization and then ReLU, and dropout before the classifier. The learning
// rate halves after every epoch, set from the on_epoch callback. It reports
// test accuracy after each epoch, saves the network as deep_cnn.bin in the
// data directory, and draws a test digit it gets wrong. Fetch the data first:
//
//     scripts/fetch_mnist.sh
//     odin run examples/mnist_deep -o:speed
//     odin run examples/mnist_deep -o:speed -- path/to/mnist 5 4
//
// The optional arguments are the data directory, the number of epochs and the
// number of threads, which defaults to the physical core count.
package mnist_deep

import "core:fmt"
import "core:time"
import ann "../.."
import tensor "../../../odin-tensor"
import "../mnist/digits"

LEARNING_RATE :: 0.003

main :: proc() {
	arguments, ok := digits.parse_arguments(epochs = 3, min_epochs = 1)
	if !ok do return
	defer tensor.set_thread_count(1)

	train, train_ok := digits.load_dataset(arguments.dir, "train")
	if !train_ok do return
	defer digits.dataset_release(&train)
	test, test_ok := digits.load_dataset(arguments.dir, "t10k")
	if !test_ok do return
	defer digits.dataset_release(&test)
	targets, targets_err := ann.one_hot(f32, train.labels, digits.CLASSES)
	if targets_err != nil {
		fmt.eprintln("targets:", targets_err)
		return
	}
	defer tensor.release(&targets)
	fmt.printf("%d training and %d test images, %d threads\n\n", len(train.labels), len(test.labels), tensor.thread_count())

	network := ann.new_network(f32, seed = 1)
	defer ann.release(&network)
	if err := build(&network); err != nil {
		fmt.eprintln("network:", err)
		return
	}
	ann.print_summary(network)
	fmt.println()

	trainer, trainer_err := ann.trainer_init(f32, .Cross_Entropy, ann.adam(learning_rate = LEARNING_RATE), seed = 3)
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return
	}
	defer ann.trainer_release(&trainer)

	progress := Progress{base = {network = &network, test = &test, start = time.now()}, trainer = &trainer}
	options := ann.Fit_Options{epochs = arguments.epochs, batch_size = 64, shuffle = true, seed = 2, on_epoch = report_epoch, user_data = &progress}
	if _, err := ann.fit(&trainer, &network, train.images, targets, options); err != nil {
		fmt.eprintln("fit:", err)
		return
	}
	if progress.base.err != nil {
		fmt.eprintln("evaluation:", progress.base.err)
		return
	}

	saved_path := fmt.tprintf("%s/deep_cnn.bin", arguments.dir)
	if err := ann.save(network, saved_path); err != nil {
		fmt.eprintf("%s: %v\n", saved_path, err)
		return
	}
	fmt.printf("\nsaved the network to %s\n", saved_path)
	digits.show_mistake(network, test)
}

// build makes 28x28x1 -> 26x26x16 -> 24x24x16 -> 12x12x16 -> 10x10x32 ->
// 8x8x32 -> 4x4x32 -> 512 -> 10. Batch normalization supplies each
// convolution's activation, so the convolutions use Identity.
build :: proc(network: ^ann.Network(f32)) -> ann.Error {
	ann.set_input_shape(network, []int{digits.SIDE, digits.SIDE, 1}) or_return
	ann.conv2d(network, 1, 16, 3) or_return
	ann.batch_norm(network, .ReLU) or_return
	ann.conv2d(network, 16, 16, 3) or_return
	ann.batch_norm(network, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.conv2d(network, 16, 32, 3) or_return
	ann.batch_norm(network, .ReLU) or_return
	ann.conv2d(network, 32, 32, 3) or_return
	ann.batch_norm(network, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.flatten(network) or_return
	ann.dropout(network, 0.3) or_return
	return ann.dense(network, 4*4*32, digits.CLASSES, .Softmax)
}

Progress :: struct {
	base:    digits.Progress,
	trainer: ^ann.Trainer(f32),
}

// report_epoch prints the epoch's results, then halves the learning rate.
report_epoch :: proc(epoch: int, loss: f64, user_data: rawptr) -> bool {
	progress := (^Progress)(user_data)
	if !digits.report_epoch(epoch, loss, &progress.base) {
		return false
	}
	progress.base.err = ann.set_learning_rate(progress.trainer, LEARNING_RATE/f64(int(2) << uint(epoch)))
	return progress.base.err == nil
}
