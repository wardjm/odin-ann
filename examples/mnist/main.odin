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

import "core:fmt"
import "core:time"
import ann "../.."
import tensor "../../../odin-tensor"
import "digits"

main :: proc() {
	arguments, ok := digits.parse_arguments(epochs = 3, min_epochs = 0)
	if !ok do return
	defer tensor.set_thread_count(1)

	test, test_ok := digits.load_dataset(arguments.dir, "t10k")
	if !test_ok do return
	defer digits.dataset_release(&test)
	saved_path := fmt.tprintf("%s/cnn.bin", arguments.dir)

	network: ann.Network(f32)
	defer ann.release(&network)
	if arguments.epochs == 0 {
		err: ann.Error
		if network, err = ann.load(f32, saved_path); err != nil {
			fmt.eprintf("%s: %v (train with a positive epoch count to save one)\n", saved_path, err)
			return
		}
		correct, eval_err := digits.count_correct(network, test)
		if eval_err != nil {
			fmt.eprintln("evaluation:", eval_err)
			return
		}
		fmt.printf("loaded %s\n\n", saved_path)
		ann.print_summary(network)
		fmt.printf("\ntest accuracy %.2f%%\n", 100*f64(correct)/f64(len(test.labels)))
	} else {
		train, train_ok := digits.load_dataset(arguments.dir, "train")
		if !train_ok do return
		defer digits.dataset_release(&train)
		if !train_network(&network, train, &test, arguments.epochs) do return
		if err := ann.save(network, saved_path); err != nil {
			fmt.eprintf("%s: %v\n", saved_path, err)
			return
		}
		fmt.printf("\nsaved the network to %s; pass 0 epochs to load it\n", saved_path)
	}
	digits.show_mistake(network, test)
}

// train_network builds the network and trains it on train for epochs,
// reporting test accuracy after each one. It prints any error and returns
// false.
train_network :: proc(network: ^ann.Network(f32), train: digits.Dataset, test: ^digits.Dataset, epochs: int) -> bool {
	targets, targets_err := ann.one_hot(f32, train.labels, digits.CLASSES)
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

	progress := digits.Progress{network = network, test = test, start = time.now()}
	options := ann.Fit_Options{epochs = epochs, batch_size = 64, shuffle = true, seed = 2, on_epoch = digits.report_epoch, user_data = &progress}
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
	ann.set_input_shape(network, []int{digits.SIDE, digits.SIDE, 1}) or_return
	ann.conv2d(network, 1, 8, 5, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.conv2d(network, 8, 16, 5, .ReLU) or_return
	ann.max_pool2d(network) or_return
	ann.flatten(network) or_return
	return ann.dense(network, 4*4*16, digits.CLASSES, .Softmax)
}
