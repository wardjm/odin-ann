// Trains a convolutional autoencoder to remove noise from the MNIST digits.
// The encoder shrinks each 28x28 image to 7x7 with strided convolutions, and
// the decoder grows it back with upsample2d and padded convolutions. Training
// pairs a noisy copy of each image, drawn afresh every epoch from on_epoch,
// with the clean image as the target, under binary cross-entropy. It reports
// the test images' mean squared error after each epoch, saves the network as
// denoise.bin in the data directory, and draws a few noisy test digits next to
// the network's output and the originals. Fetch the data first:
//
//     scripts/fetch_mnist.sh
//     odin run examples/mnist_denoise -o:speed
//     odin run examples/mnist_denoise -o:speed -- path/to/mnist 5 4
//
// The optional arguments are the data directory, the number of epochs and the
// number of threads, which defaults to the physical core count. With 0
// epochs, the example loads the saved network instead of training one.
package mnist_denoise

import "core:fmt"
import "core:math/rand"
import "core:strings"
import "core:time"
import ann "../.."
import tensor "../../../odin-tensor"
import "../mnist/digits"

// NOISE is the standard deviation of the Gaussian noise added to each pixel,
// before clamping to [0, 1].
NOISE :: 0.5
// SHOWN test digits are drawn at the end.
SHOWN :: 3

main :: proc() {
	arguments, ok := digits.parse_arguments(epochs = 3, min_epochs = 0)
	if !ok do return
	defer tensor.set_thread_count(1)

	test, test_ok := digits.load_dataset(arguments.dir, "t10k")
	if !test_ok do return
	defer digits.dataset_release(&test)
	// The test images get one fixed draw of noise, so every evaluation sees
	// the same inputs.
	noisy_test, noisy_err := tensor.copy_view(tensor.as_view(test.images))
	if noisy_err != .None {
		fmt.eprintln("noisy test images:", noisy_err)
		return
	}
	defer tensor.release(&noisy_test)
	test_rng := rand.create(4)
	add_noise(&noisy_test, test.images, &test_rng)
	saved_path := fmt.tprintf("%s/denoise.bin", arguments.dir)

	network: ann.Network(f32)
	defer ann.release(&network)
	if arguments.epochs == 0 {
		err: ann.Error
		if network, err = ann.load(f32, saved_path); err != nil {
			fmt.eprintf("%s: %v (train with a positive epoch count to save one)\n", saved_path, err)
			return
		}
		fmt.printf("loaded %s\n\n", saved_path)
		ann.print_summary(network)
		denoised_error, eval_err := test_error(network, noisy_test, test.images)
		if eval_err != nil {
			fmt.eprintln("evaluation:", eval_err)
			return
		}
		fmt.printf("\ntest error %.4f, against %.4f for the noisy images\n", denoised_error, squared_error(noisy_test.data, test.images.data))
	} else {
		train, train_ok := digits.load_dataset(arguments.dir, "train")
		if !train_ok do return
		defer digits.dataset_release(&train)
		if !train_network(&network, train, test, noisy_test, arguments.epochs) do return
		if err := ann.save(network, saved_path); err != nil {
			fmt.eprintf("%s: %v\n", saved_path, err)
			return
		}
		fmt.printf("\nsaved the network to %s; pass 0 epochs to load it\n", saved_path)
	}
	show_examples(network, noisy_test, test.images)
}

// build makes 28x28x1 -> 14x14x16 -> 7x7x32 -> 14x14x32 -> 14x14x16 ->
// 28x28x16 -> 28x28x1. The output's sigmoid gives pixels in [0, 1], and
// binary cross-entropy compares them with the clean pixels.
build :: proc(network: ^ann.Network(f32)) -> ann.Error {
	ann.set_input_shape(network, []int{digits.SIDE, digits.SIDE, 1}) or_return
	ann.conv2d(network, 1, 16, 3, .ReLU, stride = 2, padding = 1) or_return
	ann.conv2d(network, 16, 32, 3, .ReLU, stride = 2, padding = 1) or_return
	ann.upsample2d(network) or_return
	ann.conv2d(network, 32, 16, 3, .ReLU, padding = 1) or_return
	ann.upsample2d(network) or_return
	return ann.conv2d(network, 16, 1, 3, .Sigmoid, padding = 1)
}

// train_network builds the network and trains it on noisy copies of the
// training images for epochs, reporting the test error after each one. It
// prints any error and returns false.
train_network :: proc(network: ^ann.Network(f32), train, test: digits.Dataset, noisy_test: tensor.Tensor(f32), epochs: int) -> bool {
	noisy_train, noisy_err := tensor.copy_view(tensor.as_view(train.images))
	if noisy_err != .None {
		fmt.eprintln("noisy training images:", noisy_err)
		return false
	}
	defer tensor.release(&noisy_train)
	progress := Progress{
		network     = network,
		clean_train = train.images,
		noisy_train = &noisy_train,
		clean_test  = test.images,
		noisy_test  = noisy_test,
		rng         = rand.create(3),
		start       = time.now(),
	}
	add_noise(&noisy_train, train.images, &progress.rng)
	fmt.printf("%d training and %d test images, %d threads\n\n", len(train.labels), len(test.labels), tensor.thread_count())

	network^ = ann.new_network(f32, seed = 1)
	if err := build(network); err != nil {
		fmt.eprintln("network:", err)
		return false
	}
	ann.print_summary(network^)
	fmt.printf("\nnoisy test images have error %.4f\n", squared_error(noisy_test.data, test.images.data))

	trainer, trainer_err := ann.trainer_init(f32, .Binary_Cross_Entropy, ann.adam(learning_rate = 0.002))
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return false
	}
	defer ann.trainer_release(&trainer)

	options := ann.Fit_Options{epochs = epochs, batch_size = 64, shuffle = true, seed = 2, on_epoch = report_epoch, user_data = &progress}
	if _, err := ann.fit(&trainer, network, noisy_train, train.images, options); err != nil {
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
	network:                 ^ann.Network(f32),
	clean_train:             tensor.Tensor(f32),
	noisy_train:             ^tensor.Tensor(f32),
	clean_test, noisy_test:  tensor.Tensor(f32),
	rng:                     rand.Default_Random_State,
	start:                   time.Time,
	err:                     ann.Error,
}

// report_epoch prints the epoch's training loss and the test error, then
// draws new noise for the next epoch, so the network never sees the same
// noisy image twice.
report_epoch :: proc(epoch: int, loss: f64, user_data: rawptr) -> bool {
	progress := (^Progress)(user_data)
	denoised_error, err := test_error(progress.network^, progress.noisy_test, progress.clean_test)
	if err != nil {
		progress.err = err
		return false
	}
	elapsed := time.duration_seconds(time.since(progress.start))
	fmt.printf("epoch %d  loss %.4f  test error %.4f  (%.0fs)\n", epoch+1, loss, denoised_error, elapsed)
	add_noise(progress.noisy_train, progress.clean_train, &progress.rng)
	return true
}

// add_noise sets noisy to clean plus Gaussian noise, clamped to [0, 1].
add_noise :: proc(noisy: ^tensor.Tensor(f32), clean: tensor.Tensor(f32), rng: ^rand.Default_Random_State) {
	generator := rand.default_random_generator(rng)
	for pixel, i in clean.data {
		noisy.data[i] = clamp(pixel + f32(NOISE*rand.norm_float64(generator)), 0, 1)
	}
}

// test_error denoises the noisy test images digits.EVAL_BATCH at a time and
// returns the mean squared difference from the clean ones.
test_error :: proc(network: ann.Network(f32), noisy, clean: tensor.Tensor(f32)) -> (error: f64, err: ann.Error) {
	shape, strides: [tensor.MAX_RANK]int
	count := noisy.shape[0]
	pixels := len(noisy.data) / count
	for start := 0; start < count; start += digits.EVAL_BATCH {
		batch := min(digits.EVAL_BATCH, count-start)
		images := tensor.slice_view(tensor.as_view(noisy), 0, start, batch, 1, shape[:], strides[:]) or_return
		output := ann.forward_views(network, images) or_return
		defer tensor.release(&output)
		error += squared_error(output.data, clean.data[start*pixels:][:batch*pixels]) * f64(batch)
	}
	return error / f64(count), nil
}

// squared_error is the mean squared difference between a and b.
squared_error :: proc(a, b: []f32) -> f64 {
	total: f64
	for value, i in a {
		difference := f64(value - b[i])
		total += difference * difference
	}
	return total / f64(len(a))
}

// show_examples draws the first SHOWN noisy test digits, the network's
// output for them and the originals side by side.
show_examples :: proc(network: ann.Network(f32), noisy, clean: tensor.Tensor(f32)) {
	shape, strides: [tensor.MAX_RANK]int
	images, slice_err := tensor.slice_view(tensor.as_view(noisy), 0, 0, SHOWN, 1, shape[:], strides[:])
	if slice_err != .None do return
	output, err := ann.forward_views(network, images)
	if err != nil do return
	defer tensor.release(&output)
	PIXELS :: digits.SIDE * digits.SIDE
	fmt.printf("\n%-*s   %-*s   %s\n", digits.SIDE, "noisy", digits.SIDE, "denoised", "original")
	for i in 0..<SHOWN {
		draw([3][]f32{noisy.data[i*PIXELS:][:PIXELS], output.data[i*PIXELS:][:PIXELS], clean.data[i*PIXELS:][:PIXELS]})
	}
}

// draw prints images side by side, one character per pixel horizontally and
// per two rows vertically, which keeps a terminal's tall characters roughly
// square, shading by the pair's mean.
draw :: proc(images: [3][]f32) {
	ramp := " .:-=+*#%@"
	line: strings.Builder
	strings.builder_init(&line, context.temp_allocator)
	for row := 0; row < digits.SIDE; row += 2 {
		strings.builder_reset(&line)
		for image, i in images {
			if i > 0 {
				strings.write_string(&line, "   ")
			}
			for x in 0..<digits.SIDE {
				value := (image[row*digits.SIDE + x] + image[(row+1)*digits.SIDE + x]) / 2
				strings.write_byte(&line, ramp[clamp(int(value*f32(len(ramp))), 0, len(ramp)-1)])
			}
		}
		fmt.println(strings.to_string(line))
	}
	fmt.println()
}
