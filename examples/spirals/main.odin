// Classifies three interleaved spirals with a ReLU network, softmax
// cross-entropy and Adam, then draws the learned decision regions.
//
//     odin run examples/spirals -o:speed
package spirals

import "core:fmt"
import "core:math"
import "core:math/rand"
import ann "../.."
import tensor "../../../odin-tensor"

CLASSES :: 3
POINTS_PER_CLASS :: 100
SAMPLES :: CLASSES * POINTS_PER_CLASS

main :: proc() {
	// Each arm winds outward from the origin with a little angular noise.
	points: [SAMPLES * 2]f32
	labels: [SAMPLES]int
	rng := rand.create(11)
	generator := rand.default_random_generator(&rng)
	for class in 0..<CLASSES {
		for i in 0..<POINTS_PER_CLASS {
			sample := class*POINTS_PER_CLASS + i
			radius := f64(i) / POINTS_PER_CLASS
			angle := f64(class)*2*math.PI/CLASSES + 4*radius + rand.float64_range(-0.2, 0.2, generator)
			points[2*sample] = f32(radius * math.cos(angle))
			points[2*sample+1] = f32(radius * math.sin(angle))
			labels[sample] = class
		}
	}
	inputs, err := tensor.from_slice(f32, points[:], []int{SAMPLES, 2})
	if err != .None {
		fmt.eprintln("inputs:", err)
		return
	}
	defer tensor.release(&inputs)
	targets, targets_err := ann.one_hot(f32, labels[:], CLASSES)
	if targets_err != nil {
		fmt.eprintln("targets:", targets_err)
		return
	}
	defer tensor.release(&targets)

	network := ann.new_network(f32, seed = 2)
	defer ann.release(&network)
	widths := []int{2, 64, 64, CLASSES}
	activations := []ann.Activation{.ReLU, .ReLU, .Softmax}
	for activation, i in activations {
		if err := ann.dense(&network, widths[i], widths[i+1], activation); err != nil {
			fmt.eprintln("dense:", err)
			return
		}
	}
	ann.print_summary(network)

	trainer, trainer_err := ann.trainer_init(f32, .Cross_Entropy, ann.adam(learning_rate = 0.01))
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return
	}
	defer ann.trainer_release(&trainer)

	report :: proc(epoch: int, loss: f64, _: rawptr) -> bool {
		if epoch % 50 == 0 {
			fmt.printf("epoch %3d  cross-entropy %.4f\n", epoch, loss)
		}
		return true
	}
	options := ann.Fit_Options{epochs = 300, batch_size = 32, shuffle = true, seed = 3, on_epoch = report}
	if _, fit_err := ann.fit(&trainer, &network, inputs, targets, options); fit_err != nil {
		fmt.eprintln("fit:", fit_err)
		return
	}

	probabilities, forward_err := ann.forward(network, inputs)
	if forward_err != nil {
		fmt.eprintln("forward:", forward_err)
		return
	}
	defer tensor.release(&probabilities)
	fraction, accuracy_err := ann.accuracy(probabilities, targets)
	if accuracy_err != nil {
		fmt.eprintln("accuracy:", accuracy_err)
		return
	}
	fmt.printf("training accuracy %.1f%%\n\n", 100*fraction)
	if err := draw_regions(network, points[:], labels[:]); err != nil {
		fmt.eprintln("draw:", err)
	}
}

// draw_regions classifies every cell of a character grid in one batched
// forward pass, shading each class's region and marking the training points
// with their class digit.
draw_regions :: proc(network: ann.Network(f32), points: []f32, labels: []int) -> ann.Error {
	WIDTH :: 64
	HEIGHT :: 28
	EXTENT :: 1.1
	cells: [HEIGHT * WIDTH * 2]f32
	for row in 0..<HEIGHT {
		for column in 0..<WIDTH {
			cell := row*WIDTH + column
			cells[2*cell] = f32(-EXTENT + 2*EXTENT*f64(column)/(WIDTH-1))
			cells[2*cell+1] = f32(EXTENT - 2*EXTENT*f64(row)/(HEIGHT-1))
		}
	}
	grid := tensor.from_slice(f32, cells[:], []int{HEIGHT * WIDTH, 2}) or_return
	defer tensor.release(&grid)
	classes := ann.predict_classes(network, grid) or_return
	defer tensor.release(&classes)

	shades := [CLASSES]u8{'.', '-', '~'}
	canvas: [HEIGHT][WIDTH]u8
	for row in 0..<HEIGHT {
		for column in 0..<WIDTH {
			canvas[row][column] = shades[classes.data[row*WIDTH+column]]
		}
	}
	for label, i in labels {
		column := int(math.round((f64(points[2*i]) + EXTENT) / (2*EXTENT) * (WIDTH-1)))
		row := int(math.round((EXTENT - f64(points[2*i+1])) / (2*EXTENT) * (HEIGHT-1)))
		canvas[row][column] = u8('0' + label)
	}
	fmt.println("regions: class 0 '.', class 1 '-', class 2 '~'; digits are training points")
	for &line in canvas {
		fmt.println(string(line[:]))
	}
	return nil
}
