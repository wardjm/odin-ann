// Approximates sin(x) on [-pi, pi] with two tanh hidden layers, shuffled
// minibatches, Adam and a step learning-rate schedule, then plots the fit.
//
//     odin run examples/sine -o:speed
package sine

import "core:fmt"
import "core:math"
import ann "../.."
import tensor "../../../odin-tensor"

SAMPLES :: 256

main :: proc() {
	xs, ys: [SAMPLES]f32
	for i in 0..<SAMPLES {
		xs[i] = f32(-math.PI + 2*math.PI*f64(i)/(SAMPLES-1))
		ys[i] = math.sin(xs[i])
	}
	inputs, err := tensor.from_slice(f32, xs[:], []int{SAMPLES, 1})
	if err != .None {
		fmt.eprintln("inputs:", err)
		return
	}
	defer tensor.release(&inputs)
	targets: tensor.Tensor(f32)
	targets, err = tensor.from_slice(f32, ys[:], []int{SAMPLES, 1})
	if err != .None {
		fmt.eprintln("targets:", err)
		return
	}
	defer tensor.release(&targets)

	network := ann.new_network(f32, seed = 1)
	defer ann.release(&network)
	widths := []int{1, 32, 32, 1}
	activations := []ann.Activation{.Tanh, .Tanh, .Identity}
	for activation, i in activations {
		if err := ann.dense(&network, widths[i], widths[i+1], activation); err != nil {
			fmt.eprintln("dense:", err)
			return
		}
	}
	ann.print_summary(network)

	trainer, trainer_err := ann.trainer_init(f32, .Mean_Squared_Error, ann.adam(learning_rate = 0.01))
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return
	}
	defer ann.trainer_release(&trainer)

	// The callback receives the trainer through user_data and divides the
	// learning rate by ten every 150 epochs.
	schedule :: proc(epoch: int, loss: f64, user_data: rawptr) -> bool {
		trainer := (^ann.Trainer(f32))(user_data)
		if epoch % 50 == 0 {
			fmt.printf("epoch %3d  mse %.6f\n", epoch, loss)
		}
		if epoch > 0 && epoch % 150 == 0 {
			_ = ann.set_learning_rate(trainer, 0.01 * math.pow(0.1, f64(epoch/150)))
		}
		return true
	}
	options := ann.Fit_Options{epochs = 400, batch_size = 32, shuffle = true, seed = 7, on_epoch = schedule, user_data = &trainer}
	loss, fit_err := ann.fit(&trainer, &network, inputs, targets, options)
	if fit_err != nil {
		fmt.eprintln("fit:", fit_err)
		return
	}
	fmt.printf("final mse %.6f after %d steps\n\n", loss, trainer.steps)

	predictions, forward_err := ann.forward(network, inputs)
	if forward_err != nil {
		fmt.eprintln("forward:", forward_err)
		return
	}
	defer tensor.release(&predictions)
	plot(xs[:], ys[:], predictions.data)
}

// plot draws the target as '.' and the network as '*' ('#' where they meet)
// on a character grid.
plot :: proc(xs, truth, fitted: []f32) {
	WIDTH :: 72
	HEIGHT :: 21
	grid: [HEIGHT][WIDTH]u8
	for &row in grid {
		for &cell in row {
			cell = ' '
		}
	}
	row_of :: proc(y: f32) -> int {
		return clamp(int(math.round((1.1-y) / 2.2 * (HEIGHT-1))), 0, HEIGHT-1)
	}
	for x, i in xs {
		column := int(f32(WIDTH-1) * (x - xs[0]) / (xs[len(xs)-1] - xs[0]))
		grid[row_of(truth[i])][column] = '.'
		cell := &grid[row_of(fitted[i])][column]
		cell^ = cell^ == '.' ? '#' : '*'
	}
	fmt.println("sin(x) '.', network '*', overlap '#'")
	for &row in grid {
		fmt.println(string(row[:]))
	}
}
