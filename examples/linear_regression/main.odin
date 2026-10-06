// Fits y = 3x - 2 from noisy samples with one linear unit and plain SGD,
// calling train_step directly instead of fit.
//
//     odin run examples/linear_regression
package linear_regression

import "core:fmt"
import "core:math/rand"
import ann "../.."
import tensor "../../../odin-tensor"

main :: proc() {
	// 64 samples on [-2, 2] with uniform noise of +/-0.25.
	SAMPLES :: 64
	xs, ys: [SAMPLES]f32
	rng := rand.create(42)
	generator := rand.default_random_generator(&rng)
	for i in 0..<SAMPLES {
		xs[i] = -2 + 4*f32(i)/(SAMPLES-1)
		ys[i] = 3*xs[i] - 2 + rand.float32_range(-0.25, 0.25, generator)
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

	network := ann.new_network(f32)
	defer ann.release(&network)
	if err := ann.dense(&network, 1, 1); err != nil {
		fmt.eprintln("dense:", err)
		return
	}
	trainer, trainer_err := ann.trainer_init(f32, .Mean_Squared_Error, ann.sgd(0.1))
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return
	}
	defer ann.trainer_release(&trainer)

	for step in 0..<100 {
		loss, step_err := ann.train_step(&trainer, &network, inputs, targets)
		if step_err != nil {
			fmt.eprintln("train_step:", step_err)
			return
		}
		if step % 20 == 0 || step == 99 {
			fmt.printf("step %3d  mse %.5f\n", step, loss)
		}
	}
	fmt.printf("learned y = %.3fx %+.3f (true y = 3x - 2)\n", network.layers[0].weights.data[0], network.layers[0].bias.data[0])
}
