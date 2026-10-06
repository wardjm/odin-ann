// Learns XOR, the classic problem a single linear layer cannot solve, with a
// tanh hidden layer, a sigmoid output, binary cross-entropy and Adam.
//
//     odin run examples/xor
package xor

import "core:fmt"
import ann "../.."
import tensor "../../../odin-tensor"

main :: proc() {
	inputs, err := tensor.from_slice(f64, []f64{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	if err != .None {
		fmt.eprintln("inputs:", err)
		return
	}
	defer tensor.release(&inputs)
	targets: tensor.Tensor(f64)
	targets, err = tensor.from_slice(f64, []f64{0, 1, 1, 0}, []int{4, 1})
	if err != .None {
		fmt.eprintln("targets:", err)
		return
	}
	defer tensor.release(&targets)

	network := ann.new_network(f64, seed = 3)
	defer ann.release(&network)
	if err := ann.dense(&network, 2, 4, .Tanh); err != nil {
		fmt.eprintln("dense:", err)
		return
	}
	if err := ann.dense(&network, 4, 1, .Sigmoid); err != nil {
		fmt.eprintln("dense:", err)
		return
	}
	ann.print_summary(network)

	trainer, trainer_err := ann.trainer_init(f64, .Binary_Cross_Entropy, ann.adam(learning_rate = 0.05))
	if trainer_err != nil {
		fmt.eprintln("trainer:", trainer_err)
		return
	}
	defer ann.trainer_release(&trainer)

	report :: proc(epoch: int, loss: f64, _: rawptr) -> bool {
		if epoch % 100 == 0 {
			fmt.printf("epoch %4d  loss %.5f\n", epoch, loss)
		}
		// Stop once the loss is small; the outputs are then well separated.
		return loss > 0.01
	}
	loss, fit_err := ann.fit(&trainer, &network, inputs, targets, {epochs = 2000, on_epoch = report})
	if fit_err != nil {
		fmt.eprintln("fit:", fit_err)
		return
	}
	fmt.printf("stopped after %d steps with loss %.5f\n\n", trainer.steps, loss)

	predictions, forward_err := ann.forward(network, inputs)
	if forward_err != nil {
		fmt.eprintln("forward:", forward_err)
		return
	}
	defer tensor.release(&predictions)
	fmt.println(" a  b | a xor b | network")
	for i in 0..<4 {
		fmt.printf(" %.0f  %.0f |    %.0f    | %.4f\n", inputs.data[2*i], inputs.data[2*i+1], targets.data[i], predictions.data[i])
	}
}
