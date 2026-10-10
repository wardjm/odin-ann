# odin-ann

A small Odin neural-network library built on the [`odin-tensor`](https://github.com/wardjm/odin-tensor) package. It covers sequential dense networks, five activations, three losses, SGD with momentum and Adam/AdamW, and a minibatch training loop. Gradients come from the tensor library's reverse-mode autodiff tape.

## Features

- Dense layers with Identity, ReLU, Sigmoid, Tanh and Softmax activations, generic over `f32` or `f64`. Weights are drawn from a seeded generator, with He scaling for ReLU and Glorot otherwise.
- Image layers on channels-last `[height, width, channels]` samples: valid, stride-1 `conv2d`, non-overlapping `max_pool2d`, and `flatten` to feed a dense layer.
- `forward` takes a single sample `[features]`, a batch `[batch, features]`, or any number of leading batch axes. A `[2, 5, features]` input gives `[2, 5, outputs]`.
- Mean-squared error, binary cross-entropy and categorical cross-entropy. The cross-entropy losses work from the final layer's logits, so they stay finite when outputs saturate.
- `SGD` (optional momentum) and `Adam` (bias-corrected; a nonzero `weight_decay` makes it AdamW), with learning-rate changes between steps.
- `train_step` for hand-written loops, and `fit` with shuffled or in-order minibatches, a per-epoch callback, and early stopping.
- `evaluate`, `accuracy`, `predict_classes` and `one_hot`.

## Installation

odin-ann imports odin-tensor as `../odin-tensor`, so the two repositories must be checked out next to each other, under those directory names:

```sh
mkdir libs && cd libs
git clone https://github.com/wardjm/odin-ann.git
git clone https://github.com/wardjm/odin-tensor.git
```

Your program must import odin-tensor from that same checkout. odin-ann's API takes and returns `tensor.Tensor` values, so you will need the import. A second copy elsewhere, such as a vendored one, does not work: two packages named `tensor` cannot be in one build, and the compiler stops with `Duplicate declaration of 'package tensor'`.

The simplest way to do this is a collection that points at the directory holding both checkouts:

```sh
odin run my_app -collection:libs=path/to/libs
```

```odin
import ann "libs:odin-ann"
import tensor "libs:odin-tensor"
```

Relative imports such as `"../libs/odin-ann"` also work, provided both point into the same `libs` directory.

## Quick start

The directory names contain hyphens, so both imports need explicit names. Replace the paths with the ones your project uses, as described in [Installation](#installation):

```odin
package main

import "core:fmt"
import ann "path/to/odin-ann"
import tensor "path/to/odin-tensor"

main :: proc() {
    inputs, err := tensor.from_slice(f32, []f32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
    if err != .None do return
    defer tensor.release(&inputs)
    targets: tensor.Tensor(f32)
    targets, err = tensor.from_slice(f32, []f32{0, 1, 1, 0}, []int{4, 1})
    if err != .None do return
    defer tensor.release(&targets)

    // 2 -> 4 (tanh) -> 1 (sigmoid), with reproducible initial weights.
    network := ann.new_network(f32, seed = 3)
    defer ann.release(&network)
    if ann.dense(&network, 2, 4, .Tanh) != nil do return
    if ann.dense(&network, 4, 1, .Sigmoid) != nil do return

    trainer, trainer_err := ann.trainer_init(f32, .Binary_Cross_Entropy, ann.adam(learning_rate = 0.05))
    if trainer_err != nil do return
    defer ann.trainer_release(&trainer)

    loss, fit_err := ann.fit(&trainer, &network, inputs, targets, {epochs = 500})
    if fit_err != nil do return
    fmt.println("final loss:", loss)

    predictions, forward_err := ann.forward(network, inputs)
    if forward_err != nil do return
    defer tensor.release(&predictions)
    fmt.println(predictions.data) // close to [0, 1, 1, 0]
}
```

## Examples

Run these from the repository root:

| Command | Shows |
|---|---|
| `odin run examples/linear_regression` | One linear unit, plain SGD, and a hand-written `train_step` loop. |
| `odin run examples/xor` | Binary cross-entropy, Adam, `print_summary`, and early stopping from the `on_epoch` callback. |
| `odin run examples/sine -o:speed` | Regression with shuffled minibatches and a learning-rate schedule set from the callback, plotted as text. |
| `odin run examples/spirals -o:speed` | Three-class softmax cross-entropy, `one_hot`, `accuracy`, and a decision map from one batched `predict_classes` call. |

## API

### Errors

Every fallible procedure returns `ann.Error`, a `#shared_nil` union of `tensor.Tensor_Error` and `ann.Ann_Error`, so `err != nil` checks both and `or_return` passes either along. Shape, bounds and allocation failures come from the tensor library (`.Shape_Mismatch`, `.Allocation_Failed`, ...). `Ann_Error` adds `.Empty_Network`, `.Invalid_Hyperparameter`, `.Loss_Activation_Mismatch`, and `.Network_Changed`, which a trainer returns when the network's layers no longer match its optimizer state.

### Networks

- `new_network(T, seed = 0, allocator)` creates an empty `Network(T)` for a float type `T`. `release(&network)` frees it. Treat networks as move-only.
- `set_input_shape(&network, shape)` fixes the shape of one input sample, such as `[]int{28, 28, 1}`. A network that starts with an image layer needs it; a first `dense` layer sets `[inputs]` itself. Calling it after layers are added returns `.Shape_Mismatch` unless the shape is unchanged.
- `dense(&network, inputs, outputs, activation = .Identity)` appends a fully connected layer and returns `.Shape_Mismatch` unless its input is `[inputs]`.
- `conv2d(&network, input_channels, output_channels, kernel_size, activation = .Identity)` appends a valid, stride-1 convolution with square kernels. `[h, w, input_channels]` becomes `[h-kernel_size+1, w-kernel_size+1, output_channels]`; weights are `[kernel_size, kernel_size, input_channels, output_channels]` and biases `[output_channels]`.
- `max_pool2d(&network, pool_size = 2)` takes the maximum of each non-overlapping `pool_size` × `pool_size` window, so `[h, w, c]` becomes `[h/pool_size, w/pool_size, c]`, rounding down. `flatten(&network)` turns each sample into one vector in row-major order. Neither has parameters.
- Each `Layer(T)` holds its `kind`, `weights`, `bias`, `activation`, `pool_size`, and the `input_shape` and `output_shape` of one sample. The parameters are ordinary tensors, so you can set or load them directly.
- `input_features_of` and `output_features_of` give the last axis of one input or output sample. `parameter_count` and `print_summary` describe a network.
- `forward(network, input, allocator)` and `forward_views` return a new tensor owned by the caller. The input's trailing axes must match the network's `input_shape`; any leading axes are batch axes, so a batch of MNIST images is `[batch, 28, 28, 1]`. Each dense layer allocates one tensor and applies its bias and activation in place.
- `activate_in_place` and `softmax_in_place` apply an activation to any tensor. Softmax normalizes over the last axis and subtracts the row maximum first.

### Losses

| `Loss` | Final activation | Targets |
|---|---|---|
| `.Mean_Squared_Error` | any | same shape as the output |
| `.Binary_Cross_Entropy` | `.Sigmoid` | values in [0, 1] |
| `.Cross_Entropy` | `.Softmax` | each row a distribution over the last axis, such as `one_hot(T, labels, classes)` |

Each loss averages over the batch. The two cross-entropy losses replace the final activation with a stable formula on its logits: `max(z, 0) - z*y + log(1 + exp(-|z|))` for binary and log-sum-exp for categorical. Any other pairing returns `.Loss_Activation_Mismatch`. `evaluate(network, loss, inputs, targets)` computes the loss exactly as training does, without changing anything. `mean_squared_error(predictions, targets)` works on any two tensors.

### Training

```odin
trainer, err := ann.trainer_init(f32, .Cross_Entropy, ann.adam(learning_rate = 0.01))
defer ann.trainer_release(&trainer)
loss, step_err := ann.train_step(&trainer, &network, inputs, targets)
```

- `sgd(learning_rate, momentum = 0, weight_decay = 0)` and `adam(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0)` build an `Optimizer`, a union of the `SGD` and `Adam` structs. Weight decay is decoupled: each step scales parameters by `1 - learning_rate*weight_decay`. Out-of-range values return `.Invalid_Hyperparameter` from `trainer_init`.
- A `Trainer(T)` owns one autodiff tape, which it resets and rebuilds every step, and the optimizer's moment buffers, which it creates on the first step. You pass the network to each call instead of storing it in the trainer, so the network can move freely. If you add or reshape layers afterwards, `train_step` returns `.Network_Changed`.
- `train_step(&trainer, &network, inputs, targets)` runs forward, backward and one optimizer update. Its result is the loss from before the update. `train_step_views` takes borrowed views. `trainer.steps` counts updates, and `set_learning_rate` changes the step size without resetting the moment buffers.
- `fit(&trainer, &network, inputs, targets, options)` trains over a dataset whose first axis indexes samples and returns the mean loss of the last epoch. `Fit_Options` sets `epochs`, `batch_size` (0 means the whole dataset), and `shuffle` with `seed`. It also takes `on_epoch(epoch, loss, user_data) -> bool`, which runs after each epoch; returning false stops early. In-order batches are zero-copy `slice_view`s of the dataset. Shuffled batches are copied into two reused buffers.

### Metrics

- `accuracy(predictions, targets)` compares the argmax over the last axis, or thresholds at 0.5 when there is a single output.
- `predict_classes(network, inputs)` returns a `Tensor(int)` of argmax classes.
- `one_hot(T, labels, classes)` builds `[len(labels), classes]` targets.

## How training uses the tape

Each step records the network on the trainer's tape. The input enters as a constant leaf (`tape_add_leaf_view`, no gradient). Every weight and bias enters as a borrowed leaf, and each dense layer becomes `tape_matmul` → `tape_add` → activation. Matmul and add broadcast, so leading batch axes need no special handling.

A convolution is `tape_im2col`, which gathers each output position's input window into one row, then a single `tape_matmul` of those rows by the weights viewed as `[kernel_size²·input_channels, output_channels]`, and a bias `tape_add`. `forward` computes the same product the same way, so it agrees with training exactly. Max pooling reshapes each sample into `[h/p, p, w/p, p, c]` blocks and takes `tape_max_axis` twice, and `flatten` is a `tape_reshape`.

The loss itself is never recorded. Its value and its gradient with respect to the network output are computed directly with tensor kernels. That gradient seeds `tape_backward_with_seed` at the output, or at the final logits for the cross-entropy losses. The fused gradients are `sigmoid(z) - y` and `softmax(z) * sum(y) - y`, computed exactly, without differentiating through `log(sigmoid)`.

A Softmax in a hidden layer, or before mean-squared error, is recorded as `exp(x - max) / sum`. The row maximum enters as a constant leaf, which is exact because softmax ignores a per-row shift.

## Tests

```sh
odin test . -vet -strict-style -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true
```

The tests compare every loss's gradients, including a fused 3-D batch, a hidden softmax and a network using every image layer, against central finite differences in `f64`. Convolution and pooling are also checked against direct loops. They check that `fit` with in-order batches matches hand-sliced `train_step` calls bit for bit. A failing allocator drives every `.Allocation_Failed` path through building, training, evaluating and measuring a network, and the define makes any leak a failure.

```sh
scripts/check_readme.sh
```

Builds every `odin` block in this README that declares `package main` against both libraries, with `-vet -strict-style`.
