# odin-ann

A small Odin neural-network library built on the [`odin-tensor`](https://github.com/wardjm/odin-tensor) package. It covers sequential dense and convolutional networks, dropout and batch normalization, five activations, three losses, SGD with momentum and Adam/AdamW, a minibatch training loop, and saving networks to files. Gradients come from the tensor library's reverse-mode autodiff tape.

## Features

- Dense layers with Identity, ReLU, Sigmoid, Tanh and Softmax activations, generic over `f32` or `f64`. Weights are drawn from a seeded generator, with He scaling for ReLU and Glorot otherwise.
- Image layers on channels-last `[height, width, channels]` samples: `conv2d` with a stride and zero padding, non-overlapping `max_pool2d`, nearest-neighbour `upsample2d`, and `flatten` to feed a dense layer.
- `dropout` and `batch_norm`, which act one way while training and another during inference.
- `forward` takes a single sample `[features]`, a batch `[batch, features]`, or any number of leading batch axes. A `[2, 5, features]` input gives `[2, 5, outputs]`.
- Mean-squared error, binary cross-entropy and categorical cross-entropy. The cross-entropy losses work from the final layer's logits, so they stay finite when outputs saturate.
- `SGD` (optional momentum) and `Adam` (bias-corrected; a nonzero `weight_decay` makes it AdamW), with learning-rate changes between steps.
- `train_step` for hand-written loops, and `fit` with shuffled or in-order minibatches, a per-epoch callback, and early stopping.
- `evaluate`, `accuracy`, `predict_classes` and `one_hot`.
- `save` and `load` store a network's layers and parameters in one compact binary file, so a trained network can be reloaded without the code that built it.

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
| `odin run examples/mnist -o:speed` | A convolutional network (`conv2d`, `max_pool2d`, `flatten`) on the MNIST digits, with test accuracy from `on_epoch` after each epoch. It saves the trained network to `data/mnist/cnn.bin`, and `-- data/mnist 0` loads that file and tests it instead of training. Run `scripts/fetch_mnist.sh` first to download the data, about 11 MB, into `data/mnist`. Three epochs reach about 98.6% test accuracy, in about 20 seconds on 8 cores or 40 on one. It uses every physical core unless a third argument, after the data directory and epoch count, sets the thread count. |
| `odin run examples/mnist_deep -o:speed` | A deeper CNN on MNIST: four 3×3 convolutions, each followed by `batch_norm` with ReLU, and `dropout` before the classifier. The learning rate halves after each epoch through `set_learning_rate` in `on_epoch`. Three epochs reach about 99.2% test accuracy, in under 2 minutes on 8 cores or about 4 on one. It takes the same arguments as `examples/mnist`, which shares its MNIST loading through `examples/mnist/digits`. |
| `odin run examples/mnist_denoise -o:speed` | A convolutional autoencoder that removes Gaussian noise from the MNIST digits. Strided, padded convolutions shrink each image to 7×7, and `upsample2d` with padded convolutions grows it back. It trains on noisy inputs with the clean images as targets under binary cross-entropy, with fresh noise each epoch drawn from `on_epoch`. Three epochs cut the test images' mean squared error from 0.116 to 0.014, in about 2¼ minutes on 8 cores. It draws noisy test digits beside the network's output and the originals, saves the network to `data/mnist/denoise.bin`, and takes the same arguments as `examples/mnist`. |
| `odin run examples/cifar10 -o:speed` | A CNN on CIFAR-10's 32×32 colour photos: three padded 3×3 convolutions with `batch_norm` and ReLU, each followed by `max_pool2d`, then `dropout` and a softmax layer. Each epoch trains on a new augmented copy of the images, shifted by up to 4 pixels and mirrored half the time, written from `on_epoch`, which also lowers the learning rate along a cosine. It prints the test set's confusion matrix and saves the network to `data/cifar10/cifar10.bin`. Run `scripts/fetch_cifar10.sh` first to download the data, about 162 MB, into `data/cifar10`. Fifteen epochs reach about 82% test accuracy, in about 15 minutes on 8 cores. It takes the same arguments as `examples/mnist`, with `data/cifar10` as the default directory. |

## API

### Errors

Every fallible procedure returns `ann.Error`, a `#shared_nil` union of `tensor.Tensor_Error` and `ann.Ann_Error`, so `err != nil` checks both and `or_return` passes either along. Shape, bounds and allocation failures come from the tensor library (`.Shape_Mismatch`, `.Allocation_Failed`, ...). `Ann_Error` adds `.Empty_Network`, `.Invalid_Hyperparameter`, `.Loss_Activation_Mismatch`, and `.Network_Changed`, which a trainer returns when the network's layers no longer match its optimizer state. Saving and loading add `.Invalid_File`, `.Unsupported_Version`, `.Read_Failed` and `.Write_Failed`.

### Networks

- `new_network(T, seed = 0, allocator)` creates an empty `Network(T)` for a float type `T`. `release(&network)` frees it. Treat networks as move-only.
- `set_input_shape(&network, shape)` fixes the shape of one input sample, such as `[]int{28, 28, 1}`. A network that starts with an image layer needs it; a first `dense` layer sets `[inputs]` itself. Calling it after layers are added returns `.Shape_Mismatch` unless the shape is unchanged.
- `dense(&network, inputs, outputs, activation = .Identity)` appends a fully connected layer and returns `.Shape_Mismatch` unless its input is `[inputs]`.
- `conv2d(&network, input_channels, output_channels, kernel_size, activation = .Identity, stride = 1, padding = 0)` appends a convolution with square kernels. `padding` surrounds the image with that many rows and columns of zeros, and the kernels move `stride` pixels at a time, so `[h, w, input_channels]` becomes `[(h+2·padding-kernel_size)/stride+1, (w+2·padding-kernel_size)/stride+1, output_channels]`, rounding down. A 3×3 kernel with `padding = 1` keeps the image's size, and adding `stride = 2` halves it. Weights are `[kernel_size, kernel_size, input_channels, output_channels]` and biases `[output_channels]`.
- `max_pool2d(&network, pool_size = 2)` takes the maximum of each non-overlapping `pool_size` × `pool_size` window, so `[h, w, c]` becomes `[h/pool_size, w/pool_size, c]`, rounding down. `upsample2d(&network, factor = 2)` repeats each pixel `factor` times along height and width, so `[h, w, c]` becomes `[h·factor, w·factor, c]`; followed by a padded `conv2d`, it lets a decoder grow an image back. `flatten(&network)` turns each sample into one vector in row-major order. None of these has parameters.
- `dropout(&network, rate)` zeroes each value with probability `rate` while training and scales the rest by `1/(1-rate)`. During inference it passes values through unchanged. `rate` must be in [0, 1).
- `batch_norm(&network, activation = .Identity, momentum = 0.1, epsilon = 1e-5)` normalizes the last axis: features after a dense layer, channels after a convolution. While training, it uses the batch's mean and variance over every other axis, then applies a trainable scale and shift that start at 1 and 0, held in `weights` and `bias`. Each step also moves `running_mean` and the unbiased `running_variance` `momentum` of the way toward the batch values. During inference it normalizes with those running values instead. This matches PyTorch's `BatchNorm1d` and `BatchNorm2d`. For the usual convolution → batch norm → ReLU order, give the convolution `.Identity` and the batch norm `.ReLU`. The bias of the layer just before a batch norm has no effect, because normalizing subtracts it again.
- Each `Layer(T)` holds its `kind`, `weights`, `bias`, `activation`, the settings of its kind (`pool_size`, `rate`, `momentum`, `epsilon`), a batch norm's running statistics, and the `input_shape` and `output_shape` of one sample. The parameters are ordinary tensors, so you can set or load them directly.
- `input_features_of` and `output_features_of` give the last axis of one input or output sample. `parameter_count` and `print_summary` describe a network.
- `forward(network, input, allocator)` and `forward_views` return a new tensor owned by the caller. The input's trailing axes must match the network's `input_shape`; any leading axes are batch axes, so a batch of MNIST images is `[batch, 28, 28, 1]`. Each dense layer allocates one tensor and applies its bias and activation in place.
- `activate_in_place` and `softmax_in_place` apply an activation to any tensor. Softmax normalizes over the last axis and subtracts the row maximum first.

### Losses

| `Loss` | Final activation | Targets |
|---|---|---|
| `.Mean_Squared_Error` | any | same shape as the output |
| `.Binary_Cross_Entropy` | `.Sigmoid` | values in [0, 1] |
| `.Cross_Entropy` | `.Softmax` | each row a distribution over the last axis, such as `one_hot(T, labels, classes)` |

Each loss averages over the batch. The two cross-entropy losses replace the final activation with a stable formula on its logits: `max(z, 0) - z*y + log(1 + exp(-|z|))` for binary and log-sum-exp for categorical. Any other pairing returns `.Loss_Activation_Mismatch`. `evaluate(network, loss, inputs, targets)` computes the loss without changing anything. It runs the network in inference mode, as `forward` does, so it equals the training loss exactly only for networks without dropout or batch norm. `mean_squared_error(predictions, targets)` works on any two tensors.

### Training

```odin
trainer, err := ann.trainer_init(f32, .Cross_Entropy, ann.adam(learning_rate = 0.01))
defer ann.trainer_release(&trainer)
loss, step_err := ann.train_step(&trainer, &network, inputs, targets)
```

- `sgd(learning_rate, momentum = 0, weight_decay = 0)` and `adam(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0)` build an `Optimizer`, a union of the `SGD` and `Adam` structs. Weight decay is decoupled: each step scales parameters by `1 - learning_rate*weight_decay`. Out-of-range values return `.Invalid_Hyperparameter` from `trainer_init`.
- `trainer_init(T, loss, optimizer, allocator, seed = 0)` creates a `Trainer(T)`. `seed` starts the generator that draws dropout masks. The trainer owns one autodiff tape, which it resets and rebuilds every step, and the optimizer's moment buffers, which it creates on the first step. You pass the network to each call instead of storing it in the trainer, so the network can move freely. If you add or reshape layers afterwards, `train_step` returns `.Network_Changed`.
- `train_step(&trainer, &network, inputs, targets)` runs forward in training mode, then backward and one optimizer update. Training mode means dropout masks values, and batch norm uses batch statistics and updates its running ones. Its result is the loss from before the update. `train_step_views` takes borrowed views. `trainer.steps` counts updates, and `set_learning_rate` changes the step size without resetting the moment buffers.
- Training and inference run on one thread unless you call odin-tensor's `tensor.set_thread_count(n)`. Matrix products, element-wise work, `im2col` and pooling are then split across `n` threads, forward and backward, with identical results. Dense layers with hundreds of units or more gain the most: a 1024-1024-1024 MLP trains about 4× faster on 8 cores. The MNIST CNN's step gains about 1.4×, because its batches are small and its weight-gradient products split poorly.
- `fit(&trainer, &network, inputs, targets, options)` trains over a dataset whose first axis indexes samples and returns the mean loss of the last epoch. `Fit_Options` sets `epochs`, `batch_size` (0 means the whole dataset), and `shuffle` with `seed`. It also takes `on_epoch(epoch, loss, user_data) -> bool`, which runs after each epoch; returning false stops early. It may change the values, but not the shapes, of `inputs` and `targets`, and the next epoch trains on the new values, which is how to draw fresh noise or augmentation each epoch. In-order batches are zero-copy `slice_view`s of the dataset. Shuffled batches are copied into two reused buffers.

### Metrics

- `accuracy(predictions, targets)` compares the argmax over the last axis, or thresholds at 0.5 when there is a single output.
- `predict_classes(network, inputs)` returns a `Tensor(int)` of argmax classes.
- `one_hot(T, labels, classes)` builds `[len(labels), classes]` targets.

### Saving and loading

```odin
ann.save(network, "model.bin") or_return
restored := ann.load(f32, "model.bin") or_return
defer ann.release(&restored)
```

- `save(network, path)` writes the network's input shape, each layer's kind, activation and sizes, and its parameters. `load(T, path, seed = 0, allocator)` rebuilds the network with the layer builders and fills in the parameters. `marshal` and `unmarshal` do the same with a byte slice instead of a file.
- `T` is `f32` or `f64`, and either type loads a file saved from either one. `seed` is used only for weights of layers you add after loading.
- A file that is truncated, has trailing bytes, or describes layers that do not fit together returns `.Invalid_File`. The size check happens before any parameter memory is allocated, so a corrupt file cannot trigger a huge allocation. A file from a newer format version returns `.Unsupported_Version`.
- Optimizer state is not saved. Training a loaded network with a new trainer starts Adam's moment estimates from zero.
- The format is little-endian, and its layout is documented at the top of `save.odin`.

## How training uses the tape

Each step records the network on the trainer's tape. The input enters as a constant leaf (`tape_add_leaf_view`, no gradient). Every weight and bias enters as a borrowed leaf, and each dense layer becomes `tape_matmul` → `tape_add` → activation. Matmul and add broadcast, so leading batch axes need no special handling.

A convolution is `tape_im2col`, which gathers each output position's input window into one row, with zeros for positions in the padding, then a single `tape_matmul` of those rows by the weights viewed as `[kernel_size²·input_channels, output_channels]`, and a bias `tape_add`. `forward` computes the same product the same way, so it agrees with training exactly. Max pooling is `tape_max_pool2d`, and `flatten` is a `tape_reshape`. Upsampling views its input as `[images, h, 1, w, 1, c]` and multiplies it by a constant leaf of ones shaped `[factor, 1, factor, 1]`, so broadcasting repeats each pixel exactly and the gradient sums its copies.

Dropout multiplies by a constant leaf of zeros and `1/(1-rate)`. Batch norm records odin-tensor's fused `tape_batch_norm`, which normalizes over every axis but the last and applies the scale, then a `tape_add` of the shift. Its gradient flows through the batch mean and variance as well, and the operation returns those statistics for the running estimates.

The loss itself is never recorded. Its value and its gradient with respect to the network output are computed directly with tensor kernels. That gradient seeds `tape_backward_with_seed` at the output, or at the final logits for the cross-entropy losses. The fused gradients are `sigmoid(z) - y` and `softmax(z) * sum(y) - y`, computed exactly, without differentiating through `log(sigmoid)`.

A Softmax in a hidden layer, or before mean-squared error, is recorded as `exp(x - max) / sum`. The row maximum enters as a constant leaf, which is exact because softmax ignores a per-row shift.

## Tests

```sh
odin test . -vet -strict-style -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true
```

The tests compare every loss's gradients, including a fused 3-D batch, a hidden softmax, a network using every image layer, and one with batch norm after a convolution and a dense layer plus dropout between them, against central finite differences in `f64`. The finite differences use the training loss with the same dropout masks. Batch norm's normalization, running statistics and inference output are checked against direct formulas. Convolution, pooling and upsampling are also checked against direct loops. They check that `fit` with in-order batches matches hand-sliced `train_step` calls bit for bit. Saved networks are checked for exact round trips across every layer kind and both float types, and every truncation or corrupted header field of a saved file is rejected. A failing allocator drives every `.Allocation_Failed` path through building, training, evaluating, measuring, saving and loading a network, with and without dropout and batch norm, and the define makes any leak a failure.

```sh
scripts/check_readme.sh
```

Builds every `odin` block in this README that declares `package main` against both libraries, with `-vet -strict-style`.
