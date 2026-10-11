# odin-ann

A small neural-network library for [Odin](https://odin-lang.org), built on [`odin-tensor`](https://github.com/wardjm/odin-tensor) and its reverse-mode autodiff.

- Sequential networks of dense, convolution, pooling, upsampling, flatten, dropout and batch-norm layers, in `f32` or `f64`.
- Identity, ReLU, Sigmoid, Tanh and Softmax activations.
- Mean-squared error, binary cross-entropy and categorical cross-entropy, with the cross-entropies computed stably from logits.
- SGD with momentum, and Adam/AdamW.
- A minibatch `fit` loop with shuffling, a per-epoch callback and early stopping, or `train_step` for your own loop.
- Saving and loading networks in a compact binary file.
- Optional multithreading, with identical results for any thread count.

## Installation

odin-ann imports odin-tensor as `../odin-tensor`, so clone the two repositories next to each other under those names:

```sh
mkdir libs && cd libs
git clone https://github.com/wardjm/odin-ann.git
git clone https://github.com/wardjm/odin-tensor.git
```

Your program needs odin-tensor too, because the API takes and returns `tensor.Tensor` values. Import it from that same checkout. A second copy elsewhere fails with `Duplicate declaration of 'package tensor'`. A collection pointing at `libs` keeps both imports consistent:

```sh
odin run my_app -collection:libs=path/to/libs
```

```odin
import ann "libs:odin-ann"
import tensor "libs:odin-tensor"
```

## Quick start

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

A convolutional classifier for 28×28 grayscale images:

```odin
build :: proc(network: ^ann.Network(f32)) -> ann.Error {
    ann.set_input_shape(network, []int{28, 28, 1}) or_return
    ann.conv2d(network, 1, 16, 3, padding = 1) or_return // 28x28x16
    ann.batch_norm(network, .ReLU) or_return
    ann.max_pool2d(network) or_return                    // 14x14x16
    ann.flatten(network) or_return                       // 3136
    ann.dropout(network, 0.3) or_return
    return ann.dense(network, 14*14*16, 10, .Softmax)
}
```

## Examples

Run these from the repository root. Each file opens with a comment describing what it shows and the arguments it takes.

| Command | Shows |
|---|---|
| `odin run examples/linear_regression` | Plain SGD with a hand-written `train_step` loop. |
| `odin run examples/xor` | Binary cross-entropy, Adam, and early stopping from `on_epoch`. |
| `odin run examples/sine -o:speed` | Regression with shuffled minibatches and a learning-rate schedule. |
| `odin run examples/spirals -o:speed` | Softmax classification with `one_hot`, `accuracy` and `predict_classes`. |
| `odin run examples/mnist -o:speed` | A small CNN on MNIST, about 98.6% test accuracy in 3 epochs, plus `save`/`load`. |
| `odin run examples/mnist_deep -o:speed` | A deeper CNN with batch norm, dropout and learning-rate decay, about 99.2%. |
| `odin run examples/mnist_denoise -o:speed` | A convolutional denoising autoencoder using strided convolutions and `upsample2d`. |
| `odin run examples/cifar10 -o:speed` | A CNN on CIFAR-10 with per-epoch data augmentation, about 83% in 15 epochs. |

The MNIST examples need `scripts/fetch_mnist.sh` (about 11 MB) and the CIFAR-10 example needs `scripts/fetch_cifar10.sh` (about 162 MB). Both scripts download into `data/`.

## API

### Shapes

Image layers use channels-last samples, `[height, width, channels]`. Every procedure that takes data accepts any number of leading batch axes. A single sample, `[batch, ...]` and `[a, b, ...]` all work, and the output keeps the same leading axes. `fit` treats the first axis as the sample index.

### Errors

Every fallible procedure returns `ann.Error`, a `#shared_nil` union of `tensor.Tensor_Error` and `ann.Ann_Error`. `err != nil` checks both, and `or_return` passes either along.

- Shape, bounds and allocation failures come from odin-tensor: `.Invalid_Shape`, `.Shape_Mismatch`, `.Out_Of_Bounds`, `.Allocation_Failed`, and so on.
- `Ann_Error` adds:
  - `.Empty_Network`: the network has no layers.
  - `.Invalid_Hyperparameter`: a rate, momentum, epoch count or similar is out of range.
  - `.Loss_Activation_Mismatch`: the loss doesn't match the final activation (see [Losses](#losses)).
  - `.Network_Changed`: layers were added or reshaped after the trainer's first step.
  - `.Invalid_File`, `.Unsupported_Version`, `.Read_Failed`, `.Write_Failed`: errors from saving and loading.

### Building a network

```odin
network := ann.new_network(f32, seed = 0, allocator = context.allocator)
defer ann.release(&network)
```

`Network(T)` owns its parameters. Treat it as move-only and release it once. `seed` makes the initial weights reproducible. The builders draw weights uniformly, with He scaling before ReLU and Glorot scaling otherwise, and start biases at zero.

| Builder | Effect on one sample |
|---|---|
| `set_input_shape(&network, shape)` | Sets the input sample shape, such as `[]int{28, 28, 1}`. A network that starts with an image layer needs it, and a first `dense` sets it automatically. |
| `dense(&network, inputs, outputs, activation = .Identity)` | `[inputs]` → `[outputs]` |
| `conv2d(&network, in_channels, out_channels, kernel_size, activation = .Identity, stride = 1, padding = 0)` | `[h, w, in_channels]` → `[(h+2p-k)/s+1, (w+2p-k)/s+1, out_channels]` |
| `max_pool2d(&network, pool_size = 2)` | `[h, w, c]` → `[h/pool_size, w/pool_size, c]` |
| `upsample2d(&network, factor = 2)` | `[h, w, c]` → `[h·factor, w·factor, c]` (nearest neighbour) |
| `flatten(&network)` | Any shape → one vector, row-major |
| `dropout(&network, rate)` | Unchanged. While training, zeroes values with probability `rate` and scales the rest by `1/(1-rate)`. |
| `batch_norm(&network, activation = .Identity, momentum = 0.1, epsilon = 1e-5)` | Unchanged. Normalizes the last axis (features or channels). |

Each builder checks its input against the previous layer's output and returns `.Shape_Mismatch` if they don't fit. Sizes in the table round down: windows that pass the padded image's edge are dropped, and so are rows or columns that don't fill a pool.

Notes:

- With `conv2d`, a 3×3 kernel and `padding = 1` keep the image size, and `stride = 2` halves it. Weights are `[k, k, in_channels, out_channels]` and biases `[out_channels]`.
- `batch_norm` follows PyTorch's `BatchNorm1d`/`BatchNorm2d`. While training, it uses the batch's statistics and moves `running_mean` and the unbiased `running_variance` toward them by `momentum`. During inference it uses the running values. Its scale and shift are stored in `weights` and `bias`. For conv → batch norm → ReLU, give the convolution `.Identity` and the batch norm `.ReLU`.
- `upsample2d` followed by a padded `conv2d` lets a decoder grow an image back.

Each `Layer(T)` in `network.layers` exposes its `kind`, `activation`, `weights` and `bias`, the settings for its kind, and `input_shape` and `output_shape`. Parameters are ordinary tensors that you can read or overwrite.

Other helpers:

- `parameter_count(network)` gives the number of trainable values, and `print_summary(network)` prints one line per layer.
- `input_features_of(network)` and `output_features_of(network)` give the last axis of one input or output sample.

### Running a network

```odin
output, err := ann.forward(network, input) // caller releases output
```

- `forward(network, input, allocator)` runs inference. Dropout passes values through, and batch norm uses its running statistics. The input's trailing axes must equal the network's input shape. `forward_views` takes a `tensor.Tensor_View`.
- `predict_classes(network, inputs)` returns a `Tensor(int)` holding the argmax of each output.
- `activate_in_place(&t, activation)` and `softmax_in_place(&t)` apply an activation to any tensor. Softmax works over the last axis.

### Losses

| `Loss` | Final activation | Targets |
|---|---|---|
| `.Mean_Squared_Error` | any | same shape as the output |
| `.Binary_Cross_Entropy` | `.Sigmoid` | values in [0, 1] |
| `.Cross_Entropy` | `.Softmax` | rows that are distributions over the last axis, such as `one_hot(T, labels, classes)` |

Every loss averages over the batch. The cross-entropy losses work from the final layer's logits, so they stay finite when outputs saturate. Any other pairing returns `.Loss_Activation_Mismatch`.

- `evaluate(network, loss, inputs, targets)` computes the loss in inference mode without changing anything. `evaluate_views` takes views.
- `mean_squared_error(predictions, targets)` works on any two tensors of the same shape.

### Training

```odin
trainer, err := ann.trainer_init(f32, .Cross_Entropy, ann.adam(learning_rate = 0.01))
defer ann.trainer_release(&trainer)

loss, step_err := ann.train_step(&trainer, &network, inputs, targets)
// or
loss, fit_err := ann.fit(&trainer, &network, inputs, targets, {epochs = 10, batch_size = 64, shuffle = true})
```

Optimizers:

- `sgd(learning_rate, momentum = 0, weight_decay = 0)`
- `adam(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0)` uses bias correction. A nonzero `weight_decay` makes it AdamW.

Both build an `Optimizer`, a union of the `SGD` and `Adam` structs. Weight decay is decoupled, so each step scales parameters by `1 - learning_rate·weight_decay`. Out-of-range values make `trainer_init` return `.Invalid_Hyperparameter`.

Trainer:

- `trainer_init(T, loss, optimizer, allocator, seed = 0)` creates a `Trainer(T)`. `seed` drives the dropout masks. Every call takes the network as an argument, so the network can move between calls.
- `train_step(&trainer, &network, inputs, targets)` does one forward pass, one backward pass and one optimizer update, and returns the loss from before the update. `train_step_views` takes views, such as minibatch slices.
- `set_learning_rate(&trainer, rate)` changes the step size for learning-rate schedules and keeps the optimizer state. `trainer.steps` counts updates.

`fit(&trainer, &network, inputs, targets, options)` trains on a dataset whose first axis indexes samples. It returns the mean loss of the last epoch. `Fit_Options` has these fields:

| Field | Meaning |
|---|---|
| `epochs` | Number of passes over the data. Must be positive. |
| `batch_size` | Samples per step. 0 uses the whole dataset. |
| `shuffle`, `seed` | Draw a new sample order each epoch. In-order batches don't copy any data. |
| `on_epoch(epoch, loss, user_data) -> bool` | Runs after each epoch. Return `false` to stop early. It may overwrite the values, but not the shapes, of `inputs` and `targets`, for fresh augmentation each epoch. |
| `user_data` | Passed to `on_epoch`. |

Training mode differs from inference only for dropout and batch norm. For networks without those layers, `evaluate` matches the training loss exactly.

### Metrics

- `accuracy(predictions, targets)` compares the argmax over the last axis. With a single output per sample, it compares predictions and targets thresholded at 0.5.
- `one_hot(T, labels, classes)` builds a `[len(labels), classes]` target tensor.

### Saving and loading

```odin
ann.save(network, "model.bin") or_return
restored := ann.load(f32, "model.bin") or_return
defer ann.release(&restored)
```

- `save(network, path)` stores the input shape, the layers and their parameters. `load(T, path, seed = 0, allocator)` rebuilds the network without the code that built it. `seed` is used only for layers you add afterwards.
- `marshal(network)` and `unmarshal(T, data)` do the same with a byte slice.
- An `f32` file loads as `f64` and vice versa.
- Truncated or inconsistent data returns `.Invalid_File` before allocating anything large. A file from a newer format version returns `.Unsupported_Version`.
- Optimizer state isn't saved, so Adam restarts from zero moments when you train a loaded network. The file layout is documented at the top of `save.odin`.

### Threads

Everything runs on one thread unless you call odin-tensor's `tensor.set_thread_count(n)`. After that, forward and backward passes use `n` threads, and the results are identical for any `n`. On 8 cores, the MNIST CNN trains about 3.6× faster than on one, and the CIFAR-10 CNN about 4.4× faster.

## Performance

Measured training step times against PyTorch 2.11 on a Ryzen 7 2700X, with the same networks and weights:

| Case | odin-ann, 1 thread | PyTorch, 1 thread | odin-ann, 8 threads | PyTorch, 8 threads |
|---|---|---|---|---|
| CIFAR-10 CNN, `f32` | 119 ms | 91 ms | 27.0 ms | 25.5 ms |
| CIFAR-10 CNN, `f64` | 231 ms | 255 ms | 63 ms | 82 ms |
| MNIST CNN, `f32` | 9.4 ms | 5.7 ms | 2.6 ms | 2.4 ms |
| 1024-1024-1024 MLP, `f32` | 55 ms | 54 ms | 11.6 ms | 12.3 ms |

The same harness checks that outputs, losses and trained parameters agree with PyTorch. See [DEVELOPMENT.md](DEVELOPMENT.md).

## Development

[DEVELOPMENT.md](DEVELOPMENT.md) covers running the tests, how training records networks on the autodiff tape, and the PyTorch comparison.

## License

MIT. See [LICENSE](LICENSE).
