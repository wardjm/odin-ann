# Development

## Layout

| File | Contents |
|---|---|
| `ann.odin` | Errors, `Layer`, `Network`, `dense`, and shape bookkeeping |
| `conv.odin` | `conv2d`, `max_pool2d`, `upsample2d`, `flatten` |
| `regularization.odin` | `dropout` and `batch_norm` |
| `forward.odin` | Inference and activations |
| `loss.odin` | Losses and their gradients |
| `train.odin` | Optimizers, `Trainer`, `train_step`, `fit` |
| `metrics.odin` | `accuracy`, `predict_classes`, `one_hot` |
| `save.odin` | The file format, `save`/`load`, `marshal`/`unmarshal` |
| `bench/` | The PyTorch comparison |
| `examples/` | Runnable examples, see the README |

## Tests

```sh
odin test . -vet -strict-style -define:ODIN_TEST_FAIL_ON_BAD_MEMORY=true
scripts/check_readme.sh
```

The first command runs the unit tests and turns any leak into a failure. The second builds every complete program in the README (each ```` ```odin ```` block that declares `package main`) with `-vet -strict-style`. CI runs both and checks that every example and the benchmark compile.

What the tests cover:

- **Gradients.** Every loss's gradients are compared against central finite differences in `f64`. The cases include a fused 3-D batch, a hidden softmax, a network that uses every image layer, and one with batch norm and dropout. The finite differences reuse the same dropout masks.
- **Layers.** Convolution, pooling and upsampling are compared against direct loops. Batch norm's normalization, running statistics and inference output are checked against the formulas.
- **Training.** `fit` with in-order batches must match hand-sliced `train_step` calls bit for bit.
- **Saving.** Every layer kind must round-trip exactly in both float types. Every truncation and every corrupted header field must be rejected.
- **Allocation failures.** A failing allocator drives every `.Allocation_Failed` path through building, training, evaluating, measuring, saving and loading, with and without dropout and batch norm.

## How training uses the tape

`forward` computes each layer directly with tensor kernels. `train_step` records the same computation on the trainer's odin-tensor autodiff tape. It resets and rebuilds the tape every step, so the network can move between calls. If the parameter shapes no longer match the optimizer state, the trainer returns `.Network_Changed`.

On the tape:

- The input is a constant leaf with no gradient. Every weight and bias is a leaf that borrows the network's tensor.
- **Dense**: one `tape_linear` (product plus bias), applied to the last axis, so leading batch axes need no special handling.
- **Conv2D**: `tape_im2col` gathers each output position's window into the last axis, `[..., out_h, out_w, k²·in_channels]`. Positions in the padding read as zeros. One `tape_linear` then applies the weights, viewed as `[k²·in_channels, out_channels]`. `forward` computes the same product the same way, so the two agree exactly.
- **Max_Pool2D**: `tape_max_pool2d`. The gradient reaches one maximal element per window.
- **Flatten**: `tape_reshape`.
- **Upsample2D**: views the input as `[images, h, 1, w, 1, c]` and multiplies it by a constant `[factor, 1, factor, 1]` of ones. Broadcasting repeats each pixel, and the gradient sums its copies.
- **Dropout**: multiplies by a constant mask of zeros and `1/(1-rate)`, drawn from the trainer's generator.
- **Batch_Norm**: odin-tensor's fused `tape_batch_norm`. It normalizes over every axis but the last, applies the scale and shift, and fuses a following ReLU. The gradient flows through the batch mean and variance. The batch statistics it returns update the running estimates.
- **Softmax** in a hidden layer, or before mean-squared error, is recorded as `exp(x - max) / sum`. The row maximum is a constant leaf, which is exact because softmax ignores a per-row shift.

The loss itself isn't recorded. Its value and its gradient with respect to the output are computed directly. That gradient seeds `tape_backward_with_seed` at the network output, or at the final logits for the cross-entropy losses. The fused gradients are `sigmoid(z) - y` and `softmax(z)·sum(y) - y`, so the backward pass never differentiates through `log(sigmoid)`. Binary cross-entropy's value is `max(z, 0) - z·y + log(1 + exp(-|z|))`, and categorical cross-entropy uses log-sum-exp.

In `fit`, in-order batches are `slice_view`s of the dataset. `fit` gathers shuffled batches into two buffers it reuses across steps.

## Comparing with PyTorch

```sh
python3 bench/torch_parity.py                   # every case, one thread
python3 bench/torch_parity.py cifar --threads 8 # cases matching "cifar"
```

This needs Python with PyTorch. Each case, from XOR up to the CIFAR-10 network, draws a random network, inputs and targets. It trains for a few steps in both libraries from the same weights and reports the largest relative difference in the forward output, each step's loss and the final parameters.

The tolerances are 1e-9 for `f64` and 1e-4 for `f32`. The CIFAR-10 network in `f32` gets a looser tolerance because its summation order matters more, as the script explains. The `f64` cases agree to 1e-13 or better.

The script then compares the median times of a forward pass and of one `train_step`. The README's 8-thread columns ran under `taskset -c 0-7`.
