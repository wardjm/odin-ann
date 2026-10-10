#!/usr/bin/env python3
"""Compare odin-ann against PyTorch for accuracy and speed.

Run from the repository root:

    python3 bench/torch_parity.py              # all cases
    python3 bench/torch_parity.py mnist xor    # cases whose name contains a filter

For each case this script draws a random network, inputs and targets, writes
them to a scratch directory, and runs bench/torch_parity (built with
-o:speed). It runs the same network in PyTorch, treated as ground truth, and
compares the forward output, the loss of every training step and the final
parameters. It then compares the median time of one forward pass and one
train_step. Both run on the CPU with one thread by default; --threads N gives
each N threads.
"""

import argparse
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


# Layers mirror odin-ann's builders: ("dense", inputs, outputs, activation),
# ("conv2d", in_channels, out_channels, kernel, activation), ("max_pool2d",
# size) and ("flatten",). Image samples are channels-last [h, w, c], as in
# odin-ann; PyTorch's half converts to and from its channels-first layout.


def dense_layers(sizes, activations):
    return [("dense", a, b, act) for a, b, act in zip(sizes, sizes[1:], activations)]


@dataclass
class Case:
    name: str
    dtype: str
    layers: list
    loss: str  # Mean_Squared_Error, Binary_Cross_Entropy, Cross_Entropy
    optimizer: tuple  # ("adam", lr, b1, b2, eps, wd) or ("sgd", lr, momentum)
    batch: int
    # One sample's shape; [features] of the first dense layer if omitted.
    input_shape: list = None
    steps: int = 20
    fwd_reps: int = 50
    train_reps: int = 20
    # Tolerance on the largest relative error, max|a - b| / max|b|.
    tolerance: float = field(default=0.0)

    def __post_init__(self):
        if self.tolerance == 0.0:
            self.tolerance = 1e-9 if self.dtype == "f64" else 1e-4
        if self.input_shape is None:
            self.input_shape = [self.layers[0][1]]

    @property
    def outputs(self):
        return next(layer[2] for layer in reversed(self.layers) if layer[0] == "dense")


ADAM = ("adam", 1e-3, 0.9, 0.999, 1e-8, 0.0)
# examples/mnist: 28x28x1 -> 24x24x8 -> 12x12x8 -> 8x8x16 -> 4x4x16 -> 256 -> 10.
MNIST_CNN = [("conv2d", 1, 8, 5, "ReLU"), ("max_pool2d", 2), ("conv2d", 8, 16, 5, "ReLU"),
             ("max_pool2d", 2), ("flatten",), ("dense", 256, 10, "Softmax")]

CASES = [
    Case("xor_f64", "f64", dense_layers([2, 4, 1], ["Tanh", "Sigmoid"]), "Binary_Cross_Entropy",
         ("adam", 0.05, 0.9, 0.999, 1e-8, 0.0), batch=4, steps=200),
    Case("regression_sgd_f64", "f64", dense_layers([8, 32, 32, 3], ["Tanh", "ReLU", "Identity"]), "Mean_Squared_Error",
         ("sgd", 0.05, 0.9), batch=64),
    Case("hidden_softmax_f64", "f64", dense_layers([10, 16, 4], ["Softmax", "Identity"]), "Mean_Squared_Error",
         ("adam", 0.01, 0.9, 0.999, 1e-8, 0.01), batch=32),
    Case("mnist_mlp_f32", "f32", dense_layers([784, 256, 128, 10], ["ReLU", "ReLU", "Softmax"]), "Cross_Entropy",
         ADAM, batch=128),
    Case("mnist_mlp_f64", "f64", dense_layers([784, 256, 128, 10], ["ReLU", "ReLU", "Softmax"]), "Cross_Entropy",
         ADAM, batch=128),
    Case("multilabel_f32", "f32", dense_layers([64, 128, 16], ["Sigmoid", "Sigmoid"]), "Binary_Cross_Entropy",
         ("adam", 1e-3, 0.9, 0.999, 1e-8, 0.01), batch=256),
    Case("wide_f32", "f32", dense_layers([1024, 1024, 1024, 10], ["ReLU", "Tanh", "Softmax"]), "Cross_Entropy",
         ("sgd", 0.01, 0.9), batch=256, steps=5, fwd_reps=10, train_reps=5),
    # Max pooling and ReLU have kinks: when a window holds a near tie, float32
    # rounding can pick a different element than float64, which reroutes
    # that gradient. Over 20 steps this happens to PyTorch's own float32 run
    # on about half of all data seeds, so this case stops after 3 steps; the
    # f64 cases check full runs.
    Case("mnist_cnn_f32", "f32", MNIST_CNN, "Cross_Entropy", ("adam", 2e-3, 0.9, 0.999, 1e-8, 0.0),
         batch=64, input_shape=[28, 28, 1], steps=3, fwd_reps=20),
    Case("mnist_cnn_f64", "f64", MNIST_CNN, "Cross_Entropy", ("adam", 2e-3, 0.9, 0.999, 1e-8, 0.0),
         batch=64, input_shape=[28, 28, 1], fwd_reps=20),
    # Odd sizes leave a row and column for the pool to drop, and the second
    # convolution sees several input channels with a non-square image.
    Case("odd_cnn_f64", "f64", [("conv2d", 3, 4, 3, "Tanh"), ("max_pool2d", 2), ("conv2d", 4, 5, 2, "Sigmoid"),
                                ("flatten",), ("dense", 30, 3, "Identity")], "Mean_Squared_Error",
         ("sgd", 0.05, 0.9), batch=16, input_shape=[11, 9, 3]),
]


def torch_activation(name, x, dim=-1):
    if name == "Identity":
        return x
    if name == "Softmax":
        return torch.softmax(x, dim=dim)
    return {"ReLU": torch.relu, "Sigmoid": torch.sigmoid, "Tanh": torch.tanh}[name](x)


def make_data(case, rng):
    """Draws Glorot-scaled parameters in odin-ann's layouts: dense weights
    [in, out] and conv2d weights [k, k, in_channels, out_channels]."""
    np_dtype = np.float32 if case.dtype == "f32" else np.float64
    parameters = {}
    for i, layer in enumerate(case.layers):
        if layer[0] == "dense":
            shape, fan = (layer[1], layer[2]), layer[1] + layer[2]
        elif layer[0] == "conv2d":
            shape, fan = (layer[3], layer[3], layer[1], layer[2]), layer[3] ** 2 * (layer[1] + layer[2])
        else:
            continue
        limit = np.sqrt(6.0 / fan)
        parameters[i] = (rng.uniform(-limit, limit, shape).astype(np_dtype),
                         rng.uniform(-0.1, 0.1, shape[-1]).astype(np_dtype))
    x = rng.standard_normal([case.batch] + case.input_shape).astype(np_dtype)
    outputs = case.outputs
    if case.loss == "Cross_Entropy":
        y = np.eye(outputs, dtype=np_dtype)[rng.integers(0, outputs, case.batch)]
    elif case.loss == "Binary_Cross_Entropy":
        y = rng.integers(0, 2, (case.batch, outputs)).astype(np_dtype)
    else:
        y = rng.standard_normal((case.batch, outputs)).astype(np_dtype)
    if case.name.startswith("xor"):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np_dtype)
        y = np.array([[0], [1], [1], [0]], dtype=np_dtype)
    return parameters, x, y


def write_case(case, directory, parameters, x, y, threads):
    lines = [f"dtype {case.dtype}", f"threads {threads}"]
    if case.layers[0][0] != "dense":
        lines.append("input " + " ".join(map(str, case.input_shape)))
    lines += [" ".join(map(str, layer)) for layer in case.layers]
    lines.append(f"loss {case.loss}")
    kind, *params = case.optimizer
    if kind == "sgd":
        params = params + [0.0]  # no weight decay: torch's SGD decay is coupled L2, odin-ann's is decoupled
    lines.append(" ".join([kind] + [repr(float(p)) for p in params]))
    lines += [f"batch {case.batch}", f"steps {case.steps}",
              f"fwd_reps {case.fwd_reps}", f"train_reps {case.train_reps}"]
    (directory / "spec.txt").write_text("\n".join(lines) + "\n")
    for i, (w, b) in parameters.items():
        w.tofile(directory / f"w{i}.bin")
        b.tofile(directory / f"b{i}.bin")
    x.tofile(directory / "x.bin")
    y.tofile(directory / "y.bin")


def torch_forward(case, params, x, logits):
    """Runs the network on odin-ann-layout parameters (a dict from layer index
    to (weights, bias)); with logits set, skips the final activation."""
    h = x.permute(0, 3, 1, 2) if len(case.input_shape) == 3 else x  # to NCHW
    for i, layer in enumerate(case.layers):
        kind, last = layer[0], i == len(case.layers) - 1
        if kind == "dense":
            w, b = params[i]
            h = h @ w + b  # weights are [in, out], as in odin-ann
        elif kind == "conv2d":
            w, b = params[i]
            h = F.conv2d(h, w.permute(3, 2, 0, 1), b)
        elif kind == "max_pool2d":
            h = F.max_pool2d(h, layer[1])
        else:
            h = h.permute(0, 2, 3, 1).flatten(1)  # odin-ann flattens channels last
        if kind in ("dense", "conv2d") and not (logits and last):
            h = torch_activation(layer[-1], h, dim=1 if kind == "conv2d" else -1)
    return h


def torch_loss(case, params, x, y):
    if case.loss == "Mean_Squared_Error":
        return F.mse_loss(torch_forward(case, params, x, False), y)
    z = torch_forward(case, params, x, True)
    if case.loss == "Binary_Cross_Entropy":
        return F.binary_cross_entropy_with_logits(z, y)
    return F.cross_entropy(z, y)  # y holds probabilities; averages over rows


def make_optimizer(case, params):
    kind, lr, *rest = case.optimizer
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=rest[0])
    b1, b2, eps, wd = rest
    return torch.optim.AdamW(params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)


def median_ms(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(samples))


def run_torch(case, parameters, x_np, y_np):
    params = {i: (torch.tensor(w, requires_grad=True), torch.tensor(b, requires_grad=True))
              for i, (w, b) in parameters.items()}
    flat = [p for pair in params.values() for p in pair]
    x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

    with torch.no_grad():
        forward = torch_forward(case, params, x, False).numpy().copy()
    optimizer = make_optimizer(case, flat)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = torch_loss(case, params, x, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    losses = np.array([step() for _ in range(case.steps)])
    final = [p.detach().numpy().copy() for p in flat]

    def forward_only():
        with torch.no_grad():
            torch_forward(case, params, x, False)

    timing = {"forward_ms": median_ms(forward_only, case.fwd_reps),
              "train_step_ms": median_ms(step, case.train_reps)}
    return forward, losses, final, timing


def run_odin(case, binary, directory, parameters):
    np_dtype = np.float32 if case.dtype == "f32" else np.float64
    result = subprocess.run([str(binary), str(directory)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"odin harness failed:\n{result.stdout}{result.stderr}")
    forward = np.fromfile(directory / "out_forward.bin", dtype=np_dtype)
    losses = np.fromfile(directory / "out_losses.bin", dtype=np.float64)
    final = []
    for i in parameters:
        final.append(np.fromfile(directory / f"out_w{i}.bin", dtype=np_dtype))
        final.append(np.fromfile(directory / f"out_b{i}.bin", dtype=np_dtype))
    timing = {}
    for line in (directory / "timing.txt").read_text().splitlines():
        key, value = line.split()
        timing[key] = float(value)
    return forward, losses, final, timing


def relative_error(odin, reference):
    odin = np.asarray(odin, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    scale = max(np.abs(reference).max(), 1e-30)
    return float(np.abs(odin - reference).max() / scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filters", nargs="*", help="run only cases whose name contains one of these")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for both odin-tensor and PyTorch (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cases = [c for c in CASES if not args.filters or any(f in c.name for f in args.filters)]
    if not cases:
        sys.exit(f"no case matches {args.filters}; cases: {', '.join(c.name for c in CASES)}")

    with tempfile.TemporaryDirectory(prefix="torch_parity_") as scratch:
        scratch = Path(scratch)
        binary = scratch / "torch_parity"
        subprocess.run(["odin", "build", str(ROOT / "bench" / "torch_parity"), "-o:speed",
                        f"-out:{binary}"], check=True, cwd=ROOT)

        print(f"torch {torch.__version__}, {args.threads} CPU thread(s) each; errors are max|odin - torch| / max|torch|\n")
        header = f"{'case':<20} {'forward':>9} {'losses':>9} {'params':>9} {'':>4}  " \
                 f"{'fwd odin':>9} {'torch':>9} {'step odin':>10} {'torch':>9}"
        print(header)
        print("-" * len(header))
        failures = 0
        rng = np.random.default_rng(args.seed)
        for case in cases:
            directory = scratch / case.name
            directory.mkdir()
            parameters, x, y = make_data(case, rng)
            write_case(case, directory, parameters, x, y, args.threads)

            o_forward, o_losses, o_final, o_time = run_odin(case, binary, directory, parameters)
            t_forward, t_losses, t_final, t_time = run_torch(case, parameters, x, y)

            errors = [relative_error(o_forward, t_forward), relative_error(o_losses, t_losses),
                      max(relative_error(o, t) for o, t in zip(o_final, t_final))]
            ok = max(errors) <= case.tolerance
            failures += not ok
            print(f"{case.name:<20} {errors[0]:>9.1e} {errors[1]:>9.1e} {errors[2]:>9.1e} {'ok' if ok else 'FAIL':>4}  "
                  f"{o_time['forward_ms']:>7.3f}ms {t_time['forward_ms']:>7.3f}ms "
                  f"{o_time['train_step_ms']:>8.3f}ms {t_time['train_step_ms']:>7.3f}ms")
        print(f"\ntolerance: 1e-9 for f64, 1e-4 for f32. {failures} case(s) failed.")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
