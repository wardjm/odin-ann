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
train_step. PyTorch runs on the CPU with one thread, because odin-tensor is
single-threaded; pass --threads N to change that.
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


@dataclass
class Case:
    name: str
    dtype: str
    sizes: list  # [features, hidden..., outputs]
    activations: list  # one per layer
    loss: str  # Mean_Squared_Error, Binary_Cross_Entropy, Cross_Entropy
    optimizer: tuple  # ("adam", lr, b1, b2, eps, wd) or ("sgd", lr, momentum)
    batch: int
    steps: int = 20
    fwd_reps: int = 50
    train_reps: int = 20
    # Tolerance on the largest relative error, max|a - b| / max|b|.
    tolerance: float = field(default=0.0)

    def __post_init__(self):
        if self.tolerance == 0.0:
            self.tolerance = 1e-9 if self.dtype == "f64" else 1e-4


CASES = [
    Case("xor_f64", "f64", [2, 4, 1], ["Tanh", "Sigmoid"], "Binary_Cross_Entropy",
         ("adam", 0.05, 0.9, 0.999, 1e-8, 0.0), batch=4, steps=200),
    Case("regression_sgd_f64", "f64", [8, 32, 32, 3], ["Tanh", "ReLU", "Identity"], "Mean_Squared_Error",
         ("sgd", 0.05, 0.9), batch=64),
    Case("hidden_softmax_f64", "f64", [10, 16, 4], ["Softmax", "Identity"], "Mean_Squared_Error",
         ("adam", 0.01, 0.9, 0.999, 1e-8, 0.01), batch=32),
    Case("mnist_mlp_f32", "f32", [784, 256, 128, 10], ["ReLU", "ReLU", "Softmax"], "Cross_Entropy",
         ("adam", 1e-3, 0.9, 0.999, 1e-8, 0.0), batch=128),
    Case("mnist_mlp_f64", "f64", [784, 256, 128, 10], ["ReLU", "ReLU", "Softmax"], "Cross_Entropy",
         ("adam", 1e-3, 0.9, 0.999, 1e-8, 0.0), batch=128),
    Case("multilabel_f32", "f32", [64, 128, 16], ["Sigmoid", "Sigmoid"], "Binary_Cross_Entropy",
         ("adam", 1e-3, 0.9, 0.999, 1e-8, 0.01), batch=256),
    Case("wide_f32", "f32", [1024, 1024, 1024, 10], ["ReLU", "Tanh", "Softmax"], "Cross_Entropy",
         ("sgd", 0.01, 0.9), batch=256, steps=5, fwd_reps=10, train_reps=5),
]

TORCH_ACTIVATIONS = {
    "Identity": lambda x: x,
    "ReLU": torch.relu,
    "Sigmoid": torch.sigmoid,
    "Tanh": torch.tanh,
    "Softmax": lambda x: torch.softmax(x, dim=-1),
}


def make_data(case, rng):
    np_dtype = np.float32 if case.dtype == "f32" else np.float64
    weights, biases = [], []
    for fan_in, fan_out in zip(case.sizes, case.sizes[1:]):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np_dtype))
        biases.append(rng.uniform(-0.1, 0.1, fan_out).astype(np_dtype))
    x = rng.standard_normal((case.batch, case.sizes[0])).astype(np_dtype)
    outputs = case.sizes[-1]
    if case.loss == "Cross_Entropy":
        y = np.eye(outputs, dtype=np_dtype)[rng.integers(0, outputs, case.batch)]
    elif case.loss == "Binary_Cross_Entropy":
        y = rng.integers(0, 2, (case.batch, outputs)).astype(np_dtype)
    else:
        y = rng.standard_normal((case.batch, outputs)).astype(np_dtype)
    if case.name.startswith("xor"):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np_dtype)
        y = np.array([[0], [1], [1], [0]], dtype=np_dtype)
    return weights, biases, x, y


def write_case(case, directory, weights, biases, x, y):
    lines = [f"dtype {case.dtype}"]
    for i, activation in enumerate(case.activations):
        lines.append(f"layer {case.sizes[i]} {case.sizes[i + 1]} {activation}")
    lines.append(f"loss {case.loss}")
    kind, *params = case.optimizer
    if kind == "sgd":
        params = params + [0.0]  # no weight decay: torch's SGD decay is coupled L2, odin-ann's is decoupled
    lines.append(" ".join([kind] + [repr(float(p)) for p in params]))
    lines += [f"batch {case.batch}", f"steps {case.steps}",
              f"fwd_reps {case.fwd_reps}", f"train_reps {case.train_reps}"]
    (directory / "spec.txt").write_text("\n".join(lines) + "\n")
    for i, (w, b) in enumerate(zip(weights, biases)):
        w.tofile(directory / f"w{i}.bin")
        b.tofile(directory / f"b{i}.bin")
    x.tofile(directory / "x.bin")
    y.tofile(directory / "y.bin")


def torch_forward(params, activations, x, logits):
    """Runs the network; with logits set, skips the final activation."""
    h = x
    for i, activation in enumerate(activations):
        w, b = params[2 * i], params[2 * i + 1]
        h = h @ w + b  # weights are [in, out], as in odin-ann
        if not (logits and i == len(activations) - 1):
            h = TORCH_ACTIVATIONS[activation](h)
    return h


def torch_loss(case, params, x, y):
    if case.loss == "Mean_Squared_Error":
        return F.mse_loss(torch_forward(params, case.activations, x, False), y)
    z = torch_forward(params, case.activations, x, True)
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


def run_torch(case, weights, biases, x_np, y_np):
    params = []
    for w, b in zip(weights, biases):
        params += [torch.tensor(w, requires_grad=True), torch.tensor(b, requires_grad=True)]
    x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

    with torch.no_grad():
        forward = torch_forward(params, case.activations, x, False).numpy().copy()
    optimizer = make_optimizer(case, params)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = torch_loss(case, params, x, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    losses = np.array([step() for _ in range(case.steps)])
    final = [p.detach().numpy().copy() for p in params]

    def forward_only():
        with torch.no_grad():
            torch_forward(params, case.activations, x, False)

    timing = {"forward_ms": median_ms(forward_only, case.fwd_reps),
              "train_step_ms": median_ms(step, case.train_reps)}
    return forward, losses, final, timing


def run_odin(case, binary, directory):
    np_dtype = np.float32 if case.dtype == "f32" else np.float64
    result = subprocess.run([str(binary), str(directory)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"odin harness failed:\n{result.stdout}{result.stderr}")
    forward = np.fromfile(directory / "out_forward.bin", dtype=np_dtype)
    losses = np.fromfile(directory / "out_losses.bin", dtype=np.float64)
    final = []
    for i in range(len(case.activations)):
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
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default 1)")
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

        print(f"torch {torch.__version__}, {args.threads} CPU thread(s); errors are max|odin - torch| / max|torch|\n")
        header = f"{'case':<20} {'forward':>9} {'losses':>9} {'params':>9} {'':>4}  " \
                 f"{'fwd odin':>9} {'torch':>9} {'step odin':>10} {'torch':>9}"
        print(header)
        print("-" * len(header))
        failures = 0
        rng = np.random.default_rng(args.seed)
        for case in cases:
            directory = scratch / case.name
            directory.mkdir()
            weights, biases, x, y = make_data(case, rng)
            write_case(case, directory, weights, biases, x, y)

            o_forward, o_losses, o_final, o_time = run_odin(case, binary, directory)
            t_forward, t_losses, t_final, t_time = run_torch(case, weights, biases, x, y)

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
