"""
Deep Double Descent (Nakkiran et al., ICLR 2020) - reduced-scale reproduction
Roll No: 24CS3032 | Paper ID: DL-015

Modes
  model_wise  : sweep ResNet-18 width k, train to (near) interpolation, plot test error vs k
  epoch_wise  : train one large model for many epochs, plot test error vs epoch
  ablation    : repeat model-wise sweep for several label-noise levels (0, 0.1, 0.2)

"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 1. Model: ResNet-18 with variable width k (channels = k, 2k, 4k, 8k)
#    Written by hand, as in the paper (k=64 is the standard ResNet-18).
# ----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18k(nn.Module):
    def __init__(self, k=64, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, k, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(k)
        cfg = [(k, 1), (2 * k, 2), (4 * k, 2), (8 * k, 2)]
        layers, in_c = [], k
        for out_c, stride in cfg:
            layers += [BasicBlock(in_c, out_c, stride), BasicBlock(out_c, out_c, 1)]
            in_c = out_c
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(8 * k, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ----------------------------------------------------------------------------
# 2. Data: CIFAR-10 held on the GPU, label noise applied ONLY to training set
# ----------------------------------------------------------------------------
def add_label_noise(y, noise, num_classes, rng):
    """With prob `noise`, replace label by a uniformly random *different* class."""
    y = y.copy()
    flip = rng.random(len(y)) < noise
    offsets = rng.integers(1, num_classes, size=len(y))   # 1..9 => always different
    y[flip] = (y[flip] + offsets[flip]) % num_classes
    return y, flip


def load_data(args, device):
    rng = np.random.default_rng(args.seed)
    if args.smoke:                                         # random data for testing code
        xtr = rng.normal(size=(256, 3, 32, 32)).astype("float32")
        ytr = rng.integers(0, 10, 256)
        xte = rng.normal(size=(128, 3, 32, 32)).astype("float32")
        yte = rng.integers(0, 10, 128)
    else:
        import torchvision
        tr = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True)
        te = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True)
        xtr = tr.data.transpose(0, 3, 1, 2).astype("float32") / 255.0
        xte = te.data.transpose(0, 3, 1, 2).astype("float32") / 255.0
        ytr, yte = np.array(tr.targets), np.array(te.targets)
        mean = xtr.mean((0, 2, 3), keepdims=True)
        std = xtr.std((0, 2, 3), keepdims=True)
        xtr, xte = (xtr - mean) / std, (xte - mean) / std
        idx = rng.permutation(len(xtr))[: args.n_train]    # reduced-scale subset
        xtr, ytr = xtr[idx], ytr[idx]
    return xtr, ytr, xte, yte, rng


def to_dev(x, y, device):
    return torch.tensor(x, device=device), torch.tensor(y, device=device, dtype=torch.long)


# ----------------------------------------------------------------------------
# 3. Train / eval
# ----------------------------------------------------------------------------
@torch.no_grad()
def error_rate(model, x, y, bs=1000):
    model.eval()
    wrong = 0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", enabled=x.is_cuda):
            pred = model(x[i:i + bs]).argmax(1)
        wrong += (pred != y[i:i + bs]).sum().item()
    return 100.0 * wrong / len(x)


def train_model(k, xtr, ytr_noisy, xte, yte, args, device, log_every_epoch=False):
    torch.manual_seed(args.seed)
    model = ResNet18k(k).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    n, hist = len(xtr), []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            if len(b) < 2:
                continue
            with torch.autocast("cuda", enabled=(device == "cuda")):
                loss = F.cross_entropy(model(xtr[b]), ytr_noisy[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if log_every_epoch or ep == args.epochs:
            tr_err = error_rate(model, xtr, ytr_noisy)     # error on the NOISY train labels
            te_err = error_rate(model, xte, yte)           # error on CLEAN test labels
            hist.append(dict(epoch=ep, train_err=tr_err, test_err=te_err))
            if log_every_epoch and ep % 10 == 0:
                print(f"  k={k} ep={ep} train_err={tr_err:.2f} test_err={te_err:.2f}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    return hist, n_params


# ----------------------------------------------------------------------------
# 4. Experiments
# ----------------------------------------------------------------------------
def run_model_wise(args, device, noise, tag):
    xtr, ytr, xte, yte, rng = load_data(args, device)
    ytr_n, flipped = add_label_noise(ytr, noise, 10, rng)
    print(f"[{tag}] noise={noise}  actually flipped={flipped.mean()*100:.1f}%")
    xtr_t, ytr_t = to_dev(xtr, ytr_n, device)
    xte_t, yte_t = to_dev(xte, yte, device)
    results = []
    for k in args.widths:
        t0 = time.time()
        hist, n_params = train_model(k, xtr_t, ytr_t, xte_t, yte_t, args, device)
        r = dict(k=k, params=n_params, **hist[-1])
        results.append(r)
        print(f"[{tag}] k={k:3d} params={n_params/1e6:6.2f}M train_err={r['train_err']:6.2f} "
              f"test_err={r['test_err']:6.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(os.path.join(args.out_dir, f"{tag}.json"), "w"), indent=1)
    return results


def run_epoch_wise(args, device):
    xtr, ytr, xte, yte, rng = load_data(args, device)
    ytr_n, _ = add_label_noise(ytr, args.noise, 10, rng)
    xtr_t, ytr_t = to_dev(xtr, ytr_n, device)
    xte_t, yte_t = to_dev(xte, yte, device)
    hist, _ = train_model(args.epoch_k, xtr_t, ytr_t, xte_t, yte_t, args, device, log_every_epoch=True)
    json.dump(hist, open(os.path.join(args.out_dir, "epoch_wise.json"), "w"), indent=1)
    return hist


# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
def plot_model_wise(results_by_tag, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for tag, res in results_by_tag.items():
        ks = [r["k"] for r in res]
        plt.plot(ks, [r["test_err"] for r in res], "o-", label=f"test ({tag})")
        if len(results_by_tag) == 1:
            plt.plot(ks, [r["train_err"] for r in res], "s--", label="train (noisy labels)")
    plt.xlabel("ResNet-18 width parameter k"); plt.ylabel("Error (%)")
    plt.title("Model-wise double descent"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_epoch_wise(hist, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["test_err"] for h in hist], label="test error")
    plt.plot(ep, [h["train_err"] for h in hist], "--", label="train error (noisy)")
    plt.xlabel("Epoch"); plt.ylabel("Error (%)"); plt.title("Epoch-wise double descent")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="model_wise", choices=["model_wise", "epoch_wise", "ablation"])
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--epoch_k", type=int, default=64, help="width for epoch-wise run")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="./results")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.widths, args.epochs, args.epoch_k, args.bs = [1, 2], 2, 2, 64
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| args:", vars(args))

    if args.mode == "model_wise":
        res = run_model_wise(args, device, args.noise, f"model_wise_noise{args.noise}")
        plot_model_wise({f"noise {args.noise}": res}, os.path.join(args.out_dir, "model_wise.png"))
    elif args.mode == "epoch_wise":
        hist = run_epoch_wise(args, device)
        plot_epoch_wise(hist, os.path.join(args.out_dir, "epoch_wise.png"))
    else:
        allres = {}
        for nz in [0.0, 0.1, 0.2]:
            allres[f"noise {nz}"] = run_model_wise(args, device, nz, f"ablation_noise{nz}")
        plot_model_wise(allres, os.path.join(args.out_dir, "ablation_noise.png"))
    print("done. results in", args.out_dir)
