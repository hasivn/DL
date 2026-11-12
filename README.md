# Deep Double Descent - Reduced-Scale Reproduction

**Roll No:** 24CS3032 | **Paper ID:** DL-015
**Paper:** Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever. *Deep Double Descent: Where Bigger Models and More Data Hurt.* ICLR 2020. https://arxiv.org/abs/1912.02292

## What this repo does
Reproduces, at reduced scale, model-wise and epoch-wise double descent on CIFAR-10 with label noise, using a ResNet-18 of variable width `k` (channels k, 2k, 4k, 8k), written by hand in PyTorch (`double_descent.py`). Includes a noise-level ablation (0%, 10%, 20%).

## Files
| File | Purpose |
|---|---|
| `double_descent.py` | Full implementation (model, label noise, training, plots) |
| `NOTEBOOK.ipynb` | Colab notebook used to run the experiments |
| `model_wise.png`, `model_wise_noise0.15.json` | Model-wise sweep, 15% noise |
| `epoch_wise.png`, `epoch_wise.json` | Epoch-wise run, k=64, 200 epochs |
| `ablation_noise.png`, `ablation_noise0.{0,1,2}.json` | Noise ablation |

## Setup
```
pip install -r requirements.txt
```
A GPU is recommended (Colab free T4 was used).

## How to run
```
python double_descent.py --smoke                       # quick sanity check
python double_descent.py --mode model_wise --noise 0.15
python double_descent.py --mode epoch_wise --epochs 200
python double_descent.py --mode ablation
```
Results are written to `./results/`.

## Setup details
- Data: 10,000-image random subset of CIFAR-10 training set, full 10,000-image test set (clean labels).
- Label noise: 15% of training labels replaced with a different random class (training set only).
- Optimizer: Adam, lr 1e-4, batch size 128, no augmentation, no weight decay.
- Epochs: 120 per width (model-wise), 200 (epoch-wise).
- Widths: k in {1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48}.
- **Seed: 0** (single seed).

## Results (single seed)
| Setting | Result |
|---|---|
| Noise 15%, train error reaches ~0 | k = 4-6 |
| Noise 15%, test error k=2 -> k=3 | 70.67% -> 72.22% (small bump) |
| Noise 15%, test error at k=48 | 55.32% |
| Test error at k=48, noise 0% / 10% / 20% | 45.74% / 55.57% / 59.23% |
| Epoch-wise (k=64) test error, epoch 1 / 4 / 200 | 57.0% / 62.2% / 52.89% |

Paper's numbers: see Fig. 1 of the paper. The paper uses the full 50k training set and far longer training, so absolute errors there are much lower than here.

## Differences from the paper
- 10k training subset (paper: 50k).
- 120 / 200 epochs (paper: thousands).
- Widths up to 48 (paper: up to 64).
- One seed only.
- The double-descent peak is weak in our runs; the clearest effects are the second descent and the effect of label noise. Periodic train-error spikes in the epoch-wise run are Adam instabilities, not double descent.

## References / code consulted
- Paper above and the accompanying OpenAI blog post "Deep Double Descent".
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 (ResNet architecture).
