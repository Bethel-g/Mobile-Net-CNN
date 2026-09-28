# From LeNet-5 to MobileNetV1 — Two CNN Architectures, Measured

A from-scratch Keras implementation of **two** convolutional neural networks that bracket
nineteen years of CNN design, trained and evaluated in the same codebase with the same recipe:

- **MobileNetV1** (Howard et al., 2017) — the influential architecture. Depthwise separable
  convolution, plus an ablation that isolates its effect.
- **LeNet-5** (LeCun et al., 1998) — the architecture every CNN still follows, run on MNIST
  (its own dataset) and on CIFAR-10 (so the two models are directly comparable).

**No pretrained weights. No `keras.applications`.** MobileNetV1 is generated from a 9-entry
config table with depthwise separable convolution built from its two constituent primitives;
LeNet-5 is written out layer by layer, as in the original paper.

| | MobileNetV1 | LeNet-5 |
|---|---|---|
| **Dataset** | CIFAR-10 | CIFAR-10 and MNIST |
| **Test accuracy** | **86.74%** (macro F1 0.867) | **66.85%** CIFAR-10 · **99.40%** MNIST |
| **Parameters** | 3,217,226 — 12.27 MiB fp32 | 62,006 / 61,706 — **0.24 MiB** |
| **MACs per image** | 43,182,080 | 651,720 / 416,520 |
| **Verified against the paper** | trunk = 3,185,088; with a 1000-class head = 4,210,088 ≈ the paper's **4.2 M** | **61,706** = the paper's "about 60,000" |
| **Training cost** | ~135 min of compute (30 epochs, CPU) | 2.8 min / 3.1 min |

**Head to head on identical CIFAR-10 data:** MobileNetV1 is **+19.89 accuracy points** at
**51.9× the parameters** and **66.3× the MACs**. LeNet-5 is not obsolete — on MNIST it hits
99.40% — but it cannot resolve CIFAR-10 detail. The comparison is dissected in
[report.md §11](report.md).

**Ablation (MobileNetV1):** an identical net without depthwise separation needs **8.79×** the
parameters, **8.54×** the compute — against the paper's 8.6×.

Full write-up: **[report.md](report.md)** (13 sections + 2 appendices) ·
Slides: **`outputs/mobilenetv1_deck.pptx`** (27 slides, speaker notes) ·
Voiceover script: **[voiceover.md](voiceover.md)** (generated from the notes)

---

## 1. Quick start

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn python-pptx

# MobileNetV1 on CIFAR-10  (~2.25 h on CPU)
python src/train.py --arch mobilenetv1 --dataset cifar10 --tag mobilenetv1_a1.0

# LeNet-5 on both datasets  (~3 min each)
python src/train.py --arch lenet5 --dataset mnist   --tag lenet5_mnist
python src/train.py --arch lenet5 --dataset cifar10 --tag lenet5_cifar10

# Evaluate; each run also rebuilds outputs/architecture_comparison.*
python src/evaluate.py --tag mobilenetv1_a1.0
python src/evaluate.py --tag lenet5_mnist
python src/evaluate.py --tag lenet5_cifar10

python src/build_deck.py                        # regenerate the .pptx
```

Already-trained artefacts are committed in `outputs/`, so **the evaluate steps alone reproduce
every figure and number** without retraining. Only run `train.py` to retrain from scratch.

**Smoke test** — exercises the whole pipeline in a few minutes (training is capped at 5 steps;
evaluation over the test set is the bulk of the time):

```bash
python src/train.py --tag smoke --epochs 1 --steps-per-epoch 5
python src/evaluate.py --tag smoke
```

---

## 2. Scripts

| Script | Purpose | Runtime |
|---|---|---|
| `src/mobilenet.py` | MobileNetV1. `depthwise_separable_block`, `mobilenet_v1`, `mobilenet_v1_standard_conv` | import |
| `src/lenet5.py` | LeNet-5. `lenet5`, with configurable activation / subsampling / BatchNorm | import |
| `src/data.py` | CIFAR-10 + MNIST loading, preprocessing, augmentation, `tf.data` pipelines | import |
| `src/train.py` | Training loop, `WarmupCosineDecay` schedule, best-val checkpointing | ~2.3 h total |
| `src/flops.py` | Analytic MAC and parameter counters | seconds |
| `src/benchmark.py` | Batch-size throughput measurement | ~4 min |
| `src/evaluate.py` | Test evaluation, all figures/tables, cross-architecture comparison | ~2 min each |
| `src/build_deck.py` | Generates the `.pptx` with speaker notes | seconds |
| `src/build_voiceover.py` | Renders the notes to a standalone `voiceover.md` script with timings | seconds |
| `src/audit_deck.py` | Static deck check: bounds, overflow, notes coverage | seconds |
| `src/verify_deck_layout.py` | Exact layout check via PowerPoint's engine (optional) | seconds |
| `src/verify_docs.py` | Asserts every number quoted in `report.md`/`README.md` matches `outputs/*.json` | seconds |

### Useful flags

```bash
# Architecture and dataset
python src/train.py --arch lenet5 --dataset mnist

# Width multiplier (generates the whole MobileNet family)
python src/train.py --arch mobilenetv1 --alpha 0.5 --tag mobilenetv1_a0.5

# Ablation baseline: same schedule, no depthwise separation (~8.5x the compute)
python src/train.py --standard-conv --tag standard_conv

# LeNet-5 as originally published (tanh + average subsampling, no ReLU/maxpool)
python src/train.py --arch lenet5 --dataset mnist \
    --lenet-activation tanh --lenet-pooling avg --tag lenet5_tanh_avg_mnist

# Hyperparameters
python src/train.py --epochs 50 --batch-size 64 --learning-rate 0.05 --dropout 0.2

# Evaluate a specific run, skipping the architecture tables
python src/evaluate.py --tag mobilenetv1_a1.0 --skip-experiments

# Faster batch benchmark
python src/benchmark.py --timed 10 --repeats 2
```

`--alpha` is the width multiplier, and applies to MobileNetV1 only. Changing it changes the
number of steps in the learning-rate schedule, so accuracy comparisons across α are only fair
at equal epoch count. The three LeNet-5 flags change the *operators*, not the topology: all
three variants have exactly 61,706 parameters, since the flags swap ReLU for tanh, max pool for
average pool, and add or omit BatchNorm.

---

## 3. Code explanation

### 3.1 The core idea

A standard convolution does two jobs at once: **spatial filtering** (combining neighbouring
pixels) and **channel mixing** (combining channels). MobileNet separates them.

```
Input (H × W × C_in)
  └─ DepthwiseConv2D 3×3, stride s, depth_multiplier=1   spatial filtering, no channel mixing
       └─ BatchNorm → ReLU6
            └─ Conv2D 1×1, filters = C_out              channel mixing, no spatial extent
                 └─ BatchNorm → ReLU6
```

The multiply-accumulate counts:

```
standard    = d_k² · m · n · d_f²
separable   = ( d_k² · m  +  m · n ) · d_f²
────────────────────────────────────────────
reduction   = d_k² · n / ( d_k² + n )
```

The reduction factor depends **only on kernel size and output channel count** — not on
resolution, not on input channels. For `d_k = 3` and 512 channels that is **8.84×**, at any
input size. Full derivation in [report.md §2](report.md).

### 3.2 Why Keras's `DepthwiseConv2D` is the right primitive

`layers.DepthwiseConv2D(..., depth_multiplier=1)` applies one filter per input channel with no
channel mixing. Internally it is a `Conv2D` with `groups == C_in`, i.e. a block-diagonal kernel.
That is why `src/flops.py` charges it only `k_h · k_w` per output channel:

```python
# src/flops.py — one formula covers both cases
MACs = H_out · W_out · C_out · (k_h · k_w · C_in / groups)
```

For a depthwise layer `groups == C_in`, so the term collapses automatically. No special-casing.

### 3.3 Why the input layer is unnamed

```python
# src/mobilenet.py
inputs = keras.Input(shape=input_shape)   # deliberately NOT name="image"
```

Keras 3 requires a model with a *named* input to be fed **dicts** rather than plain tuples,
which breaks `model.fit` on a standard `(x, y)` `tf.data` pipeline. Leaving the input unnamed
keeps the `(x, y)` path working. This looks like an omission; it is load-bearing.

### 3.4 Why the stem stride is 1, not 2

The paper's stem is `3×3, stride 2`, appropriate for 224×224 ImageNet images. At CIFAR-10's
32×32, halving immediately would discard three-quarters of the signal before the network does
any work. The five stride-2 layers in the body are unchanged and still take
32 → 16 → 8 → 4 → 2 → 1, mirroring the paper's 224 → 112 → 56 → 28 → 14 → 7. Every channel
count is unchanged. This is the **only** architectural adaptation.

### 3.5 Why ReLU6, and why no conv biases

The paper's body text says "ReLU", but the authors' released TensorFlow models use the
hard-limited ReLU6, and this follows the released models. Clamping to [0, 6] bounds the dynamic
range, which is what keeps the network stable in the float16/int8 regime mobile hardware uses.

Convolution biases are omitted because BatchNorm already learns a per-channel shift — a conv
bias would be redundant and would be absorbed into the BatchNorm offset during training.

### 3.6 Why augmentation is per-image, not per-batch

Keras 3's image-preprocessing layers (`RandomCrop`, `ZeroPadding2D`, …) require rank-4 input and
apply **one shared decision across the whole batch**. Every image in a batch would get the
*same* crop, which defeats the purpose. `src/data.py` therefore uses `tf.image` ops applied to a
single un-batched image:

```python
# src/data.py — per element, so randomness is per image
def _augment(x, y, shift_px, hflip):
    x = tf.pad(x, [[shift_px, shift_px], [shift_px, shift_px], [0, 0]])
    x = tf.image.random_crop(x, (32, 32, x.shape[2]))   # random translation
    if hflip:
        x = tf.image.random_flip_left_right(x)
    return x, y
```

`shift_px` is 4 for CIFAR-10 and 2 for MNIST, and `hflip` is `False` for MNIST — a mirrored 3
is not a 3, so horizontal reflection is label-preserving for natural photographs but not for
digits.

### 3.7 Why warmup + cosine decay

A network with 27 BatchNorm layers starting from random initialisation is unstable at a large
learning rate; the first few hundred steps produce large, uninformative gradients. `train.py`
ramps the rate in linearly over 3 epochs, then decays it to zero on a cosine tail.

`WarmupCosineDecay` is decorated with `@keras.saving.register_keras_serializable()` so the
schedule survives the round-trip through a `.keras` checkpoint. `evaluate.py` also passes it via
`custom_objects` — **both** are needed, the first for new checkpoints, the second for ones saved
before the decorator existed.

### 3.8 Why MACs are counted analytically, not profiled

`src/flops.py` recomputes the cost from a built model using the paper's own "Mult-Adds"
convention — one multiply-accumulate per output element. A profiler would report whatever the
hardware actually executed, which depends on fusion and backend. The analytic count is
reproducible and matches the paper's definition, which is what makes the 8.54× vs 8.6×
comparison meaningful.

Keras 3 removed per-layer `.input_shape`, so `trace_shapes()` recovers intermediate shapes by
replaying the already-built layers on a fresh `keras.Input`. This reads no weights and is
effectively free.

### 3.9 The three deliberate deviations from the paper

| Deviation | Paper | Here | Why |
|---|---|---|---|
| Stem stride | 2 | 1 | 32×32 cannot afford to halve immediately (§3.4) |
| Dropout | 0.001 | 0.1 | 0.001 is tuned for 1.28 M images / 1000 classes; 45k images need more |
| LR schedule | "SGD" | warmup + cosine | Necessary for stable BatchNorm training |

Everything else — all channel counts, strides, block repeats, ReLU6, absent conv biases, and the
global-pooled head — follows the paper directly.

### 3.10 Why LeNet-5 uses VALID padding, and why MNIST is padded up to 32×32

```python
# src/lenet5.py
x = layers.Conv2D(6, kernel_size=5, padding="valid", name="C1_conv")(inputs)
```

All three LeNet-5 convolutions use `valid`, never `same`. The map therefore shrinks at every
step: **32 → 28 → 14 → 10 → 5 → 1**. That terminating at exactly 1×1 is what lets C5 be a 5×5
convolution whose entire output is a single 120-unit vector — i.e. a dense layer expressed as a
convolution.

This is why MNIST is zero-padded from 28×28 to 32×32 in `data.py`. Feeding 28×28 directly
breaks the network: C5's 5×5 kernel would find no valid 5×5 window in a 5×5 map reached from
28×28 (28→24→12→8→4, and a 5×5 kernel on a 4×4 map is invalid). Padding preserves the original
shape arithmetic exactly, and is the standard way LeNet-5 is run on MNIST.

Padding happens **before** normalisation, so the zero border stays exactly zero afterwards.

### 3.11 LeNet-5's three deviations, and why none change the parameter count

| Deviation | Original | Default here | Flag |
|---|---|---|---|
| Non-linearity | tanh (C1–C5), RBF output | ReLU, softmax output | `--lenet-activation tanh` |
| Subsampling | average pool with fixed gain | max pool | `--lenet-pooling avg` |
| C3 connectivity | sparse hand-designed table | dense | not flaggable — documented in the module docstring |
| BatchNorm | none | none | `--lenet-batch-norm` to add it |

Every one of these is an *operator* swap, not a topology change, so all variants hold exactly
**61,706** parameters. That is the point worth making: the architecture — layer types, kernel
sizes, channel counts, strides — is LeNet-5 in every case.

The dense-C3 deviation is the honest exception. LeCun et al. connected each of the 16 C3 units to
a *subset* of the 6 S2 maps via a hand-designed binary table, which is why the original C3 is
smaller than this implementation's 2,416 weights. Nearly every modern reimplementation connects
C3 densely, and encoding the 16-row table would add a lot of bespoke code for a model whose
parameter count is then no longer the canonical 61,706. It is documented rather than
implemented.

---

## 4. Project layout

```
├── README.md                        this file
├── report.md                        full academic report (13 sections + 2 appendices)
├── src/
│   ├── mobilenet.py                 MobileNetV1 + the standard-conv ablation
│   ├── lenet5.py                    LeNet-5
│   ├── data.py                      CIFAR-10 + MNIST loading, preprocessing, augmentation
│   ├── train.py                     training loop, LR schedule, checkpointing, --arch/--dataset
│   ├── flops.py                     analytic MAC and parameter counters
│   ├── benchmark.py                 batch-size throughput measurement
│   ├── evaluate.py                  test evaluation, all figures and tables, comparison
│   ├── build_deck.py               generates the .pptx
│   ├── build_voiceover.py          renders notes to voiceover.md with timings
│   ├── audit_deck.py                static deck check (no PowerPoint needed)
│   ├── verify_deck_layout.py        exact layout check via PowerPoint COM
│   └── verify_docs.py               checks report/README numbers against outputs/*.json
└── outputs/
    ├── mobilenetv1_a1.0.keras        best-val checkpoint (12 MB)
    ├── mobilenetv1_a1.0_metrics.json  every headline number, machine-readable
    ├── mobilenetv1_a1.0_history.json  30-epoch history + per-epoch timings
    ├── mobilenetv1_a1.0_arch.json     params/MACs by layer type
    ├── mobilenetv1_a1.0_classification_report.txt
    ├── mobilenetv1_a1.0_confusion_matrix.csv
    ├── lenet5_mnist.keras            LeNet-5 MNIST checkpoint (0.25 MB)
    ├── lenet5_mnist_metrics.json     99.40% test accuracy
    ├── lenet5_cifar10.keras          LeNet-5 CIFAR-10 checkpoint (0.25 MB)
    ├── lenet5_cifar10_metrics.json   66.85% test accuracy
    ├── architecture_comparison.csv   the LeNet-5 vs MobileNetV1 table
    ├── architecture_comparison.json  same rows + derived ratios
    ├── width_multiplier.csv          α sweep
    ├── separable_vs_standard.csv     the ablation
    ├── batch_size_benchmark.csv      throughput sweep
    ├── val_split_cifar10.npz         the held-out validation images
    ├── val_split_mnist.npz           the held-out validation images
    ├── mobilenetv1_deck.pptx         27 slides with speaker notes
    └── figures/                      all PNG figures
```

---

## 5. Results

### Headline

| Metric | MobileNetV1 / CIFAR-10 | LeNet-5 / CIFAR-10 | LeNet-5 / MNIST |
|---|---|---|---|
| Top-1 test accuracy | **86.74%** | **66.85%** | **99.40%** |
| Macro F1 | 0.8672 | 0.6666 | 0.9940 |
| Best validation accuracy | 86.84% (ep 30) | 66.86% (ep 28) | 99.33% (ep 22) |
| Parameters | 3,217,226 | 62,006 | 61,706 |
| MACs per image | 43,182,080 | 651,720 | 416,520 |
| Model size (fp32) | 12.27 MiB | 0.24 MiB | 0.24 MiB |
| Training time (30 ep) | 487 min wall / 135 min compute | 2.8 min | 3.1 min |

MobileNetV1's test accuracy sits **0.1 points** below its validation accuracy — the validation
split was representative and the model is not overfitted to it.

### LeNet-5: the ablation is the model

| Layer | Arithmetic | Parameters |
|---|---|---|
| C1 | 5·5·1·6 + 6 | 156 |
| C3 | 5·5·6·16 + 16 | 2,416 |
| C5 | 5·5·16·120 + 120 | 48,120 |
| F6 | 120·84 + 84 | 10,164 |
| Out | 84·10 + 10 | 850 |
| **Total** | | **61,706** |

**61,706 is the "about 60,000 parameters" figure of the 1998 paper** — the same kind of
fidelity check that validated MobileNetV1 against its 4.2 M. Note that C5 alone is 78% of the
model: LeNet-5's capacity sits in one expensive high-channel convolution, the opposite of
MobileNetV1's profile.

On MNIST the result essentially reproduces the original — LeCun et al. reported 99.05%, this
implementation reaches **99.40%**, with per-class F1 between 0.9921 and 0.9959 across all ten
digits. On CIFAR-10 it reaches 66.85%, and the per-class profile shows why: cat (F1 0.458) and
dog (0.560) are the weakest classes and dominate the errors, the same ambiguity MobileNetV1
shows. A network this small cannot resolve fine texture at 32×32.

### The MobileNetV1 ablation

Identical layer schedule, identical channels; the factorisation is the only difference.

| Variant | Parameters | MACs / image |
|---|---|---|
| Depthwise separable | 3,217,226 | 43,182,080 |
| Standard 3×3 | 28,279,338 | 368,946,176 |
| **Reduction** | **8.79×** | **8.54×** |

Against the paper's Table 4 (ImageNet, 224×224): **8.54× vs 8.6×** — agreement to within 1%, and
predicted in closed form by `d_k²n/(d_k²+n)`.

### Where the compute goes

| Layer type | MACs / image | Share |
|---|---|---|
| Conv2D (1×1 pointwise) | 41,779,200 | **96.8%** |
| DepthwiseConv2D (3×3) | 1,391,616 | 3.2% |
| Dense + pooling | 11,264 | ~0% |

Depthwise separable convolution does not make every layer cheap — it makes the **expensive kind
unnecessary**, by shrinking the resolution at which expensive work happens.

### Reading the head-to-head

Three observations, because the naive summary ("bigger is better") is wrong here:

1. **Parameter count is not the efficiency metric.** 51.9× the parameters and 66.3× the MACs are
   different numbers, and MACs are what determine device latency. The framing is not "52× the
   parameters bought 20 points" but "52× the parameters bought 20 points, *and* the architecture
   was designed to make those 52× parameters cheap." Both halves matter; the second is the
   actual contribution.
2. **The gap is depth and width, not the template.** Both models have the same outer structure
   — conv, non-linearity, downsample, repeat, dense head. Swapping the template would not close
   a 20-point gap. See report.md §10.
3. **The error structure is identical.** Both fail on cat and dog, and both preserve the
   ordering of classes by difficulty. The smaller model is not confused about *different*
   things; it is confused about more of the same ones.

And the comparison is not one-directional: on MNIST, LeNet-5's 99.40% at 0.24 MiB and 3 minutes
of CPU training is the better engineering answer, and it has no headroom left to optimise.

---

## 6. Known limitations

Stated plainly, because each bounds how far the results generalise:

- **The accuracy half of the ablation was not measured.** The 8.5× compute reduction is
  established; what it costs in accuracy is not — the standard-conv baseline was never trained
  (it would take ~an order of magnitude longer). The paper's 1.0-point difference is the best
  available reference, but it is the paper's number, not this project's.
- **CPU-only.** TensorFlow has no native Windows GPU support after 2.10. Timings are relative,
  not absolute.
- **Single run, single seed** (1337). Differences below ~±0.5% are not meaningful.
- **Stopped before convergence.** Validation was still rising at epoch 30 when the cosine tail
  hit zero.
- **Overfitting unresolved.** A 10.7-point train/validation gap. Stronger augmentation, mixup,
  cutout and label smoothing would all be expected to help; none were tried.
- **Batch size chosen on convention.** The benchmark showed throughput is flat across 32–256, so
  it could not inform the choice; the accuracy trade-off was not measured.
- **No LeNet-5 ablation.** The decisive MobileNetV1 experiment has no LeNet-5 counterpart.
  Nothing here decomposes which of LeNet-5's seven layers contribute what; a reduced-filter or
  no-sparsity variant would be needed to explain its 66.85% on CIFAR-10.
- **LeNet-5 was not tuned per architecture.** It shares MobileNetV1's recipe, which fits it in
  most respects but not all — warmup exists to stabilise 27 BatchNorm layers and LeNet-5 has
  none, and weight decay 5e-4 is calibrated for 3.2 M parameters, not 62 k. Its numbers are a
  floor rather than a ceiling.
- **The training-time comparison is unfair to MobileNetV1.** The 174× wall-clock ratio is
  dominated by host stalls during the MobileNetV1 run. The comparable figure is the median
  epoch: ~4.5 min against ~5.6 s, about 48×, close to the 66.3× MAC ratio as expected.

### On training time

Total wall-clock was 487 min, but the per-epoch timings are strongly bimodal: epochs 8–18 and
20–30 each took 4.0–4.9 min, while **epoch 4 took 286 min and epoch 19 took 66 min**. Those are
host stalls (suspend/contention), not compute. Excluding them, actual compute was **135 min**;
the median epoch is 4.4 min, implying ~0.77 s/step at batch 128.

Both figures are reported deliberately: quoting only 135 min would misstate how long the run
occupied the machine, and quoting only 487 min would misrepresent the cost of the method.

---

## 7. Requirements

- Python 3.13.15 (3.10+ should work)
- TensorFlow 2.21.0, Keras 3.15.1
- NumPy, pandas, scikit-learn, Matplotlib, seaborn
- `python-pptx` for the deck; `pywin32` only for the optional PowerPoint layout check

CIFAR-10 downloads automatically on first run (~170 MB, cached in `~/.keras/datasets`).
MNIST downloads automatically too (~11 MB).

---

## 8. References

1. Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. **"Gradient-Based Learning Applied to Document
   Recognition."** *Proceedings of the IEEE* 86(11), 1998.
2. A. G. Howard, M. Zhu, B. Chen, et al. **"MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications."** *arXiv:1704.04861*, 2017.
3. A. Krizhevsky. **"Learning Multiple Layers of Features from Tiny Images."** 2009.
4. K. He, et al. **"Deep Residual Learning for Image Recognition."** *CVPR 2016*.
5. F. Chollet. **"Xception: Inception with Depthwise Separable Convolutions."** *arXiv:1610.02358*, 2016.
