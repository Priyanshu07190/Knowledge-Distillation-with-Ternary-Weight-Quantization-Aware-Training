# Knowledge Distillation with Ternary-Weight Quantization-Aware Training (QAT)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-blue?style=for-the-badge)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![HuggingFace Datasets](https://img.shields.io/badge/HuggingFace-CIFARNet-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/EleutherAI/cifarnet)
[![Model Compression](https://img.shields.io/badge/Compression-12.86x%20(~92.2%25)-green?style=for-the-badge)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

---

## 📌 Executive Summary

This project implements an end-to-end framework for **ultra-low-bit deep neural network compression** combining **Ternary Weight Networks (TWN)**, **Quantization-Aware Training (QAT)** via the **Straight-Through Estimator (STE)**, and **Knowledge Distillation (KD)**.

The framework compresses a full-precision model down to discrete ternary weights $\mathbf{\{-\alpha, 0, +\alpha\}}$ on **CIFAR-10** while evaluating generalization robustness on **EleutherAI's CIFARNet** and real-world external test sets.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          TEACHER NETWORK                               │
│              ResNet-34 FP32 (28,957,002 parameters)                    │
│             Top-1 Test Accuracy: 93.71%  |  Storage: ~110.5 MB         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
               Soft Targets: Logits Scaled by Temperature (T)
               KL Divergence Loss: L_KD = T² · KL(σ(z_s/T) || σ(z_t/T))
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          STUDENT NETWORK                               │
│     Ternary ResNet-18 QAT + KD (11,173,962 parameters, 98.4% Ternary)  │
│          Weight Discretization: W ∈ {-α, 0, +α} via STE                │
│         Top-1 Test Accuracy: 93.55% (Only 0.39% drop from FP32!)       │
│         Sparsity: ~60.8% - 64.2%  |  Theoretical Size: 3.31 MB         │
│          Theoretical Compression Ratio: 12.86x (92.23% reduction)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Foundations & Architecture

### 1. CIFAR-10 Architectural Adaptations
Standard ResNet architectures designed for ImageNet ($224 \times 224$) feature an initial $7 \times 7$ convolution (stride 2) followed by a $3 \times 3$ max-pooling operation. On CIFAR-10 ($32 \times 32$), this aggressive downsampling reduces resolution to $8 \times 8$ before residual stages, destroying critical spatial features.
- **Stem Modification**: The stem is replaced by a $3 \times 3$ convolution, `stride=1`, `padding=1`, with **no initial max-pooling**, preserving spatial resolution ($32 \times 32$) into Stage 1.
- **Classification Head**: Replaced with an adaptive average pooling layer followed by a $512 \to 10$ linear layer.

### 2. Ternary Weight Quantization (TWN Formulation)
Following the formulation of Li et al. (2016), layer weights are quantized into three discrete values:
$$W_i^t = \begin{cases} +\alpha, & W_i > \Delta \\ 0, & |W_i| \le \Delta \\ -\alpha, & W_i < -\Delta \end{cases}$$

Where:
- **Optimal Dynamic Threshold $\Delta$**:
  $$\Delta = 0.7 \times \mathbb{E}[|W|] = \frac{0.7}{n}\sum_{i=1}^n |W_i|$$
- **Layer-wise Positive / Negative Scale Factor $\alpha$**:
  $$I_\Delta = \{i \mid |W_i| > \Delta\}, \quad \alpha = \frac{1}{|I_\Delta|}\sum_{i \in I_\Delta} |W_i|$$

### 3. Straight-Through Estimator (STE)
Because quantization is a piece-wise step function whose derivative is zero almost everywhere, standard gradient descent cannot update continuous weights. The **Straight-Through Estimator (STE)** decouples the forward pass from the backward pass:
$$W_{\text{ste}} = W + \text{detach}(W^t - W)$$
- **Forward Pass**: $W_{\text{ste}} = W^t \in \{-\alpha, 0, +\alpha\}$.
- **Backward Pass**: $\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W_{\text{ste}}}$, allowing continuous FP32 latent weights to receive exact gradient updates.

### 4. Knowledge Distillation Loss Formulation
To compensate for quantization noise, the student is trained using a composite objective:
$$\mathcal{L}_{\text{total}} = (1 - \lambda)\mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \lambda T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right)$$

Where:
- $z_s, z_t$ are the student and teacher logits respectively.
- $T$ is the distillation temperature ($T > 1$ softens class probabilities).
- $\lambda \in [0, 1]$ balances task-specific hard cross-entropy and soft teacher guidance.
- $T^2$ balances the gradient scale as $T$ scales the logits.

---

## 📊 Comprehensive Experimental Results

All models were trained for **182 epochs** on CIFAR-10 with identical optimization schedules: SGD optimizer, initial learning rate $\eta_0 = 0.1$, momentum $0.9$, weight decay $10^{-4}$, batch size $128$, and MultiStepLR decay at epochs $[91, 136]$ with $\gamma = 0.1$.

### 1. Master Model Benchmark Comparison

| Model Architecture | Precision | Parameters | Model Size | Weight Sparsity | Best Val Acc | CIFAR-10 Test Acc | CIFARNet (OOD) Acc | External 101 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-34 Teacher** | FP32 | 28,957,002 | ~110.5 MB | 0.00% | 94.06% | **93.71%** | — | — |
| **ResNet-18 Baseline** | FP32 | 11,173,962 | 42.63 MB | 0.00% | 94.70% | **93.94%** | — | — |
| **Ternary Student (Exp 1)**<br>($T=4.0, \lambda=0.50$) | Ternary + FP32 | 11,173,962 | **3.31 MB** | 64.16% | 93.92% | **93.21%** | — | **97.03%** |
| **Ternary Student (Exp 2)**<br>($T=2.0, \lambda=0.25$) | Ternary + FP32 | 11,173,962 | **3.31 MB** | 58.79% | 93.54% | **93.42%** | **73.11%** | — |
| **Ternary Student (Exp 3)**<br>($T=3.0, \lambda=0.40$) | Ternary + FP32 | 11,173,962 | **3.31 MB** | 62.99% | 94.18% | **93.21%** | **71.90%** | — |
| **Ternary Student (Exp 4)**<br>($T=4.0, \lambda=0.25$) ⭐ | Ternary + FP32 | 11,173,962 | **3.31 MB** | 60.81% | **94.40%** | **93.55%** | **72.76%** | — |
| **Ternary Student (Exp 5)**<br>($T=4.0, \lambda=0.40$) | Ternary + FP32 | 11,173,962 | **3.31 MB** | 64.16% | 94.18% | **93.31%** | **72.59%** | — |

⭐ **Best Student Configuration**: **Experiment 4 ($T=4.0, \lambda=0.25$)** retained **93.55% test accuracy** — within **0.39%** of the full-precision baseline — despite over **60% parameter sparsity** and **92.23% theoretical memory savings**.

---

### 2. Parameter Breakdown & Storage Efficiency

Each ternary weight can be encoded in **2 bits** (or 1 bit for non-zero indicator + 1 bit for sign), while batch normalization parameters and per-layer scale factors $\alpha$ are maintained in FP32:

$$\begin{aligned}
\text{Ternary Weights} &= 10,992,320 \times 2\text{ bits} = 21,984,640\text{ bits} \approx 2.62\text{ MB} \\
\text{FP32 Parameters} &= 181,642 \times 32\text{ bits} = 5,812,544\text{ bits} \approx 0.69\text{ MB} \\
\mathbf{\text{Total Storage}} &\approx \mathbf{3.31\text{ MB}} \quad (\text{vs. } 42.63\text{ MB for FP32 ResNet-18}) \\
\mathbf{\text{Compression Ratio}} &\approx \mathbf{12.86\times} \quad (\mathbf{92.23\%\text{ storage reduction}})
\end{aligned}$$

#### Ternary State Distribution (Exp 4 vs. Exp 5)
```
Experiment 4 (Sparsity: 60.81%):
  ├── Zero weights ( 0)   : 6,684,172  (60.81%)
  ├── Positive (+alpha)   : 1,852,825  (16.86%)
  └── Negative (-alpha)   : 2,455,323  (22.34%)

Experiment 5 (Sparsity: 64.16%):
  ├── Zero weights ( 0)   : 7,052,812  (64.16%)
  ├── Positive (+alpha)   : 1,694,898  (15.42%)
  └── Negative (-alpha)   : 2,244,610  (20.42%)
```

---

### 3. Generalization & Out-of-Distribution (OOD) Benchmarks

To ensure the ternary quantized student does not overfit to standard CIFAR-10 test distributions, models were evaluated on two independent test sets:

#### A. Synthetic CIFARNet (`EleutherAI/cifarnet` - 10,000 images)
- **Exp 2 ($T=2, \lambda=0.25$)**: **73.11%** (7,311 / 10,000 correct)
- **Exp 3 ($T=3, \lambda=0.40$)**: **71.90%** (7,190 / 10,000 correct)
- **Exp 4 ($T=4, \lambda=0.25$)**: **72.76%** (7,276 / 10,000 correct)
- **Exp 5 ($T=4, \lambda=0.40$)**: **72.59%** (7,259 / 10,000 correct)

#### Per-Class Accuracy on CIFARNet (Experiment 4)
| Class Name | Correct / Total | Accuracy |
| :--- | :---: | :---: |
| **Airplane** | 808 / 965 | 83.73% |
| **Automobile** | 691 / 1018 | 67.88% |
| **Bird** | 674 / 1023 | 65.88% |
| **Cat** | 644 / 954 | 67.51% |
| **Deer** | 866 / 1027 | 84.32% |
| **Dog** | 544 / 1051 | 51.76% |
| **Frog** | 808 / 1000 | 80.80% |
| **Horse** | 771 / 1048 | 73.57% |
| **Ship** | 763 / 946 | 80.66% |
| **Truck** | 707 / 968 | 73.04% |

#### B. External Real-World Image Evaluation (Experiment 1)
- Evaluated on **101 out-of-domain images** across all 10 CIFAR classes.
- **Accuracy**: **97.03%** (98 out of 101 images classified correctly with high confidence).

---

## 📁 Repository Structure

```
ATDL KD/
├── ATDL.ipynb              # Master interactive Jupyter notebook (all 353 cells)
├── requirements.txt        # Pinned & tested Python dependencies
├── README.md               # Detailed technical documentation and experiment results
├── data/                   # CIFAR-10 downloaded files (auto-generated)
├── checkpoints/            # Saved weights & state dicts (.pth files)
│   ├── resnet34_cifar10_teacher_best.pth
│   ├── resnet18_cifar10_fp32_best.pth
│   ├── resnet18_cifar10_ternary_kd_best.pth            (Exp 1: T=4, λ=0.50)
│   ├── resnet18_cifar10_ternary_kd_T2_lambda025_best.pth (Exp 2: T=2, λ=0.25)
│   ├── resnet18_cifar10_ternary_kd_T3_lambda04_best.pth  (Exp 3: T=3, λ=0.40)
│   ├── resnet18_cifar10_ternary_kd_T4_lambda025_best.pth (Exp 4: T=4, λ=0.25)
│   └── resnet18_cifar10_ternary_kd_T4_lambda04_best.pth  (Exp 5: T=4, λ=0.40)
└── results/                # Loss curves, accuracy plots, and training histories (.pt)
```

---

## 🚀 Step-by-Step Installation & Execution Guide

### Step 1: Environment Setup
Ensure you have **Python 3.10+** installed. We recommend setting up a virtual environment:

```bash
# Clone or navigate to the directory
cd "path/to/ATDL KD"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies

For **GPU acceleration (CUDA)**, install the appropriate PyTorch build first:
```bash
# For CUDA 12.1+ (Recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only execution (Testing / Small verification)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install all remaining requirements:
```bash
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebook

Launch the notebook interface:
```bash
jupyter lab
# or
jupyter notebook
```
Open [`ATDL.ipynb`](file:///c:/Users/radhe/Downloads/ATDL%20KD/ATDL.ipynb) and run cells sequentially.

---

## ⚠️ Important Execution Notes & Tips

1. **Hardware & VRAM Requirements**:
   - The experiments were originally trained on an **NVIDIA RTX 5000 Ada Generation GPU**.
   - Training each model for 182 epochs on CIFAR-10 takes approximately 30–60 minutes on modern GPUs.
   - If running on a GPU with $\le 6\text{ GB}$ VRAM, reduce `BATCH_SIZE = 64` or `32` in Section 1.3.

2. **External Images Path (Cell 191)**:
   - In Cell 191 (`EXTERNAL_DATASET_PATH = "/home/sem_7th/ATDL/external_images"`), an optional test was run on a local folder of 101 custom images.
   - **Tip**: If you do not have that folder, simply skip Cells 191–195 and run Cell 199 onwards (`EleutherAI/cifarnet`), which downloads the 10,000 CIFARNet test images automatically via the HuggingFace `datasets` API!

3. **HuggingFace Hub Unauthenticated Rate Limits**:
   - When downloading `EleutherAI/cifarnet` (Cell 200), if you experience Hugging Face download rate-limiting, set an environment variable:
     ```bash
     export HF_TOKEN="your_huggingface_token"  # Linux/macOS
     $env:HF_TOKEN="your_huggingface_token"     # Windows PowerShell
     ```

4. **Directory Auto-Creation**:
   - The notebook automatically executes `os.makedirs(..., exist_ok=True)` for `data/`, `checkpoints/`, and `results/` in the current working directory.

---

## ⚙️ Hyperparameter Configuration Reference

| Parameter | Value | Description |
| :--- | :---: | :--- |
| `SEED` | `42` | Global seed for `torch`, `numpy`, and `random` reproducibility |
| `BATCH_SIZE` | `128` | Batch size for train/val/test data loaders |
| `NUM_EPOCHS` | `182` | Total training epochs per model |
| `INITIAL_LR` | `0.1` | Starting learning rate for SGD |
| `MOMENTUM` | `0.9` | Momentum factor for SGD |
| `WEIGHT_DECAY` | `1e-4` | $L_2$ weight penalty |
| `LR_MILESTONES` | `[91, 136]` | MultiStepLR epochs where LR drops by $\times 0.1$ |
| `LR_GAMMA` | `0.1` | Learning rate reduction factor |
| `NUM_CLASSES` | `10` | CIFAR-10 output dimension |
| `THRESHOLD_FACTOR` | `0.7` | Threshold multiplier: $\Delta = 0.7 \times \mathbb{E}[|W|]$ |

---

## 📚 References & Academic Citations

1. **Knowledge Distillation**:
   - Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *"Distilling the Knowledge in a Neural Network."* NeurIPS Workshop on Deep Learning, 2014. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531).
2. **Ternary Weight Networks**:
   - Fengfu Li, Bo Zhang, Bin Liu. *"Ternary Weight Networks."* NIPS Workshop on Efficient Methods for Deep Neural Networks, 2016. [arXiv:1605.04711](https://arxiv.org/abs/1605.04711).
3. **Deep Residual Learning**:
   - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *"Deep Residual Learning for Image Recognition."* IEEE CVPR, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
4. **CIFAR-10 Dataset**:
   - Alex Krizhevsky. *"Learning Multiple Layers of Features from Tiny Images."* Technical Report, University of Toronto, 2009.
5. **CIFARNet**:
   - EleutherAI. *"CIFARNet: Synthetic Benchmark for Vision Generalization."* HuggingFace Datasets Hub: [`EleutherAI/cifarnet`](https://huggingface.co/datasets/EleutherAI/cifarnet).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
