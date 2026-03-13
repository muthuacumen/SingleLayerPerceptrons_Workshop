# Single-Layer Perceptrons Workshop

An interactive Jupyter notebook workshop covering the foundations of artificial neural networks — from biological inspiration through to a working medical classifier built with PyTorch.

---

## Contents

| Notebook | Description |
|---|---|
| `SingleLayerPerceptrons_Workshop.ipynb` | Main workshop — theory, exercises, and full solution |
| `Wonderland_ANN_Case_Study.ipynb` | Supporting case study: binary classification with a fun real-world scenario |

---

## Workshop Overview

The workshop builds up the concept of a perceptron in four stages:

1. **Biological to Artificial Neurons** — how a real neuron fires and how the mathematical model mirrors it
2. **Mathematical Model** — the linear combination `z = wᵀx + b` and activation functions (sigmoid, ReLU, step)
3. **Canada's Wonderland Case Study** — a worked single-layer ANN applied to a binary decision problem, covering forward pass, error calculation, and weight updates via gradient descent
4. **Challenge: Prostate Cancer Prediction ANN** — a student-led implementation applying the same architecture to a real medical classification problem

---

## Prostate Cancer Prediction — ANN Challenge

This is the capstone exercise of the workshop, implemented in `SingleLayerPerceptrons_Workshop.ipynb` (cells 49–61).

### Problem Statement

Predict whether a tumour is **malignant (cancer)** or **benign (no cancer)** from five diagnostic measurements, using a single-layer ANN built in PyTorch.

### Dataset

**Wisconsin Diagnostic Breast Cancer** dataset (`sklearn.datasets.load_breast_cancer`) — 569 patients, 30 features (5 used).

| Feature | Description |
|---|---|
| Mean Radius | Average distance from centre to perimeter |
| Mean Texture | Standard deviation of grey-scale values |
| Mean Perimeter | Average perimeter of the tumour |
| Mean Area | Average area of the tumour |
| Mean Smoothness | Local variation in radius lengths |

**Label:** `1` = Benign, `0` = Malignant

### What Was Implemented

Following the same five-step framework from the Wonderland case study:

**Step 1 — Data Loading & Preprocessing**
- Loaded the breast cancer dataset via `sklearn`
- Extracted 5 diagnostic features across 569 patients
- Applied `StandardScaler` normalisation — critical because Mean Area (≈100–2500) would otherwise dominate Mean Smoothness (≈0.05–0.16), distorting gradient descent
- Converted to PyTorch tensors

**Step 2 — Forward Pass**
- Initialised weights randomly (`torch.randn`) and bias at zero
- Computed the linear combination: `z = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + w₅x₅ + b`
- Applied sigmoid activation: `ŷ = σ(z) = 1 / (1 + e⁻ᶻ)` to produce a probability output

**Step 3 — Error Calculation**
- Demonstrated the raw error: `E = y − ŷ`
- Explained MSE vs Binary Cross-Entropy (BCE) as loss functions
- Used `nn.BCELoss()` for training — the industry standard for binary classification because it penalises confident wrong predictions exponentially

**Step 4 — Weight Update (Single Patient)**
- Computed gradients via `loss.backward()` (PyTorch autograd)
- Applied gradient descent: `w ← w − η · ∂L/∂w` with learning rate η = 0.1
- Explained the relationship between gradient descent (the algorithm) and backpropagation (how gradients flow through layers)

**Step 5 — Full Training Loop**
- Trained over 200 epochs across all 569 patients simultaneously using `torch.mv` (matrix-vector multiply)
- Tracked loss and accuracy every 50 epochs
- Achieved **~92% accuracy** on the training set

### Key Takeaways (Talking Points)

1. **Same neuron, different domain** — the Wonderland perceptron and this cancer classifier share identical architecture (`z = wᵀx + b` → sigmoid). A single-layer ANN is a general-purpose binary classifier.

2. **Feature normalisation is non-negotiable in medical data** — without `StandardScaler`, scale differences between features corrupt the gradient signal from the first update.

3. **Sigmoid output is a clinical confidence score** — the network outputs a probability, not just a label. In clinical practice the threshold can be tuned to prioritise recall (catch every cancer) over precision depending on the cost of a false negative vs. a false positive.

---

## Setup

### Requirements

- Python **3.12** (TensorFlow does not support Python 3.14+)
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Create a virtual environment with Python 3.12
py -3.12 -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.10.0 | Neural network implementation (primary framework) |
| `tensorflow` | latest | TensorFlow Playground examples and comparisons |
| `scikit-learn` | 1.8.0 | Datasets, preprocessing (`StandardScaler`) |
| `numpy` | 2.4.3 | Numerical computation |
| `matplotlib` | 3.10.8 | Visualisations |
| `pandas` | 3.0.1 | Summary tables |
| `seaborn` | 0.13.2 | Statistical plots |
| `ipykernel` | 7.2.0 | Jupyter kernel |

---

## Concepts Covered

- Biological vs. artificial neurons
- Perceptron architecture: weights, bias, activation functions
- Forward pass and prediction (`ŷ`)
- Loss functions: MSE and Binary Cross-Entropy
- Gradient Descent and Backpropagation
- Logic gates as perceptrons (AND, OR, inhibitory `A ∧ ¬B`)
- Linear separability and decision boundaries
- Feature normalisation in real-world datasets
- PyTorch autograd for automatic differentiation
