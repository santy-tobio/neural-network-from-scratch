# Neural Networks from Scratch

A complete implementation of multilayer perceptrons (MLPs) built entirely from scratch using NumPy/CuPy, with systematic hyperparameter exploration and validation against PyTorch. This project demonstrates deep understanding of neural network fundamentals by implementing forward propagation, backpropagation, optimizers (SGD, Adam), and regularization techniques without relying on high-level ML frameworks.

## Overview

This work addresses handwritten character classification on the EMNIST Bymerge dataset (809,555 images, 47 classes). The implementation handles significant class imbalance (15:1 ratio), achieves 88.83% test accuracy, and includes comprehensive robustness analysis under Gaussian noise perturbations.

**Key contributions:**
- From-scratch neural network implementation using only CuPy for GPU acceleration
- Custom backpropagation engine with automatic gradient computation
- SGD and Adam optimizers implemented manually with momentum and adaptive learning rates
- Validation against PyTorch showing <0.3% accuracy difference, confirming implementation correctness
- Systematic exploration of 12 model configurations evaluating capacity, optimizers, regularization, and architecture depth
- Robustness analysis demonstrating batch normalization's impact on noise resistance

## Results

The best model (M3b) achieves:
- **88.83%** test accuracy (+1.38% over baseline)
- **85.81%** macro F1-score across 47 imbalanced classes
- **87.60%** accuracy under Gaussian noise (σ=0.1), demonstrating exceptional robustness

Key findings:
- Model capacity (~112k to ~445k parameters) is the most critical factor for performance
- Adam outperforms SGD with momentum by 6.11 percentage points
- Deep architectures [400, 240, 120] slightly outperform wide ones [370, 370] with equivalent capacity
- Batch normalization dramatically improves robustness: only 1.39% degradation under noise vs 8.48% for baseline

## Implementation Details

The codebase is organized into modular components:

**Core neural network (`neural_net/`)**
- `layers/`: Dense, activation (ReLU, SiLU, GELU), batch normalization, dropout
- `models/`: MLP architecture with flexible layer configuration
- `optimizers/`: SGD with momentum, Adam with bias correction
- `training/`: Training loop, early stopping, L2 regularization, learning rate schedulers
- `evaluation/`: Metrics (accuracy, F1-macro, cross-entropy), confusion matrices
- `data/`: Stratified splitting preserving class proportions

**Experimental framework (`experiments/`)**
- Unified configuration system for reproducible experiments
- Automated model construction and training pipeline
- Support for both custom (CuPy) and PyTorch implementations

**GPU memory management**
- Hybrid CPU-GPU strategy: dataset in RAM, mini-batches transferred to GPU on-demand
- Enables training on limited hardware (4GB VRAM) without out-of-memory errors
- Batch size 512 empirically determined as optimal balance between speed and stability

## Project Structure

```
.
├── neural_net/          # Core implementation
│   ├── layers/          # Layer implementations (dense, activation, normalization)
│   ├── models/          # MLP architectures (custom and PyTorch)
│   ├── optimizers/      # SGD and Adam optimizers
│   ├── training/        # Training loop, regularization, schedulers
│   ├── evaluation/      # Metrics and evaluation utilities
│   └── data/            # Data loading and preprocessing
├── experiments/         # Experimental configurations and runners
├── notebooks/           # Jupyter notebook with complete analysis
├── report/              # Academic reports (English and Spanish)
│   ├── Tobio_Santiago_Report_TP3.tex     # English version
│   ├── Tobio_Santiago_Informe_TP3.tex    # Spanish version
│   └── figures/         # Generated plots and visualizations
├── figures/             # High-quality figures for README
└── data/                # EMNIST Bymerge dataset
```

## Installation

### Requirements
- Python ≥3.10
- CUDA-compatible GPU (optional but recommended)
- CUDA Toolkit 13.x for CuPy GPU acceleration

### Setup

```bash
# Clone repository
git clone https://github.com/santy-tobio/neural-network-from-scratch.git
cd neural-networks-from-scratch

# Install core dependencies
pip install -e .

# Install PyTorch for validation experiments (optional)
pip install -e .[pytorch]

# Install development tools (optional)
pip install -e .[dev]
```

**Note:** CuPy requires CUDA Toolkit. For CPU-only, replace `cupy-cuda13x` in `pyproject.toml` with `numpy` (performance will be significantly reduced).

## Usage

### Running experiments

All experiments are configured in `experiments/configs/configurations.py`. To reproduce results:

```python
from experiments.runner import run_experiment
from experiments.configs import get_m1g_config

# Run best custom model (M1g)
config = get_m1g_config()
results = run_experiment(config)
```

### Training a model

```python
from neural_net import NeuralNetwork
from neural_net.optimizers import Adam
from neural_net.training import Trainer

# Define architecture
model = NeuralNetwork(
    input_dim=784,
    hidden_dims=[400, 240, 120],
    output_dim=47,
    activation='relu'
)

# Configure optimizer
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

# Train
trainer = Trainer(model, optimizer, batch_size=512)
history = trainer.train(X_train, y_train, X_val, y_val, epochs=50)
```

### Evaluating robustness

```python
from neural_net.evaluation import evaluate_robustness

# Test model under Gaussian noise
noise_levels = [0.0, 0.1, 0.2, 0.3]
robustness = evaluate_robustness(model, X_test, y_test, noise_levels)
```

## Experimental Configurations

**M0 - Baseline**
- Architecture: [128, 64]
- Optimizer: SGD (η=0.001, no momentum)
- Result: 87.45% test accuracy

**M1g - Best Custom**
- Architecture: [400, 240, 120] (444,847 parameters)
- Optimizer: Adam (α=0.001, β₁=0.9, β₂=0.999)
- Regularization: Early stopping (patience=5)
- Result: 88.58% test accuracy

**M2 - PyTorch Validation**
- Same as M1g, reimplemented in PyTorch
- Result: 88.36% test accuracy (0.22% difference validates custom implementation)

**M3b - Best Overall**
- Architecture: [400, 400] with batch normalization
- Activation: SiLU (Swish)
- Regularization: Dropout 0.3, weight decay 0.01, early stopping
- Result: 88.83% test accuracy, exceptional robustness

## Academic Report

Complete technical details, mathematical formulations, and experimental analysis are available in the academic report:

- **English:** [`report/Tobio_Santiago_Report_TP3.pdf`](report/Tobio_Santiago_Report_TP3.pdf)
- **Spanish:** [`report/Tobio_Santiago_Informe_TP3.pdf`](report/Tobio_Santiago_Informe_TP3.pdf)

The report covers:
- Theoretical foundations (MLP architecture, backpropagation derivation, optimization algorithms)
- Dataset analysis and stratified splitting strategy
- Systematic hyperparameter exploration across 12 configurations
- Implementation validation against PyTorch
- Robustness analysis under adversarial noise
- Complete mathematical equations for Adam optimizer, cross-entropy loss, and regularization

## Interactive Notebook

The Jupyter notebook [`notebooks/Tobio_Santiago_Notebook_TP3.ipynb`](notebooks/Tobio_Santiago_Notebook_TP3.ipynb) provides:
- Complete implementation walkthrough
- Exploratory data analysis with visualizations
- Training curves and learning dynamics
- Confusion matrices and error analysis
- Robustness experiments with noise visualization
- Comparison plots across all model configurations

## Technical Highlights

**Custom backpropagation engine**
The implementation computes gradients analytically through the chain rule without automatic differentiation frameworks. Each layer maintains forward and backward methods:

```python
# Forward pass computes activations and caches for backward
z = layer.forward(a_prev)  # Linear transformation
a = activation(z)          # Nonlinearity

# Backward pass computes gradients via chain rule
dz = da * activation_derivative(z)
dW = dz @ a_prev.T / batch_size
db = dz.mean(axis=1)
da_prev = W.T @ dz
```

**Adam optimizer with bias correction**
Full implementation of adaptive moment estimation including bias correction terms:

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * (grad ** 2)
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

**Hybrid memory management**
Dataset remains in CPU RAM while only mini-batches are transferred to GPU, enabling training on consumer hardware:

```python
for batch in data_loader:
    # Transfer batch to GPU
    X_batch_gpu = cp.asarray(batch['X'])
    y_batch_gpu = cp.asarray(batch['y'])

    # Forward and backward on GPU
    loss, grads = model.train_step(X_batch_gpu, y_batch_gpu)

    # Update parameters on GPU
    optimizer.update(model.parameters, grads)
```

## Validation Against PyTorch

The custom implementation was validated by reimplementing the optimal configuration (M1g) in PyTorch with identical hyperparameters. Results:

| Metric | Custom (CuPy) | PyTorch | Difference |
|--------|---------------|---------|------------|
| Test Accuracy | 88.58% | 88.36% | 0.22% |
| Validation Loss | 0.3322 | 0.3494 | 5.18% |

This <0.3% accuracy difference confirms that the from-scratch implementation correctly replicates professional frameworks, validating the manual implementation of forward propagation, backpropagation, and optimization algorithms.

## Author

**Santiago Tobio**
AI Engineering Student, Universidad de San Andrés
Buenos Aires, Argentina
[stobio@udesa.edu.ar](mailto:stobio@udesa.edu.ar)

## License

MIT License - see LICENSE file for details
