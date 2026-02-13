# ASBB: Adaptive Stochastic Barzilai-Borwein Optimizer

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An adaptive stochastic optimization algorithm for neural network training based on the Barzilai-Borwein method with dynamic parameter tuning.

## Overview

ASBB (Adaptive Stochastic Barzilai-Borwein) is a novel optimization algorithm that extends the classical Barzilai-Borwein method for stochastic optimization in deep learning. The optimizer features:

- **Adaptive step-size selection** using the Barzilai-Borwein spectral method
- **Dynamic parameter tuning** for τ (tau) and β (beta) based on training dynamics
- **Gradient accumulation** support for memory-efficient training
- **Built-in gradient clipping** for stable convergence
- **Comprehensive loss tracking** and BB statistics monitoring

## Installation

### Requirements

- Python ≥ 3.11, < 3.15
- PyTorch ≥ 2.10.0, < 3.0.0
- NumPy ≥ 2.4.2, < 3.0.0
- scikit-learn ≥ 1.8.0, < 2.0.0

### Option 1: Using Poetry (Recommended)

[Poetry](https://python-poetry.org/) is a modern dependency management tool for Python.

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   Or on Windows (PowerShell):
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/asbb.git
   cd asbb
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

### Option 2: Using pip

```bash
git clone https://github.com/yourusername/asbb.git
cd asbb
pip install torch>=2.10.0 numpy>=2.4.2 scikit-learn>=1.8.0
```

Or install in editable mode:

```bash
pip install -e .
```

## Quick Start

```python
import torch
from optimizer.ASBB import StepTunedSGD, AdaptiveBBTuner
from models.neural_network import CNN1D

# Initialize your model
model = CNN1D(input_dim=140)

# Create the ASBB optimizer
optimizer = StepTunedSGD(
    params=model.parameters(),
    alpha=1.0,
    m_tilde=0.01,
    M_tilde=10.0,
    weight_decay=1e-2
)

# Create adaptive parameter tuner
tuner = AdaptiveBBTuner(
    initial_tau=0.51,
    initial_beta=0.95,
    tau_range=(0.1, 0.8),
    beta_range=(0.85, 0.99),
    window_size=3,
    adjustment_rate=0.05
)

# Training loop with adaptive tuning
from utils.train import train_ASBB

history = train_ASBB(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=torch.nn.BCELoss(),
    epochs=20,
    adaptive_tuner=tuner
)
```

## Getting Started

### Step-by-Step Guide

1. **Set up your environment**:
   ```bash
   # Using Poetry (recommended)
   poetry install
   poetry shell
   
   # Or using pip
   pip install -e .
   ```

2. **Prepare your data**:
   ```python
   from torch.utils.data import DataLoader, TensorDataset
   
   train_loader = DataLoader(
       dataset=TensorDataset(X_train, y_train),
       batch_size=256,
       shuffle=True
   )
   ```

3. **Initialize model and optimizer**:
   ```python
   from optimizer.ASBB import StepTunedSGD, AdaptiveBBTuner
   
   model = YourModel()
   optimizer = StepTunedSGD(model.parameters(), alpha=1.0)
   tuner = AdaptiveBBTuner(initial_tau=0.51)
   ```

4. **Train your model**:
   ```python
   from utils.train import train_ASBB
   
   history = train_ASBB(
       model=model,
       train_loader=train_loader,
       test_loader=test_loader,
       optimizer=optimizer,
       criterion=torch.nn.BCELoss(),
       epochs=20,
       adaptive_tuner=tuner
   )
   ```

5. **Access training history**:
   ```python
   print(f"Final test accuracy: {history['test_metrics'][-1]['Accuracy']:.2f}%")
   print(f"Training time: {history['time'][-1]:.2f}s")
   ```

## Repository Structure

```
asbb/
├── optimizer/
│   └── ASBB.py                    # Core optimizer implementation
├── models/
│   └── neural_network.py          # CNN1D architecture
├── utils/
│   ├── train.py                   # Training loop with ASBB
│   └── classification_evaluation.py  # Performance metrics
├── ECG5000.py                     # ECG classification example
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Algorithm Details

### StepTunedSGD

The core optimizer implements a two-step Barzilai-Borwein method:

1. **First half-step**: Compute search direction using accumulated gradients
2. **Second half-step**: Refine the step using updated gradients
3. **Adaptive step-size**: Spectral step-size selection with safeguards

**Key Parameters:**
- `alpha`: Scaling factor for step-size (default: 1.0)
- `m_tilde`: Lower bound for step-size (default: 0.01)
- `M_tilde`: Upper bound for step-size (default: 10.0)
- `weight_decay`: L2 regularization coefficient (default: 0.0)

### AdaptiveBBTuner

Dynamically adjusts τ and β parameters based on training performance:

- **τ (tau)**: Controls the BB step-size selection strategy
- **β (beta)**: Momentum-like parameter for gradient smoothing
- **Window-based adaptation**: Monitors loss trends over a sliding window
- **Bounded adjustments**: Ensures parameters stay within valid ranges

## Example: ECG5000 Classification

The repository includes a complete example using the ECG5000 dataset for binary heartbeat classification.

### Dataset Setup

1. **Download the ECG5000 dataset**:
   - Create a `Data/ECG5000/` directory in the project root
   - Download the training and test files:
     - `ECG5000_TRAIN.txt`
     - `ECG5000_TEST.txt`
   
   ```bash
   mkdir -p Data/ECG5000
   # Download files to Data/ECG5000/
   ```

### Running the Example

**With Poetry:**
```bash
poetry run python ECG5000.py --epochs 20 --batch_size 256 --initial_tau 0.51
```

**With pip:**
```bash
python ECG5000.py --epochs 20 --batch_size 256 --initial_tau 0.51
```

### Expected Output

```
[1/20] [ASBB] [τ=0.510 β=0.950] Train Loss: 0.3245   Test Loss: 0.2987 | Accuracy: 89.34%   AUC: 0.8756   GM: 0.8621 | Time: 2.15s
[2/20] [ASBB] [τ=0.515 β=0.952] Train Loss: 0.2891   Test Loss: 0.2654 | Accuracy: 91.22%   AUC: 0.8932   GM: 0.8845 | Time: 4.32s
...
```

### Command-line Arguments

```
--epochs              Number of training epochs (default: 20)
--batch_size          Mini-batch size (default: 256)
--learning_rate       Initial learning rate (default: 1e-4)
--weight_decay        L2 regularization (default: 1e-2)
--alpha               ASBB alpha parameter (default: 1.0)
--m_tilde             Lower step-size bound (default: 0.01)
--M_tilde             Upper step-size bound (default: 10.0)
--window_size         Window for parameter adaptation (default: 3)
--initial_tau         Initial tau value (default: 0.51)
```

## Features

### Performance Evaluation

Comprehensive metrics for both binary and multiclass classification:

- **Accuracy**: Overall classification accuracy
- **AUC-ROC**: Area under the ROC curve (binary) or macro recall (multiclass)
- **Geometric Mean (GM)**: Balanced measure across classes
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Confusion Matrix**: Detailed classification breakdown

### Training Utilities

- Gradient accumulation for effective batch size scaling
- Gradient clipping for stability
- Learning rate scheduling support
- Detailed logging with parameter tracking
- BB statistics monitoring (step-size, curvature, etc.)

## Model Architecture

The included CNN1D model provides:

- Configurable convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Dropout regularization
- Fully connected layers for classification
- Suitable for time-series and sequential data

## Citation

If you use this optimizer in your research, please cite:

```bibtex
@software{asbb2026,
  author = {Livieris, Ioannis E. and Pintelas, Emmanuel},
  title = {Adaptive Stochastic Barzilai-Borwein algorithm for neural network training},
  year = {2026},
  publisher = {submitted to Neurocomputing},
  url = {https://github.com/yourusername/asbb}
}
```

## Authors

- **Ioannis E. Livieris** - [livieris@upatras.gr](mailto:livieris@upatras.gr)
- **Emmanuel Pintelas** - [e.pintelas@upatras.gr](mailto:e.pintelas@upatras.gr)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.