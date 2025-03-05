# PhySecGuard

PhySecGuard is a Python library designed to evaluate adversarial vulnerabilities in cyber-physical systems. It provides a comprehensive suite of tools for testing machine learning models under adversarial conditions, including a range of adversarial attack methods, evaluation functions, visualization utilities, and detailed reporting.

## Features

- **Model Definitions:**  
  Includes neural network models such as `SimpleCNN` for quick experimentation.

- **Adversarial Attacks:**  
  Implements common attack methods including:
  - **FGSM** (Fast Gradient Sign Method)
  - **PGD** (Projected Gradient Descent)
  - **CW** (Carlini & Wagner L2 attack)

- **Evaluation Tools:**  
  Functions to evaluate model performance on both clean and adversarial examples, including per-class vulnerability analysis and epsilon sweep analysis.

- **Visualization Utilities:**  
  Tools to visualize:
  - Adversarial examples and their perturbations
  - Per-class vulnerability
  - Epsilon sweep results
  - Misclassification heatmaps

- **Reporting:**  
  Generate comprehensive vulnerability assessment reports with detailed analysis and recommendations.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/PhySecGuard.git
cd PhySecGuard
pip install -r requirements.txt
```

You can also install the library locally using the provided setup.py:

```bash
pip install .
```

## Usage Example

Below is a basic example using the MNIST dataset to evaluate a simple CNN model:

```python
from PhySecGuard.models import SimpleCNN
from PhySecGuard.runner import run_comprehensive_evaluation
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = SimpleCNN().to(device)
model.eval()

# Run comprehensive adversarial vulnerability evaluation
results = run_comprehensive_evaluation(model, test_loader, device)
```

## Project Structure

```
PhySecGuard/
│
├── PhySecGuard/
│   ├── __init__.py           # Exposes key classes/functions from submodules.
│   ├── models.py             # Contains model definitions (e.g., SimpleCNN).
│   ├── attacks.py            # Contains adversarial attack methods (FGSM, PGD, CW, etc.).
│   ├── evaluation.py         # Functions for evaluating model performance.
│   ├── visualization.py      # Visualization utilities (plots, heatmaps, etc.).
│   ├── reporting.py          # Functions to generate comprehensive reports.
│   └── runner.py             # Main evaluation pipeline.
│
├── tests/                    # Unit tests for the modules.
├── setup.py                  # Packaging script.
├── requirements.txt          # List of dependencies.
└── README.md                 # Documentation and usage instructions.
```

## Documentation

Detailed API documentation can be generated using Sphinx. See the [Sphinx documentation](https://www.sphinx-doc.org/en/master/) for more details on how to set up and generate docs for your project.

## Contributing

Contributions to PhySecGuard are welcome! Please open issues or submit pull requests via GitHub.

## License

PhySecGuard is released under the MIT License. See the LICENSE file for more details.

## Reputable Sources

- [Python Packaging Tutorial](https://packaging.python.org/tutorials/packaging-projects/)
- [Setuptools Documentation](https://setuptools.readthedocs.io/en/latest/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Sphinx Documentation](https://www.sphinx-doc.org/en/master/)
