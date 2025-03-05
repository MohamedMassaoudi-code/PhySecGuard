"""
PhySecGuard Library
-------------------
PhySecGuard is a Python library designed for evaluating adversarial vulnerabilities in cyber-physical systems.
It provides model definitions, adversarial attack implementations, evaluation tools, visualization utilities, and reporting capabilities.

Modules:
    - models: Contains model definitions (e.g., SimpleCNN).
    - attacks: Contains adversarial attack methods (e.g., FGSM, PGD, CW).
    - evaluation: Contains functions for evaluating model performance.
    - visualization: Contains utilities for plotting and generating visual insights.
    - reporting: Contains functions to generate comprehensive evaluation reports.
    - runner: Contains the main evaluation pipeline.

Reputable Sources:
    - Python Packaging Tutorial: https://packaging.python.org/tutorials/packaging-projects/
    - Setuptools Documentation: https://setuptools.pypa.io/en/latest/
    - PyTorch Documentation: https://pytorch.org/docs/stable/index.html
"""

from .models import SimpleCNN
from .attacks import fgsm_attack, pgd_attack, cw_attack
from .evaluation import (
    evaluate_model,
    evaluate_class_vulnerabilities,
    epsilon_sweep,
    get_successful_adversarial_examples,
    analyze_transferability,
)
from .visualization import (
    visualize_adversarial_examples,
    plot_class_vulnerability,
    plot_epsilon_sweep,
    plot_misclassification_heatmap,
)
from .reporting import generate_comprehensive_report
from .runner import run_comprehensive_evaluation

__all__ = [
    "SimpleCNN",
    "fgsm_attack",
    "pgd_attack",
    "cw_attack",
    "evaluate_model",
    "evaluate_class_vulnerabilities",
    "epsilon_sweep",
    "get_successful_adversarial_examples",
    "analyze_transferability",
    "visualize_adversarial_examples",
    "plot_class_vulnerability",
    "plot_epsilon_sweep",
    "plot_misclassification_heatmap",
    "generate_comprehensive_report",
    "run_comprehensive_evaluation",
]
