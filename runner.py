"""
runner.py
---------
This module provides the main evaluation pipeline for PhySecGuard.
It sets up the environment, loads the dataset, initializes the model,
and runs a comprehensive adversarial vulnerability evaluation. Results,
visualizations, and reports are saved for further analysis.
"""

import os
from datetime import datetime
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import functions and classes from PhySecGuard submodules
from PhySecGuard.models import SimpleCNN
from PhySecGuard.attacks import fgsm_attack, pgd_attack, cw_attack
from PhySecGuard.evaluation import (
    evaluate_model,
    evaluate_class_vulnerabilities,
    epsilon_sweep,
    get_successful_adversarial_examples,
    analyze_transferability,
)
from PhySecGuard.visualization import (
    visualize_adversarial_examples,
    plot_class_vulnerability,
    plot_epsilon_sweep,
    plot_misclassification_heatmap,
)
from PhySecGuard.reporting import generate_comprehensive_report

def run_comprehensive_evaluation(model, data_loader, device, save_dir="results"):
    """
    Run a comprehensive adversarial vulnerability evaluation on a model.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run evaluation on.
        save_dir (str, optional): Directory to save evaluation results.

    Returns:
        dict: Dictionary containing all evaluation results.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(save_dir, f"eval_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "attack_methods": {},
        "epsilon_sweep": {},
        "class_vulnerabilities": {}
    }

    # Define attack methods to evaluate
    attack_methods = {
        "FGSM": fgsm_attack,
        "PGD-10": lambda model, loss_fn, data, target, epsilon: pgd_attack(model, loss_fn, data, target, epsilon, iterations=10),
        "PGD-40": lambda model, loss_fn, data, target, epsilon: pgd_attack(model, loss_fn, data, target, epsilon, iterations=40),
        "CW": cw_attack  # CW attack has a different signature but is included for evaluation
    }

    print("Evaluating vulnerability to different attack methods...")
    for attack_name, attack_fn in attack_methods.items():
        print(f"Testing {attack_name}...")
        orig_acc, adv_acc = evaluate_model(model, data_loader, device, attack_fn, epsilon=0.1)
        drop = orig_acc - adv_acc
        results["attack_methods"][attack_name] = {
            "orig_acc": orig_acc,
            "adv_acc": adv_acc,
            "drop": drop
        }

    print("Performing epsilon sweep analysis...")
    epsilons = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
    for attack_name, attack_fn in attack_methods.items():
        print(f"Epsilon sweep for {attack_name}...")
        eps_results = epsilon_sweep(model, data_loader, device, attack_fn, epsilons)
        results["epsilon_sweep"][attack_name] = eps_results
        plot_epsilon_sweep(eps_results, save_path=os.path.join(results_dir, f"epsilon_sweep_{attack_name}.png"))

    print("Analyzing per-class vulnerabilities...")
    for attack_name, attack_fn in attack_methods.items():
        print(f"Class analysis for {attack_name}...")
        class_metrics = evaluate_class_vulnerabilities(model, data_loader, device, attack_fn, epsilon=0.1)
        results["class_vulnerabilities"][attack_name] = class_metrics
        plot_class_vulnerability(class_metrics, save_path=os.path.join(results_dir, f"class_vuln_{attack_name}.png"))

    print("Generating adversarial examples for visualization...")
    strongest_attack = max(results["attack_methods"].items(), key=lambda x: x[1]["drop"])[0]
    adv_examples = get_successful_adversarial_examples(model, data_loader, device, attack_methods[strongest_attack], epsilon=0.1, num_examples=5)
    if adv_examples:
        visualize_adversarial_examples(adv_examples, save_path=os.path.join(results_dir, "adversarial_examples.png"))
        results["adversarial_examples"] = [
            {k: v.tolist() if hasattr(v, "tolist") else v for k, v in ex.items()}
            for ex in adv_examples
        ]

    print("Generating misclassification heatmap...")
    confusion_matrix = plot_misclassification_heatmap(
        model, data_loader, device,
        attack_methods[strongest_attack],
        epsilon=0.1, num_classes=10,
        save_path=os.path.join(results_dir, "misclassification_heatmap.png")
    )
    results["confusion_matrix"] = confusion_matrix.tolist()

    print("Generating comprehensive report...")
    report = generate_comprehensive_report(model, data_loader, device, results, report_dir=results_dir)

    # Save overall evaluation results as JSON
    with open(os.path.join(results_dir, "evaluation_results.json"), "w") as f:
        import json
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    print(f"Evaluation complete! Results saved to {results_dir}")
    return results

def main():
    # Setup device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (e.g., MNIST)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model and move it to the selected device
    model = SimpleCNN().to(device)
    model.eval()

    # Run comprehensive adversarial vulnerability evaluation
    results = run_comprehensive_evaluation(model, test_loader, device)
    print("\nEvaluation Summary:")
    for attack_name, metrics in results["attack_methods"].items():
        print(f"{attack_name}: Original Acc: {metrics['orig_acc']*100:.2f}% | Adversarial Acc: {metrics['adv_acc']*100:.2f}% | Drop: {metrics['drop']*100:.2f}%")
    print("Detailed results and visualizations are available in the results directory.")

if __name__ == '__main__':
    main()
