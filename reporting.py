"""
reporting.py
------------
This module provides reporting utilities for PhySecGuard.
It contains functions to generate comprehensive vulnerability assessment reports
that summarize model evaluation results, attack performance, epsilon sweep analyses,
and per-class vulnerabilities.
"""

import os
import json
import numpy as np
from datetime import datetime

def generate_comprehensive_report(model, data_loader, device, results, report_dir="reports"):
    """
    Generate a comprehensive vulnerability assessment report.

    Parameters:
        model (torch.nn.Module): The model being evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device used for evaluation.
        results (dict): Dictionary containing evaluation results.
        report_dir (str, optional): Directory in which to save the report.

    Returns:
        str: The comprehensive report content as a string.
    """
    # Create report directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"vulnerability_report_{timestamp}.txt")

    report = "=" * 80 + "\n"
    report += "COMPREHENSIVE AI MODEL VULNERABILITY ASSESSMENT REPORT\n"
    report += "=" * 80 + "\n\n"
    report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # 1. Model Information
    report += "1. MODEL INFORMATION\n"
    report += "-" * 40 + "\n"
    report += f"Model Architecture: {model.__class__.__name__}\n"
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report += f"Total Parameters: {total_params:,}\n"
    report += f"Trainable Parameters: {trainable_params:,}\n\n"

    # 2. Vulnerability to Different Attack Methods
    report += "2. VULNERABILITY TO DIFFERENT ATTACK METHODS\n"
    report += "-" * 40 + "\n"
    for attack_name, attack_results in results.get("attack_methods", {}).items():
        orig_acc = attack_results.get("orig_acc", 0)
        adv_acc = attack_results.get("adv_acc", 0)
        drop = attack_results.get("drop", 0)
        report += f"Attack: {attack_name}\n"
        report += f"  Original Accuracy: {orig_acc*100:.2f}%\n"
        report += f"  Adversarial Accuracy: {adv_acc*100:.2f}%\n"
        report += f"  Accuracy Drop: {drop*100:.2f}%\n"
        if drop > 0.5:
            vuln_level = "HIGHLY VULNERABLE"
        elif drop > 0.2:
            vuln_level = "MODERATELY VULNERABLE"
        else:
            vuln_level = "RELATIVELY ROBUST"
        report += f"  Assessment: {vuln_level}\n\n"

    # 3. Epsilon Sweep Analysis
    report += "3. EPSILON SWEEP ANALYSIS\n"
    report += "-" * 40 + "\n"
    strongest_attack = max(results.get("attack_methods", {}).items(), key=lambda x: x[1].get("drop", 0))[0]
    eps_results = results.get("epsilon_sweep", {}).get(strongest_attack, [])
    report += "Epsilon | Original Acc | Adversarial Acc | Accuracy Drop\n"
    report += "-" * 60 + "\n"
    for res in eps_results:
        report += f"{res['epsilon']:.2f} | {res['orig_acc']*100:.2f}% | {res['adv_acc']*100:.2f}% | {res['drop']*100:.2f}%\n"
    report += "\n"
    threshold_eps = next((r["epsilon"] for r in eps_results if r["drop"] > 0.2), None)
    if threshold_eps:
        report += f"Vulnerability Threshold: ε = {threshold_eps:.2f} (accuracy drop >20%)\n\n"
    else:
        report += "Vulnerability Threshold: Model maintains >80% original accuracy across tested epsilons\n\n"

    # 4. Per-Class Vulnerability Analysis
    report += "4. PER-CLASS VULNERABILITY ANALYSIS\n"
    report += "-" * 40 + "\n"
    class_metrics = results.get("class_vulnerabilities", {}).get(strongest_attack, {})
    if class_metrics:
        most_vuln = max(class_metrics.items(), key=lambda x: x[1].get("drop", 0))
        least_vuln = min(class_metrics.items(), key=lambda x: x[1].get("drop", 0))
        report += f"Most vulnerable class: {most_vuln[0]} (Accuracy drop: {most_vuln[1]['drop']*100:.2f}%)\n"
        report += f"Least vulnerable class: {least_vuln[0]} (Accuracy drop: {least_vuln[1]['drop']*100:.2f}%)\n\n"
        report += "Class | Original Acc | Adversarial Acc | Accuracy Drop | Samples\n"
        report += "-" * 65 + "\n"
        for class_idx, metrics in sorted(class_metrics.items()):
            report += f"{class_idx} | {metrics['orig_acc']*100:.2f}% | {metrics['adv_acc']*100:.2f}% | {metrics['drop']*100:.2f}% | {metrics['samples']}\n"
    else:
        report += "No per-class vulnerability data available.\n"
    report += "\n"

    # 5. Recommendations
    report += "5. RECOMMENDATIONS\n"
    report += "-" * 40 + "\n"
    max_drop = max([v.get("drop", 0) for v in results.get("attack_methods", {}).values()] or [0])
    if max_drop > 0.7:
        report += "- CRITICAL: Consider adversarial training for highly vulnerable classes.\n"
    if threshold_eps is not None and threshold_eps <= 0.05:
        report += "- HIGH PRIORITY: Model is vulnerable even to small perturbations (ε ≤ 0.05).\n"
    report += "- Recommended measures:\n"
    report += "  * Increase training data diversity for vulnerable classes.\n"
    report += "  * Implement input preprocessing defenses (e.g., feature squeezing, JPEG compression).\n"
    report += "  * Consider randomized smoothing and adversarial training (e.g., with PGD examples).\n\n"

    with open(report_path, "w") as f:
        f.write(report)
    print(f"Comprehensive report saved to {report_path}")
    return report
