"""
visualization.py
----------------
This module provides visualization utilities for PhySecGuard.
It includes functions to:
  - Visualize successful adversarial examples and their perturbations.
  - Plot per-class vulnerability analysis.
  - Plot epsilon sweep results.
  - Generate a misclassification heatmap of adversarial examples.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

def visualize_adversarial_examples(examples, save_path=None):
    """
    Visualize successful adversarial examples and their perturbations.
    
    Parameters:
        examples (list): List of dictionaries containing keys:
                         "original", "adversarial", "perturbation",
                         "true_label", "orig_pred", "adv_pred".
        save_path (str, optional): File path to save the visualization.
    """
    if not examples:
        print("No successful adversarial examples found.")
        return

    num_examples = len(examples)
    plt.figure(figsize=(15, 5 * num_examples))
    
    for i, example in enumerate(examples):
        # Plot original image
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(example["original"].reshape(28, 28), cmap='gray')
        plt.title(f"Original: {example['true_label']}\nPredicted: {example['orig_pred']}")
        plt.axis('off')
        
        # Plot adversarial image
        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(example["adversarial"].reshape(28, 28), cmap='gray')
        plt.title(f"Adversarial\nPredicted: {example['adv_pred']}")
        plt.axis('off')
        
        # Plot perturbation (scaled for visibility)
        plt.subplot(num_examples, 3, i * 3 + 3)
        perturbation = example["perturbation"].reshape(28, 28)
        plt.imshow(perturbation, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Perturbation\n(scaled for visibility)")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def plot_class_vulnerability(class_metrics, save_path=None):
    """
    Plot per-class vulnerability analysis.
    
    Parameters:
        class_metrics (dict): Dictionary containing per-class metrics:
                              "orig_acc", "adv_acc", "drop", and "samples".
        save_path (str, optional): File path to save the plot.
    """
    class_labels = sorted(class_metrics.keys())
    orig_acc = [class_metrics[c]["orig_acc"] for c in class_labels]
    adv_acc = [class_metrics[c]["adv_acc"] for c in class_labels]
    acc_drop = [class_metrics[c]["drop"] for c in class_labels]
    
    plt.figure(figsize=(14, 8))
    
    # Bar plot for accuracy drop by class
    plt.subplot(2, 1, 1)
    bars = plt.bar(class_labels, acc_drop, color='salmon')
    plt.axhline(y=np.mean(acc_drop), color='r', linestyle='--', label=f"Mean drop ({np.mean(acc_drop):.2f})")
    plt.xlabel('Class')
    plt.ylabel('Accuracy Drop')
    plt.title('Vulnerability by Class (Accuracy Drop)')
    plt.legend()
    
    # Annotate bars with drop percentage
    for bar, drop in zip(bars, acc_drop):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{drop:.2f}", ha='center', va='bottom')
    
    # Comparison of original vs adversarial accuracy by class
    plt.subplot(2, 1, 2)
    x = np.arange(len(class_labels))
    width = 0.35
    plt.bar(x - width/2, orig_acc, width, label='Original', color='royalblue')
    plt.bar(x + width/2, adv_acc, width, label='Adversarial', color='lightcoral')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Original vs Adversarial Accuracy by Class')
    plt.xticks(x, class_labels)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Class vulnerability plot saved to {save_path}")
    
    plt.show()

def plot_epsilon_sweep(epsilon_results, save_path=None):
    """
    Plot model vulnerability across different perturbation magnitudes.
    
    Parameters:
        epsilon_results (list): List of dictionaries containing:
                                "epsilon", "orig_acc", "adv_acc", "drop".
        save_path (str, optional): File path to save the plot.
    """
    epsilons = [result["epsilon"] for result in epsilon_results]
    orig_acc = [result["orig_acc"] for result in epsilon_results]
    adv_acc = [result["adv_acc"] for result in epsilon_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, orig_acc, 'o-', label='Original Accuracy', color='royalblue')
    plt.plot(epsilons, adv_acc, 'o-', label='Adversarial Accuracy', color='lightcoral')
    plt.fill_between(epsilons, orig_acc, adv_acc, color='salmon', alpha=0.3, label='Accuracy Drop')
    
    plt.xlabel('Perturbation Magnitude (ε)')
    plt.ylabel('Accuracy')
    plt.title('Model Vulnerability vs. Perturbation Magnitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Annotate epsilon values with adversarial accuracy
    for i, eps in enumerate(epsilons):
        plt.annotate(f"{adv_acc[i]:.2f}",
                     (eps, adv_acc[i]),
                     textcoords="offset points",
                     xytext=(0, -15),
                     ha='center',
                     fontsize=8)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Epsilon sweep plot saved to {save_path}")
    
    plt.show()

def plot_misclassification_heatmap(model, data_loader, device, attack_fn, epsilon=0.1, num_classes=10, save_path=None, **attack_kwargs):
    """
    Create a confusion matrix heatmap of adversarial misclassifications.
    
    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run evaluation on.
        attack_fn (function): Attack function to generate adversarial examples.
        epsilon (float): Perturbation magnitude for the attack.
        num_classes (int): Number of classes in the dataset.
        save_path (str, optional): File path to save the heatmap.
        attack_kwargs: Additional keyword arguments for the attack function.
    
    Returns:
        np.ndarray: The confusion matrix.
    """
    # Initialize confusion matrix (rows: true labels, columns: predicted labels)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    
    model.eval()
    for data, target in tqdm(data_loader, desc="Generating confusion matrix"):
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output, 1)
        
        # Only consider correctly classified examples
        correct_mask = (pred == target)
        if not correct_mask.any():
            continue
        
        correct_data = data[correct_mask]
        correct_target = target[correct_mask]
        
        with torch.enable_grad():
            adv_data = attack_fn(model, loss_fn, correct_data, correct_target, epsilon, **attack_kwargs)
        
        with torch.no_grad():
            adv_output = model(adv_data)
            _, adv_pred = torch.max(adv_output, 1)
        
        # Update confusion matrix
        for true_label, adv_label in zip(correct_target.cpu().numpy(), adv_pred.cpu().numpy()):
            confusion_matrix[true_label, adv_label] += 1
    
    # Normalize the confusion matrix by row (true label)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    norm_confusion_matrix = confusion_matrix / row_sums
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(norm_confusion_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Label (After Attack)')
    plt.ylabel('True Label')
    plt.title('Adversarial Misclassification Heatmap')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Misclassification heatmap saved to {save_path}")
    
    plt.show()
    
    return confusion_matrix
