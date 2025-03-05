"""
evaluation.py
-------------
This module provides functions for evaluating the performance and robustness of models
against adversarial attacks. It includes:

    - evaluate_model: Assess the overall performance of a model on clean and adversarial examples.
    - evaluate_class_vulnerabilities: Evaluate vulnerabilities broken down by class.
    - epsilon_sweep: Test model robustness across a range of perturbation magnitudes.
    - get_successful_adversarial_examples: Collect successful adversarial examples for visualization.
    - analyze_transferability: Assess if adversarial examples generated for one model transfer to others.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def evaluate_model(model, data_loader, device, attack_fn=None, epsilon=0.1, **attack_kwargs):
    """
    Evaluate model performance on clean and adversarial examples.
    
    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run evaluation on.
        attack_fn (function, optional): Attack function to generate adversarial examples.
        epsilon (float): Perturbation magnitude for the attack.
        attack_kwargs: Additional keyword arguments for the attack function.
        
    Returns:
        tuple: (original accuracy, adversarial accuracy)
    """
    correct = 0
    adv_correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            
            # Evaluate adversarial examples if an attack function is provided.
            if attack_fn is not None:
                try:
                    with torch.enable_grad():
                        # Handle attack function signatures (CW attack has a different signature)
                        if attack_fn.__name__ == 'cw_attack':
                            adv_data = attack_fn(model, data, target, epsilon, **attack_kwargs)
                        else:
                            adv_data = attack_fn(model, loss_fn, data, target, epsilon, **attack_kwargs)
                    
                    with torch.no_grad():
                        adv_output = model(adv_data)
                        _, adv_pred = torch.max(adv_output, 1)
                        adv_correct += (adv_pred == target).sum().item()
                except Exception as e:
                    print(f"Error during attack: {e}")
                    adv_correct = 0  # Consider all examples misclassified on error
            else:
                adv_correct = correct  # If no attack, adversarial accuracy equals original accuracy
                
            total += target.size(0)
    
    orig_acc = correct / total
    adv_acc = adv_correct / total
    
    return orig_acc, adv_acc

def evaluate_class_vulnerabilities(model, data_loader, device, attack_fn=None, epsilon=0.1, num_classes=10, **attack_kwargs):
    """
    Evaluate model vulnerability broken down by class.
    
    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run evaluation on.
        attack_fn (function, optional): Attack function to generate adversarial examples.
        epsilon (float): Perturbation magnitude for the attack.
        num_classes (int): Number of classes in the dataset.
        attack_kwargs: Additional keyword arguments for the attack function.
        
    Returns:
        dict: Dictionary containing per-class metrics including original accuracy, adversarial accuracy,
              accuracy drop, and sample count.
    """
    class_correct = {i: 0 for i in range(num_classes)}
    class_adv_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    loss_fn = nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            
            if attack_fn is not None:
                with torch.enable_grad():
                    adv_data = attack_fn(model, loss_fn, data, target, epsilon, **attack_kwargs)
                with torch.no_grad():
                    adv_output = model(adv_data)
                    _, adv_pred = torch.max(adv_output, 1)
            else:
                adv_pred = pred
            
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                class_correct[label] += (pred[i] == label).item()
                class_adv_correct[label] += (adv_pred[i] == label).item()
    
    class_metrics = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            orig_acc = class_correct[i] / class_total[i]
            adv_acc = class_adv_correct[i] / class_total[i]
            class_metrics[i] = {
                "orig_acc": orig_acc,
                "adv_acc": adv_acc,
                "drop": orig_acc - adv_acc,
                "samples": class_total[i]
            }
    return class_metrics

def epsilon_sweep(model, data_loader, device, attack_fn, epsilons=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], **attack_kwargs):
    """
    Test model vulnerability across different perturbation magnitudes.
    
    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run evaluation on.
        attack_fn (function): Attack function to generate adversarial examples.
        epsilons (list): List of perturbation magnitudes to test.
        attack_kwargs: Additional keyword arguments for the attack function.
        
    Returns:
        list: List of dictionaries with results for each epsilon value (epsilon, original accuracy,
              adversarial accuracy, and accuracy drop).
    """
    results = []
    
    for eps in tqdm(epsilons, desc="Epsilon Sweep"):
        orig_acc, adv_acc = evaluate_model(model, data_loader, device, attack_fn, eps, **attack_kwargs)
        results.append({
            "epsilon": eps,
            "orig_acc": orig_acc,
            "adv_acc": adv_acc,
            "drop": orig_acc - adv_acc
        })
    
    return results

def get_successful_adversarial_examples(model, data_loader, device, attack_fn, epsilon=0.1, num_examples=5, **attack_kwargs):
    """
    Generate and collect successful adversarial examples for visualization.
    
    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run evaluation on.
        attack_fn (function): Attack function to generate adversarial examples.
        epsilon (float): Perturbation magnitude for the attack.
        num_examples (int): Number of successful adversarial examples to collect.
        attack_kwargs: Additional keyword arguments for the attack function.
        
    Returns:
        list: List of dictionaries, each containing the original image, adversarial example,
              perturbation applied, true label, and both original and adversarial predictions.
    """
    examples = []
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output, 1)
        
        with torch.enable_grad():
            adv_data = attack_fn(model, loss_fn, data, target, epsilon, **attack_kwargs)
        
        with torch.no_grad():
            adv_output = model(adv_data)
            _, adv_pred = torch.max(adv_output, 1)
        
        for i in range(len(target)):
            if pred[i] == target[i] and adv_pred[i] != target[i]:
                examples.append({
                    "original": data[i].cpu().numpy(),
                    "adversarial": adv_data[i].cpu().detach().numpy(),
                    "perturbation": (adv_data[i] - data[i]).cpu().detach().numpy(),
                    "true_label": target[i].item(),
                    "orig_pred": pred[i].item(),
                    "adv_pred": adv_pred[i].item()
                })
                if len(examples) >= num_examples:
                    return examples
                    
    return examples

def analyze_transferability(model, secondary_models, data_loader, device, attack_fn, epsilon=0.1, num_samples=100, **attack_kwargs):
    """
    Analyze the transferability of adversarial examples between a primary model and secondary models.
    
    Parameters:
        model (torch.nn.Module): The primary model used to generate adversarial examples.
        secondary_models (list): List of secondary models to test for transferability.
        data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
        device (torch.device): Device to run evaluation on.
        attack_fn (function): Attack function to generate adversarial examples.
        epsilon (float): Perturbation magnitude for the attack.
        num_samples (int): Number of samples to test transferability on.
        attack_kwargs: Additional keyword arguments for the attack function.
        
    Returns:
        dict: Dictionary containing transferability rates for each secondary model.
    """
    loss_fn = nn.CrossEntropyLoss()
    transfer_success = {i: 0 for i in range(len(secondary_models))}
    samples_processed = 0
    
    model.eval()
    for sec_model in secondary_models:
        sec_model.eval()
    
    with torch.no_grad():
        for data, target in data_loader:
            if samples_processed >= num_samples:
                break
            
            batch_size = min(data.size(0), num_samples - samples_processed)
            data, target = data[:batch_size].to(device), target[:batch_size].to(device)
            samples_processed += batch_size
            
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
                
                # Only count successful adversarial examples
                success_mask = (adv_pred != correct_target)
                if not success_mask.any():
                    continue
                
                successful_adv = adv_data[success_mask]
                successful_target = correct_target[success_mask]
                
                for i, sec_model in enumerate(secondary_models):
                    sec_output = sec_model(successful_adv)
                    _, sec_pred = torch.max(sec_output, 1)
                    transfer_success[i] += (sec_pred != successful_target).sum().item()
    
    transferability_results = {}
    for i in range(len(secondary_models)):
        transferability_results[f"model_{i}"] = transfer_success[i] / num_samples
    
    return transferability_results
