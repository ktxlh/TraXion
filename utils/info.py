"""Information and debugging utilities for model inspection.

This module provides functions for:
- Counting model parameters
- Printing batch data for debugging
- Validating data consistency (time ordering, no round trips)
- Displaying model parameter distribution
"""

import torch.nn as nn


def count_num_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameter_distribution(model: nn.Module, max_depth: int = 3) -> None:
    """Print hierarchical distribution of model parameters.

    Displays the number and percentage of parameters in each module,
    with indentation showing the module hierarchy.

    Args:
        model: PyTorch model to analyze
        max_depth: Maximum depth of module hierarchy to display (default: 3)
    """
    total = count_num_parameters(model)
    for name, module in model.named_modules():
        if name == "":
            print(f"Total parameters: {total:,}")
            continue
        if name.count(".") + 1 > max_depth:
            continue
        if "." in name:
            print("|   " * name.count("."), end="")
        module_params = count_num_parameters(module)
        if module_params > 0:
            print(f"{name}: {module_params:,} parameters ({module_params / total:.2%})")
