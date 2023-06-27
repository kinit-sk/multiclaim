"""
https://github.com/mosaicml/composer/blob/dev/composer/algorithms/layer_freezing/layer_freezing.py
Layer Freezing gradually makes early modules untrainable ("freezing" them), saving the cost of 
backpropagating to and updating frozen modules. The hypothesis behind Layer Freezing 
is that early layers may learn their features sooner than later layers, meaning they 
do not need to be updated later in training. Especially for fine-tuning, which is our case.
"""
import collections
from typing import List, Sequence, Tuple, Union

import torch
from torch.optim import Optimizer


def freeze_layers(
    model: torch.nn.Module,
    optimizers: Union[Optimizer, Sequence[Optimizer]],
    current_duration: float,
    freeze_start: float = 0.5,
    freeze_level: float = 1.0,
) -> Tuple[int, float]:
    """Progressively freeze the layers of the network in-place
    during training, starting with the earlier layers.
    Example:
         .. testcode::
            from composer.algorithms.layer_freezing import freeze_layers
            freeze_depth, feeze_level = freeze_layers(
                                            model=model,
                                            optimizers=optimizer,
                                            current_duration=0.5,
                                            freeze_start=0.0,
                                            freeze_level=1.0
                                        )
    Args:
        model (torch.nn.Module): The model being trained.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer]):
            The optimizers used during training.
        current_duration (float): The fraction, in ``[0, 1)`` of the training process complete.
        freeze_start (float, optional): The fraction of the training process in ``[0, 1)`` to run
            before freezing begins. Default: ``0.5``.
        freeze_level (float, optional): The maximum fraction of layers on ``[0, 1)`` to freeze.
            Default: ``1.0``.
    Return:
        (int, float): The number of layers frozen, and the percentage of the total model frozen.
    """
    # Flatten out the layers
    flat_children = []
    _get_layers(model, flat_children)
    # Determine how many layers to freeze
    freeze_percentage = _freeze_schedule(current_duration=current_duration,
                                         freeze_start=freeze_start,
                                         freeze_level=freeze_level)
    freeze_depth = int(freeze_percentage * len(flat_children[0:-1]))

    # Freeze the parameters in the chosen layers
    for i, child in enumerate(flat_children[0:-1]):
        if i < freeze_depth:
            for p in child.parameters():
                _remove_param_from_optimizers(p, optimizers)
                # Do not compute gradients for this param.
                p.requires_grad = False

    return freeze_depth, freeze_percentage


def _freeze_schedule(current_duration: float, freeze_start: float, freeze_level: float) -> float:
    """Implements a linear schedule for freezing.
    The schedule is linear and begins with no freezing and linearly
    increases the fraction of layers frozen, reaching the fraction specified by ``freeze_level`` at the end of training.
    The start of freezing is given as a fraction of the total training duration and is set with ``freeze_start``.
    Args:
        current_duration (float): The elapsed training duration.
        freeze_start (float): The fraction of training to run before freezing begins.
        freeze_level (float): The maximum fraction of levels to freeze.
    """
    # No freezing if the current epoch is less than this
    if current_duration <= freeze_start:
        return 0.0
    # `Calculate the total time for freezing to occur
    total_freezing_time = 1.0 - freeze_start
    # Calculate the amount of freezing time that has elapsed
    freezing_time_elapsed = current_duration - freeze_start
    # Calculate the fraction of the freezing time elapsed.
    freezing_time_elapsed_frac = freezing_time_elapsed / total_freezing_time
    # Scale this fraction by the amount of freezing to do.
    return freeze_level * freezing_time_elapsed_frac


def _get_layers(module: torch.nn.Module, flat_children: List[torch.nn.Module]):
    """Helper function to get all submodules.
    Does a depth first search to flatten out modules which
    contain parameters.
    Args:
        module (torch.nn.Module): Current module to search.
        flat_children (List[torch.nn.Module]): List containing modules.
    """
    # Check if given module has no children and parameters.
    if (len(list(module.children())) == 0 and len(list(module.parameters())) > 0):
        flat_children.append(module)
    else:
        # Otherwise, continue the search over its children.
        for child in module.children():
            _get_layers(child, flat_children)


def _remove_param_from_optimizers(p: torch.nn.Parameter, optimizers: Union[Optimizer, Sequence[Optimizer]]):
    """Helper function to freeze the training of a parameter.
    To freeze a parameter, it must be removed from the optimizer,
    otherwise momentum and weight decay may still be applied.
    Args:
        p (torch.nn.Parameter): The parameter being frozen.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer]): The optimizers used during training.
    """
    # Search over params in the optimizers to find and remove the
    # given param. Necessary due to the way params are stored.
    for optimizer in ensure_tuple(optimizers):
        for group in optimizer.param_groups:
            group['params'] = list(filter(lambda x: id(x) != id(p), group['params']))


def ensure_tuple(x):
    """Converts ``x`` into a tuple.
    * If ``x`` is ``None``, then ``tuple()`` is returned.
    * If ``x`` is a tuple, then ``x`` is returned as-is.
    * If ``x`` is a list, then ``tuple(x)`` is returned.
    * If ``x`` is a dict, then ``tuple(v for v in x.values())`` is returned.
    Otherwise, a single element tuple of ``(x,)`` is returned.
    Args:
        x (Any): The input to convert into a tuple.
    Returns:
        tuple: A tuple of ``x``.
    """
    if x is None:
        return ()
    if isinstance(x, (str, bytes, bytearray)):
        return (x,)
    if isinstance(x, collections.abc.Sequence):
        return tuple(x)
    if isinstance(x, dict):
        return tuple(x.values())
    return (x,)