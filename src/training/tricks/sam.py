"""
https://github.com/mosaicml/composer/blob/dev/composer/algorithms/sam/sam.py
Sharpness-Aware Minimization (SAM) is an optimization algorithm that minimizes 
both the loss and the sharpness of the loss. It finds parameters that lie in 
a neighborhood of low loss. The authors find that this improves model generalization
"""
import torch


class SAM(torch.optim.Optimizer):
    """Wraps an optimizer with sharpness-aware minimization (`Foret et al, 2020 <https://arxiv.org/abs/2010.01412>`_).
    See :class:`.SAM` for details.
    Implementation based on https://github.com/davda54/sam
    Args:
        base_optimizer (torch.optim.Optimizer) The optimizer to apply SAM to.
        rho (float, optional): The SAM neighborhood size. Must be greater than 0. Default: ``0.05``.
        epsilon (float, optional): A small value added to the gradient norm for numerical stability. Default: ``1.0e-12``.
        interval (int, optional): SAM will run once per ``interval`` steps. A value of 1 will
            cause SAM to run every step. Steps on which SAM runs take
            roughly twice as much time to complete. Default: ``1``.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        epsilon: float = 1.0e-12,
        **kwargs
    ):
        if rho < 0:
            raise ValueError(f'Invalid rho, should be non-negative: {rho}')
        self.base_optimizer = base_optimizer
        defaults = {'rho': rho, 'epsilon': epsilon, **kwargs}
        super(SAM, self).__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + group['epsilon'])
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                p.sub_(self.state[p]['e_w'])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    @torch.no_grad()
    def step(self, step_type: str):
        """ Adjusted to PyTorch mixed precision training framework
        """
        if step_type == 'first':
            self.first_step()
        elif step_type == 'second':
            self.second_step()
        elif step_type == 'skip':
            self.base_optimizer.step()


    def _grad_norm(self):
        norm = torch.norm(torch.stack(
            [p.grad.norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None]),
                          p='fro')
        return norm