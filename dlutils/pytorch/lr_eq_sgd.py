import math
import torch
from torch.optim.optimizer import Optimizer


class LREQSGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
        super(LREQSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LREQSGD, self).__setstate__(state)

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state.clear()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(1.0, grad)
                    grad = buf

                step_size = group['lr']

                if hasattr(p, 'lr_equalization_coef'):
                    step_size *= p.lr_equalization_coef
                p.data.add_(-step_size, grad)
        return loss
