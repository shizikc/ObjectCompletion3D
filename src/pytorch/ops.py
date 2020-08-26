import torch

class ModuleWithLoss(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(ModuleWithLoss, self).__init__(*args, **kwargs)
        self.loss = None


def get_module_losses(m):
    return {x: x.loss for x in m.modules() if isinstance(x, ModuleWithLoss)}

