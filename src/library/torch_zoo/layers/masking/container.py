import inspect

from torch.nn import Sequential


class MaskSequential(Sequential):

    def forward(self, inputs, mask=None):
        x = inputs
        for module in self:
            if (
                mask is not None
                and x.ndim > 2
                and 'mask' in inspect.signature(module.forward).parameters
            ):
                x = module(x, mask=mask)
            else:
                # assume module is unary operation
                x = module(x)

            if mask is not None and hasattr(module, 'compute_mask'):
                mask = module.compute_mask(mask)

        return x
