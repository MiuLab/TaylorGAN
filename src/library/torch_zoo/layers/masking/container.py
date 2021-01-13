import inspect

from torch.nn import Sequential


class MaskSequential(Sequential):

    def forward(self, inputs, mask=None):
        for module in self:
            if mask is None or inputs.ndim == 2:
                outputs = module(inputs)
            elif 'mask' in inspect.signature(module.forward).parameters:
                outputs = module(inputs, mask=mask)
                if isinstance(outputs, (list, tuple)):
                    outputs, mask = outputs
                else:
                    outputs, mask = outputs, None
            else:
                # assume module is unary operation
                outputs = module(inputs)

            inputs = outputs

        return outputs
