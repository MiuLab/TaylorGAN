import torch
from torch.nn import AvgPool1d, Module

from .mask_conv import apply_mask


class MaskAvgPool1d(AvgPool1d):

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return super().forward(inputs)

        masked_inputs = apply_mask(inputs, mask)
        unscaled_outputs = super().forward(masked_inputs)
        true_count = super().forward(mask.type_as(inputs).unsqueeze(1))
        print(unscaled_outputs)
        print(true_count)
        outputs = unscaled_outputs / true_count.clamp(min=1e-8)
        return outputs, self._compute_mask(mask)

    def _compute_mask(self, mask):
        start = self.kernel_size[0] - 1 - 2 * self.padding[0]
        return mask[:, start::self.stride[0]]


class MaskGlobalAvgPool1d(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return inputs.mean(dim=self.dim)

        mask = mask.type_as(inputs)
        sum_inputs = apply_mask(inputs, mask).sum(dim=self.dim)  # shape (N, d_in)
        true_count = mask.sum(dim=1, keepdims=True)  # shape (N, 1)
        return sum_inputs / true_count.clamp(min=1e-8)
