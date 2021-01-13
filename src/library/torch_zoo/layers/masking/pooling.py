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
        return unscaled_outputs / true_count.clamp(min=1e-8)

    def compute_mask(self, mask):
        start = self.kernel_size[0] - 1 - 2 * self.padding[0]
        return mask[:, start::self.stride[0]]


class MaskGlobalAvgPool1d(Module):

    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return inputs.mean(dim=self.dim)

        mask = mask.type_as(inputs)
        masked_inputs = apply_mask(inputs, mask, feature_dim=3 - self.dim)
        sum_inputs = masked_inputs.sum(dim=self.dim)  # shape (N, d_in)
        true_count = mask.sum(dim=1, keepdims=True)  # shape (N, 1)
        return sum_inputs / true_count.clamp(min=1e-8)
