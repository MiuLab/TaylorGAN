import torch
from torch.nn import Conv1d


class MaskConv1d(Conv1d):

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return super().forward(inputs)

        masked_inputs = apply_mask(inputs, mask)
        return super().forward(masked_inputs)

    def compute_mask(self, mask):
        start = self.dilation[0] * (self.kernel_size[0] - 1) - 2 * self.padding[0]
        return mask[:, start::self.stride[0]]


def apply_mask(inputs: torch.Tensor, mask: torch.Tensor, feature_dim: int = 1) -> torch.Tensor:
    return inputs * mask.type_as(inputs).unsqueeze(feature_dim)
