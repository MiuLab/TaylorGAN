from torch.nn import Module, ReLU

from .masking import MaskConv1d


class ResBlock(Module):

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = ReLU()

        self.conv1, self.conv2 = [
            MaskConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for _ in range(2)
        ]

    def forward(self, inputs, mask=None):
        x = self.conv1(inputs, mask=mask)
        x = self.conv2(x, mask=mask)
        return x + inputs
