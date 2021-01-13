from torch.nn import Module


class LambdaModule(Module):

    def __init__(self, func):
        super().__init__()
        self.forward = func
