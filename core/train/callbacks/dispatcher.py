from .base import Callback


class DispatchCallback(Callback):

    def __init__(self):
        self.on_epoch_begin = CallDispatcher()
        self.on_batch_begin = CallDispatcher()
        self.on_batch_end = CallDispatcher()
        self.on_epoch_end = CallDispatcher()


class CallDispatcher:

    def __init__(self):
        self.func_list = []

    def attach(self, func: callable, period: int = 1):
        self.func_list.append((func, period))

    def __call__(self, step, *args, **kwargs):
        for func, period in self.func_list:
            if step % period == 0:
                func(step, *args, **kwargs)
