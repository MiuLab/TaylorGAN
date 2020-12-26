from library.utils import FormatableMixin


class Callback(FormatableMixin):

    def on_train_begin(self, is_restored: bool):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_batch_begin(self, batch: int):
        pass

    def on_batch_end(self, batch: int, batch_data):
        pass

    def on_epoch_end(self, epoch: int):
        pass

    def on_train_end(self):
        pass

    def get_config(self):
        return {}


NullCallback = Callback
