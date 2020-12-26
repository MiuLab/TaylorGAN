from library.utils import format_object, logging_indent, ObjectWrapper, extract_wrapped


class OptimizerWrapper(ObjectWrapper):

    '''Just for late printing'''

    def __init__(self, optimizer, wrapper_info=(), **params):
        super().__init__(optimizer)
        self._params = params
        self._wrapper_info = wrapper_info

    def summary(self):
        optimizer = extract_wrapped(self._body, attr_name='optimizer')
        with logging_indent(f"Optimizer: {format_object(optimizer, **self._params)}"):
            for wrapper in self._wrapper_info:
                print(wrapper)
