import argparse
import ast

from .formatters import format_choices


class IdKwargs(argparse.Action):

    class IdKwargsPair:

        def __init__(self, id, **kwargs):  # noqa: A002
            self.id = id  # noqa: A002
            self.kwargs = kwargs

        def __iter__(self):
            return iter((self.id, self.kwargs))

        def __str__(self):
            if not self.kwargs:
                return self.id
            kwarg_string = ",".join(f"{k}={v}" for k, v in self.kwargs.items())
            return f"{self.id} {kwarg_string}"

        def __eq__(self, other):
            return (self.id, self.kwargs) == other

    def __init__(
            self,
            id_choices,
            split_token=',',
            use_bool_abbreviation=True,
            default_as_string=True,
            sub_action='store',
            metavar=None,
            **kwargs,
        ):
        super().__init__(nargs='+', **kwargs)
        self.id_choices = id_choices
        self.split_token = split_token
        self.default_as_string = default_as_string
        self.use_bool_abbreviation = use_bool_abbreviation
        if sub_action not in ('store', 'append'):
            raise ValueError(f'unknown sub_action {sub_action:r}')
        self.sub_action = sub_action
        if metavar is None:
            metavar = (format_choices(id_choices), f'k1=v1{split_token}k2=v2')
        elif isinstance(metavar, str):
            metavar = (metavar, f'k1=v1{split_token}k2=v2')
        self.metavar = metavar

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 2:
            id_, kwarg_string = values
            try:
                kwargs = self._process_kwargs_string(kwarg_string)
            except ValueError:
                raise argparse.ArgumentError(
                    self,
                    f"invalid kwargs: {kwarg_string!r}",
                )
            except NameError:
                raise argparse.ArgumentError(
                    self,
                    "value should be built-in types.",
                )
        elif len(values) == 1:
            id_, kwargs = values[0], {}
        else:
            raise argparse.ArgumentError(
                self,
                'expected at most 2 argument',
            )

        if id_ not in self.id_choices:
            raise argparse.ArgumentError(
                self,
                f"invalid choice: '{id_}' "
                f"(choose from {', '.join(map(repr, self.id_choices))})",
            )

        if self.sub_action == 'store':
            setattr(namespace, self.dest, self.IdKwargsPair(id_, **kwargs))
        elif self.sub_action == 'append':
            if not getattr(namespace, self.dest, None):
                setattr(namespace, self.dest, [])
            getattr(namespace, self.dest).append(self.IdKwargsPair(id_, **kwargs))
        else:
            raise AssertionError

    def _process_kwargs_string(self, kwarg_string):
        kwargs = {}
        kvs = kwarg_string.split(self.split_token)
        for kv in kvs:
            if '=' in kv:
                key, val = kv.split('=')
                try:
                    val = ast.literal_eval(val)
                except (SyntaxError, ValueError):
                    if not self.default_as_string:
                        raise
                    # default as string if can't eval
            elif self.use_bool_abbreviation:
                key, val = kv, True
            else:
                raise ValueError

            if key in kwargs:
                raise ValueError
            kwargs[key] = val

        return kwargs
