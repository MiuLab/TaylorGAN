from typing import List

from flexparse import ArgumentParser, create_action, Action


def create_factory_action(
        *args,
        type: callable,  # noqa
        help_prefix: str = '',
        default=None,
        **kwargs,
    ) -> Action:
    return create_action(
        *args,
        type=type,
        default=default,
        help=(
            help_prefix + "custom options and registry: \n" + "\n".join(type.get_helps()) + "\n"
        ),
        **kwargs,
    )


def parent_parser(
        title: str,
        description: str,
        arguments: List[Action],
        **kwargs,
    ) -> ArgumentParser:
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group(title, description=description)
    for action in arguments:
        group._add_action(action)

    return parser
