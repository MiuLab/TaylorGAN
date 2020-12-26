from typing import List

from flexparse import ArgumentParser, create_action, Action, FactoryMethod


def create_factory_action(
        *args,
        registry: dict,
        help_prefix: str = '',
        default=None,
        return_info: bool = False,
        **kwargs,
    ) -> Action:
    factory = FactoryMethod(registry, return_info=return_info)
    return create_action(
        *args,
        type=factory,
        default=default,
        help=help_prefix + factory.get_registry_help(),
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
