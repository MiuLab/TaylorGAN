from typing import List

from flexparse import ArgumentParser, create_action, Action, LookUpCall


def create_factory_action(
        *args,
        registry: dict,
        help_prefix: str = '',
        default=None,
        set_info: bool = False,
        **kwargs,
    ) -> Action:
    type_ = LookUpCall(registry, set_info=set_info, match_abbrev=False)
    return create_action(
        *args,
        type=type_,
        default=default,
        help=(
            help_prefix + "custom options and registry: \n" + "\n".join(type_.get_helps()) + "\n"
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
