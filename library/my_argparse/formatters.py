import argparse
import re
import shutil
import termcolor


class MyFormatter(argparse.ArgumentDefaultsHelpFormatter):

    def __init__(self, prog):
        super().__init__(prog, max_help_position=4, width=shutil.get_terminal_size()[0])

    # HACK Override: Change some format part to avoid [], () bracket broken
    # https://github.com/python/cpython/blob/3.7/Lib/argparse.py#L391-L486
    def _format_actions_usage(self, actions, groups):  # noqa: C901
        ########## COPY FROM SOURCE CODE ##########
        # find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                end = start + len(group._group_actions)
                if actions[start:end] == group._group_actions:
                    for action in group._group_actions:
                        group_actions.add(action)
                    if not group.required:
                        if start in inserts:
                            inserts[start] += ' ['
                        else:
                            inserts[start] = '['
                        inserts[end] = ']'
                    else:
                        if start in inserts:
                            inserts[start] += ' ('
                        else:
                            inserts[start] = '('
                        inserts[end] = ')'
                    for i in range(start + 1, end):
                        inserts[i] = '|'

        # collect all actions format strings
        parts = []
        for i, action in enumerate(actions):

            # suppressed arguments are marked with None
            # remove | separators for suppressed arguments
            if action.help is argparse.SUPPRESS:
                parts.append(None)
                if inserts.get(i) == '|':
                    inserts.pop(i)
                elif inserts.get(i + 1) == '|':
                    inserts.pop(i + 1)

            # produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # if it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == '[' and part[-1] == ']':
                        part = part[1:-1]

                # add the action string to the list
                parts.append(part)

            # produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = '%s' % option_string

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    part = '%s %s' % (option_string, args_string)

                # make it look optional if it's not required or in a group
                if not action.required and action not in group_actions:
                    part = '[%s]' % part

                # add the action string to the list
                parts.append(part)

        # insert things at the necessary indices
        for i in sorted(inserts, reverse=True):
            parts[i:i] = [inserts[i]]

        # join all the action items with spaces
        text = ' '.join([item for item in parts if item is not None])

        # clean up separators for mutually exclusive groups
        open_ = r'[\[(]'
        close = r'[\])]'
        text = re.sub(r'(%s) ' % open_, r'\1', text)
        text = re.sub(r' (%s)' % close, r'\1', text)
        text = re.sub(r'%s *%s' % (open_, close), r'', text)
        ########## END COPY FROM SOURCE CODE ##########

        # remove L482 from source code to avoid [], () bracket broken
        text = text.strip()

        # return the text
        return text

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            # custom format for choices
            result = format_choices(action.choices)
        else:
            result = default_metavar

        def format_func(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format_func

    def _get_default_metavar_for_optional(self, action):
        # add default if action.type has no '__name__' attribute
        return getattr(action.type, '__name__', repr(action.type))

    def _get_default_metavar_for_positional(self, action):
        # add default if action.type has no '__name__' attribute
        return getattr(action.type, '__name__', repr(action.type))


class MyRawTextHelpFormatter(MyFormatter):

    def _split_lines(self, text, width):
        return argparse.RawTextHelpFormatter._split_lines(self, text, width)


def format_choices(choices):
    choice_strs = [
        termcolor.colored(str(choice), color='cyan')
        for choice in choices
    ]
    return f"{{{', '.join(choice_strs)}}}"
