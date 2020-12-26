from itertools import starmap

from factories import optimizer_factory
from factories.base import get_help_of_id_kwargs
from library.my_argparse.actions import IdKwargs
from library.utils import allow_abbrev_kwargs


class ModuleFactory:

    def __init__(self, module_factory, module_name):
        self.module_factory = module_factory
        self.regularizer_table = {}
        self.module_name = module_name

    def create(self, args, meta_data):
        return self.module_factory.create(args, meta_data)

    def create_regularizers(self, args):
        @allow_abbrev_kwargs
        def create_regularizer(reg_id, params):
            return self.regularizer_table[reg_id](**params)

        regularizers = getattr(args, f'{self.module_name[0]}_regularizer')
        return list(starmap(create_regularizer, regularizers))

    def create_optimizer(self, args):
        return optimizer_factory.create(args, self.module_name)

    def add_argument_to(self, model_group, objective_group, optimizer_group):
        self.module_factory.add_argument_to(model_group)
        optimizer_factory.add_argument_to(optimizer_group, self.module_name)
        objective_group.add_argument(
            f'--{self.module_name[0]}-regularizer',
            action=IdKwargs,
            id_choices=self.regularizer_table.keys(),
            sub_action='append',
            default=[],
            split_token=',',
            metavar='REGULARIZER_ID',
            help=f"Regularizer settings.\n{get_help_of_id_kwargs(self.regularizer_table)}\n",
        )
