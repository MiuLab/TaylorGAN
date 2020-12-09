import abc

from library.my_argparse import MyArgumentParser
from library.utils import logging_block

from core.models.sequence_modeling import TokenSequenceWithPlaceholder
from core.train import GeneratorUpdater, DiscriminatorUpdater, GANTrainer
from factories.modules import discriminator_factory, generator_factory

from .objectives import estimator_factory, loss_factory


class TrainerCreator(abc.ABC):

    def __init__(self, args, meta_data, generator):
        self.args = args
        self.meta_data = meta_data
        self.generator = generator

    @abc.abstractmethod
    def create(self):
        pass

    def create_real_samples(self):
        return TokenSequenceWithPlaceholder(
            batch_size=self.args.batch_size,
            maxlen=self.meta_data.maxlen,
            eos_idx=self.meta_data.special_token_config.eos.idx,
        )

    def create_generator_updater(self, real_samples, objective):
        with logging_block("Generator Optimization:"):
            optimizer = generator_factory.create_optimizer(self.args)
            objectives = [objective, *generator_factory.create_regularizers(self.args)]
            with logging_block("Objective:"):
                for obj in objectives:
                    print(obj)

        updater = GeneratorUpdater(self.generator, optimizer=optimizer)
        for obj in objectives:
            updater.add_loss(obj)
        updater.build_graph(real_samples)
        return updater


class GANTrainerCreator(TrainerCreator):

    def create(self):
        real_samples = self.create_real_samples()
        discriminator = discriminator_factory.create(self.args, self.meta_data)
        gan_loss_tuple = loss_factory.create(self.args)
        objective = self.create_objective(discriminator, gan_loss_tuple.generator_loss)
        return GANTrainer(
            data_source=real_samples,
            generator_updater=self.create_generator_updater(real_samples, objective),
            objective_updater=self.create_objective_updater(
                real_samples, discriminator, gan_loss_tuple.discriminator_loss,
            ),
            d_steps=self.args.d_steps,
        )

    def create_objective(self, discriminator, generator_loss):
        return estimator_factory.create(
            self.args,
            discriminator=discriminator,
            generator_loss=generator_loss,
        )

    def create_objective_updater(self, real_samples, discriminator, loss):
        with logging_block("Discriminator Optimization:"):
            optimizer = discriminator_factory.create_optimizer(self.args)
            objectives = [loss, *discriminator_factory.create_regularizers(self.args)]
            with logging_block("Objective:"):
                for obj in objectives:
                    print(obj)

        updater = DiscriminatorUpdater(discriminator, optimizer=optimizer)
        for obj in objectives:
            updater.add_loss(obj)
        updater.build_graph(
            real_samples=real_samples,
            fake_samples=self.generator.generate(real_samples.batch_size, real_samples.maxlen),
        )
        return updater

    @classmethod
    def create_parser(cls):
        parser = MyArgumentParser(add_help=False)
        model_group = parser.add_argument_group(
            'model',
            description="Model's structure & hyperparameters.",
        )
        objective_group = parser.add_argument_group('objectives', description="model's objective.")
        loss_factory.add_argument_to(objective_group)
        estimator_factory.add_argument_to(objective_group)
        objective_group.add_argument(
            '--d-steps',
            type=int,
            default=1,
            help='Update generator every n discriminator iters.',
        )
        optimizer_group = parser.add_argument_group(
            'optimizer', description="optimizer's settings.",
        )
        generator_factory.add_argument_to(model_group, objective_group, optimizer_group)
        discriminator_factory.add_argument_to(model_group, objective_group, optimizer_group)
        return parser
