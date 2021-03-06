import warnings

warnings.simplefilter('ignore', category=FutureWarning)

import tensorflow as tf

from library.utils import logging_indent
from core.train import DataLoader
from factories import callback_factory, data_factory, generator_factory, trainer_factory
from scripts.snippets import (
    get_tf_config_proto,
    set_global_random_seed,
    set_package_verbosity,
)


def main(args, base_tag=None, checkpoint=None):
    set_package_verbosity(args.debug)

    with logging_indent("Set global random seed"):
        set_global_random_seed(args.random_seed)

    with logging_indent("Preprocess data"):
        data_collection, meta_data = data_factory.preprocess(args, return_meta=True)
        data_collection.summary()
        meta_data.summary()

    with logging_indent("Prepare Generator"):
        generator = generator_factory.create(args, meta_data)

    with logging_indent("Prepare Generator Trainer"):
        trainer = trainer_factory.create(args, meta_data, generator)
        trainer.summary()

    with logging_indent("Prepare Callback"):
        data_loader = DataLoader(
            data_collection.train,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
        )
        data_loader.callback = callback_factory.create(
            args,
            trainer=trainer,
            generator=generator,
            data_collection=data_collection,
            meta_data=meta_data,
            base_tag=base_tag,
        )

    with tf.Session(config=get_tf_config_proto(args.jit)) as sess:
        if checkpoint:
            print(f"Restore from checkpoint: {checkpoint}")
            tf.train.Saver().restore(sess, save_path=checkpoint)
            data_loader.skip_epochs(int(checkpoint.split('-')[-1]))
        else:
            tf.global_variables_initializer().run()

        trainer.fit(data_loader)


def parse_args(argv, algorithm):
    from flexparse import ArgumentParser
    from flexparse.formatters import RawTextHelpFormatter
    from scripts.parsers import (
        train_parser,
        evaluate_parser,
        save_parser,
        logging_parser,
        backend_parser,
        develop_parser,
    )

    parser = ArgumentParser(
        description='TextGAN.',
        formatter_class=RawTextHelpFormatter,
        fromfile_prefix_chars='@',
        parents=[
            data_factory.PARSER,
            trainer_factory.create_parser(algorithm),
            train_parser(),
            evaluate_parser(),
            save_parser(),
            logging_parser(),
            backend_parser(),
            develop_parser(),
        ],
    )
    return parser.parse_args(argv)
