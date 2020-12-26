import time
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

import tensorflow as tf

from core.train import DataGenerator
from library.utils import logging_block
from factories import CallbackFactory, data_factory, trainer_factory, generator_factory
from scripts.snippets import (
    get_tf_config_proto,
    set_global_random_seed,
    set_package_verbosity,
)


def main(argv, base_tag=None, checkpoint=None, override_namespace=None):
    args = parse_args(argv)
    if override_namespace:
        args.__dict__.update(override_namespace.__dict__)

    set_package_verbosity(args.debug)

    with logging_block("Set global random seed"):
        set_global_random_seed(args.random_seed)

    with logging_block("Preprocess data"):
        data_collection, meta_data = data_factory.preprocess(args, return_meta=True)
        data_collection.summary()
        meta_data.summary()
        data_generator = DataGenerator(data_collection.train, batch_size=args.batch_size)

    with logging_block("Prepare Generator"):
        generator = generator_factory.create(args, meta_data)

    with logging_block("Prepare Generator Trainer"):
        trainer = trainer_factory.create(args, meta_data, generator)
        trainer.summary()

    with logging_block("Prepare Callback"):
        if base_tag is None:
            base_tag = f"{args.dataset}@{time.strftime('%Y%m%d-%H%M%S')}"
        data_generator.callback = CallbackFactory(
            trainer=trainer,
            generator=generator,
            data_collection=data_collection,
            meta_data=meta_data,
            tags=args.tags + [base_tag],
        ).create_by_args(args)

    with tf.Session(config=get_tf_config_proto(args.jit)) as sess:
        if checkpoint:
            print(f"Restore from checkpoint: {checkpoint}")
            tf.train.Saver().restore(sess, save_path=checkpoint)
            data_generator.skip_epochs(int(checkpoint.split('-')[-1]))
        else:
            tf.global_variables_initializer().run()

        for batch_data in data_generator.iter_batch_until(
            n_epochs=args.epochs,
            logs={'arg_string': " ".join(argv)},  # for callback.on_train_begin()
        ):
            trainer.fit_batch(batch_data)


def parse_args(argv):
    from library.my_argparse import MyArgumentParser
    from library.my_argparse.formatters import MyRawTextHelpFormatter
    from scripts.core_parsers import (
        train_parser,
        evaluate_parser,
        save_parser,
        logging_parser,
        backend_parser,
        develop_parser,
    )

    parser = MyArgumentParser(
        description='TextGAN.',
        formatter_class=MyRawTextHelpFormatter,
        fromfile_prefix_chars='@',
        parents=[
            data_factory.create_parser(),
            trainer_factory.create_parser(),
            train_parser(),
            evaluate_parser(),
            save_parser(),
            logging_parser(),
            backend_parser(),
            develop_parser(),
        ],
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(argv=sys.argv[1:])
