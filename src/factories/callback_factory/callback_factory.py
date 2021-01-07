import time
import warnings
from pathlib import Path
from typing import List

from core.evaluate import TextGenerator
from core.preprocess import MetaData
from core.train.callbacks import (
    CallbackList,
    ModelCheckpoint,
    ModelSaver,
    ProgbarLogger,
    TensorBoardXWritter,
    TrainProfiler,
)
from library.utils import cached_property

from .evaluator_creator import EvaluatorCreator


def create(args, trainer, generator, data_collection, meta_data, base_tag):
    base_tag = base_tag or f"{args.dataset}@{time.strftime('%Y%m%d-%H%M%S')}"
    creator = CallbackCreator(
        generator=generator,
        data_collection=data_collection,
        meta_data=meta_data,
        tags=args.tags + [base_tag],
    )

    callback_list = CallbackList([
        creator.create_evaluator(
            bleu_n_gram=args.bleu,
            sample_size=args.batch_size,
            fed_sample_size=args.fed,
        ),
        *creator.create_loggers(
            updaters=trainer.updaters,
            tensorboard_logdir=args.tensorboard,
        ),
        *creator.create_savers(
            trainer=trainer,
            serving_root=args.serving_root,
            checkpoint_root=args.checkpoint_root,
            period=args.save_period,
        ),
        *creator.create_profiler(export_path=args.profile),
    ])
    callback_list.summary()
    return callback_list


class CallbackCreator:

    def __init__(self, generator, data_collection, meta_data: MetaData, tags: List[str]):
        self.generator = generator
        self.data_collection = data_collection
        self.meta_data = meta_data
        self.tag = Path(*tags)

    def create_evaluator(self, bleu_n_gram: int, sample_size: int, fed_sample_size: int):
        return EvaluatorCreator(
            text_generator=self.text_generator,
            data_collection=self.data_collection,
            meta_data=self.meta_data,
        ).create(bleu_n_gram, sample_size, fed_sample_size)

    def create_loggers(self, updaters, tensorboard_logdir: Path):
        yield ProgbarLogger(
            desc=self.tag,
            total=len(self.data_collection.train),
            updaters=updaters,
        )
        if tensorboard_logdir:
            yield TensorBoardXWritter(
                updaters=updaters,
                logdir=tensorboard_logdir / self.tag,
                log_period=10,
            )

    def create_savers(self, trainer, serving_root: Path, checkpoint_root: Path, period: int):
        if serving_root:
            serving_dir = serving_root / self.tag
            serving_dir.mkdir(exist_ok=True)
            self.meta_data.tokenizer.save(serving_dir / 'tokenizer.json')
            yield ModelSaver(
                module=self.text_generator,
                directory=serving_dir,
                period=period,
            )

        if checkpoint_root:
            yield ModelCheckpoint(
                trainer=trainer,
                directory=checkpoint_root / self.tag,
                period=period,
            )
        else:
            warnings.warn("`checkpoint_root` is not given. Training can't be restored!")

    def create_profiler(self, export_path: Path):
        if export_path:
            yield TrainProfiler(
                warm_up=100,
                duration=200,
                export_filepath=export_path,
                stop_training_when_finish=True,
            )

    @cached_property
    def text_generator(self):
        return TextGenerator(self.generator, tokenizer=self.meta_data.tokenizer)
