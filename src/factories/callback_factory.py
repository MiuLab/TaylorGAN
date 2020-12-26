import os
import warnings
from functools import partial
from typing import List

from library.utils import cached_property, logging_block, random_sample
from core.evaluate import (
    SmoothingFunction, TextGenerator, PerplexityCalculator, BLEUCalculator, FEDCalculator,
)
from core.preprocess import MetaData
from core.train.callbacks import (
    Callback,
    CallbackList,
    DispatchCallback,
    ModelCheckpoint,
    ModelSaver,
    TrainProfiler,
)
from core.train.callbacks.evaluators import GeneratorProxy, mean_length
from core.train.callbacks.loggers import ProgbarLogger, TensorBoardXLogger


class CallbackFactory:

    def __init__(self, trainer, generator, data_collection, meta_data: MetaData, tags: List[str]):
        self.trainer = trainer
        self.generator = generator
        self.data_collection = data_collection
        self.meta_data = meta_data
        self.tags = tags

    def create_by_args(self, args) -> Callback:
        callback_list = CallbackList(
            evaluater=self.create_evaluator(bleu_n_gram=args.bleu, fed_sample_size=args.fed),
            loggers=self.create_loggers(tensorboard_logdir=args.tensorboard),
            others=[
                *self.create_savers(
                    serving_root=args.serving_root,
                    checkpoint_root=args.checkpoint_root,
                    period=args.save_period,
                ),
                *self.create_profiler(export_path=args.profile),
            ],
        )
        callback_list.summary()
        return callback_list

    def create_evaluator(self, bleu_n_gram: int = None, fed_sample_size: int = None):
        proxy = GeneratorProxy(self.text_generator)
        evaluator = DispatchCallback()
        evaluator.on_batch_end.attach(
            proxy.evaluate_ids(
                partial(mean_length, eos_idx=self.meta_data.eos_idx),
                sample_size=64,
                target_channel='samples',
            ),
            period=10,
        )
        if bleu_n_gram is not None:
            for key, dataset in self.data_collection.items():
                with logging_block(f"Building '{key}' data BLEU table..."):
                    calculater = BLEUCalculator(
                        dataset.ids,
                        max_gram=bleu_n_gram,
                        eos_idx=self.meta_data.eos_idx,
                        smoothing=SmoothingFunction.fuzz_smoothing,
                        cache_dir=self.meta_data.cache_dir / f"{key}_BLEU",
                        verbose=True,
                    )
                evaluator.on_batch_end.attach(
                    proxy.evaluate_ids(calculater.mean_bleu, sample_size=64, target_channel=key),
                    period=10,
                )

            def selfbleu(word_ids):
                print("Evaluating generated data SelfBLEU...")
                print()
                return BLEUCalculator.selfbleu(
                    word_ids,
                    max_gram=bleu_n_gram,
                    eos_idx=self.meta_data.eos_idx,
                    smoothing=SmoothingFunction.fuzz_smoothing,
                )

            evaluator.on_epoch_end.attach(
                proxy.evaluate_ids(
                    selfbleu,
                    target_channel='samples',
                    sample_size=min(10000, 2 * len(self.data_collection.train)),
                ),
            )

        if fed_sample_size is not None:

            for key, dataset in self.data_collection.items():
                print(f"Building '{key}' data FED sentence encoder...")
                calculator = FEDCalculator(
                    hub_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                    references=random_sample(dataset.texts, size=fed_sample_size),
                )

                def fed(texts):
                    print("Evaluating FED Score ...")
                    print()
                    return {"FED": calculator.calculate_fed_score(candidates=texts)}

                evaluator.on_epoch_end.attach(
                    proxy.evaluate_text(
                        fed,
                        sample_size=fed_sample_size,
                        target_channel=key,
                    ),
                )

        evaluator.on_batch_end.attach(
            proxy.log_samples_command(sample_size=3),
            period=100,
        )
        return evaluator

    def create_loggers(self, tensorboard_logdir=None):
        # TODO add signals
        yield ProgbarLogger(
            desc=os.path.join(*self.tags),
            trainer=self.trainer,
        )
        if tensorboard_logdir is not None:
            yield TensorBoardXLogger(
                trainer=self.trainer,
                logdir=os.path.join(tensorboard_logdir, *self.tags),
                log_period=10,
            )

    def create_savers(self, serving_root=None, checkpoint_root=None, period=1):
        if serving_root is not None:
            serving_dir = os.path.join(serving_root, *self.tags)
            os.makedirs(serving_dir, exist_ok=True)
            self.meta_data.tokenizer.save(os.path.join(serving_dir, 'tokenizer.json'))
            yield ModelSaver(
                signature={
                    'generate': self.text_generator.signature,
                    'perplexity': PerplexityCalculator.from_model(
                        self.generator,
                        maxlen=self.meta_data.maxlen,
                    ).signature,
                },
                directory=serving_dir,
                period=period,
            )

        if checkpoint_root is not None:
            yield ModelCheckpoint(
                directory=os.path.join(checkpoint_root, *self.tags),
                period=period,
            )
        else:
            warnings.warn("`checkpoint_root` is not given. Training can't be restored!")

    def create_profiler(self, export_path: str = None):
        if export_path is not None:
            yield TrainProfiler(
                warm_up=100,
                duration=200,
                export_filepath=export_path,
                stop_training_when_finish=True,
            )

    @cached_property
    def text_generator(self):
        return TextGenerator.from_model(self.generator, tokenizer=self.meta_data.tokenizer)
