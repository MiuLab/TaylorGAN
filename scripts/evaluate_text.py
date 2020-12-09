import os
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

from library.utils import random_sample
from core.evaluate import BLEUCalculator, SmoothingFunction, FEDCalculator
from factories import data_factory
from scripts.snippets import set_package_verbosity

# HUB_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
HUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
RLM_EPOCHS = 100


def main(argv):
    args = parse_args(argv)
    set_package_verbosity(args.debug)

    data_collection, meta_data = data_factory.preprocess(args, return_meta=True)
    tokenizer = meta_data.tokenizer

    metric_calcuators = []
    if args.fed:
        metric_calcuators.append(FEDMetrics(data_collection, tokenizer, sample_size=args.fed))
    if args.bleu:
        metric_calcuators.append(
            BLEUMetrics(
                data_collection,
                max_gram=args.bleu,
                eos_idx=tokenizer.eos_idx,
                cache_dir=meta_data.cache_dir,
            ),
        )

    with open(args.eval_path, 'r') as f_in:
        texts = [line.rstrip() for line in f_in.readlines()]
        tokens = tokenizer.texts_to_array(texts)

    metrics = {}
    for calc in metric_calcuators:
        metrics.update(calc.calculate(tokens=tokens, texts=texts))

    print(
        f"{os.path.basename(args.eval_path)},",
        *[f"{key}: {val:.5f}" for key, val in metrics.items()],
        sep="\n",
    )


class BLEUMetrics:

    def __init__(self, data_collection, cache_dir, eos_idx=1, max_gram=5):
        self.calculators = {
            tag: BLEUCalculator(
                dataset.ids,
                max_gram=max_gram,
                eos_idx=eos_idx,
                smoothing=SmoothingFunction.fuzz_smoothing,
                cache_dir=cache_dir / f"{tag}_BLEU" if cache_dir else None,
                verbose=True,
            )
            for tag, dataset in data_collection.items()
        }
        self.eos_idx = eos_idx
        self.max_gram = max_gram

    def calculate(self, tokens, **kwargs):
        metrics = {}
        for tag, calc in self.calculators.items():
            mean_bleu = calc.mean_bleu(tokens)
            metrics.update({f'{tag}_{key}': score for key, score in mean_bleu.items()})

        metrics.update(BLEUCalculator.selfbleu(
            tokens,
            max_gram=self.max_gram,
            eos_idx=self.eos_idx,
            smoothing=SmoothingFunction.fuzz_smoothing,
        ))
        return metrics


class FEDMetrics:

    def __init__(self, data_collection, tokenizer, sample_size):
        self.calculators = {
            tag: FEDCalculator(
                hub_url=HUB_URL,
                references=random_sample(dataset.texts, sample_size),
            )
            for tag, dataset in data_collection.items()
        }

    def calculate(self, texts, **kwargs):
        return {
            f'{tag} FED': calc.calculate_fed_score(texts)
            for tag, calc in self.calculators.items()
        }


def parse_args(argv):
    from library.my_argparse import MyArgumentParser
    from scripts.core_parsers import evaluate_parser, develop_parser

    parser = MyArgumentParser(parents=[
        data_factory.create_parser(),
        evaluate_parser(),
        develop_parser(),
    ])
    parser.add_argument('--eval-path', required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(argv=sys.argv[1:])
