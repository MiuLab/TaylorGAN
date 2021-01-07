import pathlib
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

from core.evaluate import TextGenerator
from core.preprocess import Tokenizer


def main(args):
    tokenizer = Tokenizer.load(args.model_path.parent / 'tokenizer.json')
    generator = TextGenerator.load_traced(args.model_path, tokenizer=tokenizer)
    print(f"Generate sentences to '{args.export_path}'")
    with open(args.export_path, 'w') as f_out:
        f_out.writelines([
            sentence + "\n"
            for sentence in generator.generate_texts(args.samples)
        ])


def parse_args(argv):
    from flexparse import ArgumentParser, IntRange
    from scripts.parsers import develop_parser, load_parser

    parser = ArgumentParser(parents=[develop_parser(), load_parser()])
    parser.add_argument(
        '--export-path',
        type=pathlib.Path,
        required=True,
        help='path to save generated texts.',
    )
    parser.add_argument(
        '--samples',
        type=IntRange(minval=1),
        default=10000,
        help='number of generated samples(sentences).',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(parse_args(sys.argv[1:]))
