import warnings

warnings.simplefilter('ignore', category=FutureWarning)

from core.evaluate import TextGenerator
from factories import data_factory


def main(args):
    data_collection, meta = data_factory.preprocess(args, return_meta=True)
    generator = TextGenerator.load_traced(args.model_path, tokenizer=meta.tokenizer)
    for tag, dataset in data_collection.items():
        print(f"Evaluate {tag} perplexity:")
        perplexity = generator.perplexity(dataset.ids)
        print(f"Perplexity = {perplexity}")


def parse_args(argv):
    from flexparse import ArgumentParser
    from scripts.parsers import develop_parser, load_parser

    return ArgumentParser(
        parents=[
            data_factory.PARSER,
            develop_parser(),
            load_parser(),
        ],
    ).parse_args(argv)


if __name__ == '__main__':
    import sys
    main(parse_args(sys.argv[1:]))
