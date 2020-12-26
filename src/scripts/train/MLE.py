from scripts.train import train


def main():
    import sys
    from factories.trainer_factory.MLE import MLECreator
    train.main(args=train.parse_args(sys.argv[1:], algorithm=MLECreator))


if __name__ == '__main__':
    main()
