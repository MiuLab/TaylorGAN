from scripts.train import train


def main():
    import sys
    from factories.trainer_factory.GAN import GANCreator
    train.main(args=train.parse_args(sys.argv[1:], algorithm=GANCreator))


if __name__ == '__main__':
    main()
