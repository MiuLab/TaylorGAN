from .method_objects import GANTrainerCreator


class TrainerFactory:

    def create(self, args, meta_data, generator):
        return GANTrainerCreator(args, meta_data, generator).create()

    def create_parser(self):
        return GANTrainerCreator.create_parser()


trainer_factory = TrainerFactory()
