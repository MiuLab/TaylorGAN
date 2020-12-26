import cProfile
import pstats
import sys
from typing import Dict

from library.utils import format_path

from .base import Callback


class TrainProfiler(Callback):

    def __init__(
            self,
            warm_up: int,
            duration: int,
            export_filepath: str,
            stop_training_when_finish: bool = False,
        ):
        if warm_up < 0:
            raise ValueError("'warm_up' should not be negative!")
        self.warm_up = warm_up
        if duration <= 0:
            raise ValueError("'duration' should be positive!")
        self.duration = duration
        self.export_filepath = export_filepath
        self.stop_training_when_finish = stop_training_when_finish

    def on_train_begin(self, logs: Dict = None):
        self.profile = cProfile.Profile(subcalls=False)

    def on_batch_begin(self, batch: int, logs: Dict = None):
        if batch == self.warm_up:
            print(f"Updates {self.warm_up} times.")
            print("Complete warm-up, start profiling.")
            self.profile.enable()

    def on_batch_end(self, batch: int, batch_data):
        if batch != self.warm_up + self.duration:
            return

        self.profile.disable()
        print(f"Updates {self.warm_up} + {self.duration} times.")
        print(f"Complete profiling, export stats to {self.export_filepath}")
        with open(self.export_filepath, 'w') as f_out:
            stats = pstats.Stats(self.profile, stream=f_out)
            stats.strip_dirs().sort_stats('cumtime').print_stats()

        if self.stop_training_when_finish:
            print("Exit by TrainProfiler.")
            sys.exit(0)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(warm_up={self.warm_up}, "
            f"duration={self.duration}, path={format_path(self.export_filepath)})"
        )
