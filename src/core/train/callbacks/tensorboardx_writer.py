import os
import sys
from pathlib import Path

from tensorboardX import SummaryWriter

from core.train.pubsub_base import Subscriber
from library.utils import format_path

from .base import Callback
from .channels import channels


class TensorBoardXWritter(Callback):

    def __init__(self, logdir: Path, log_period: int, updaters):
        self.logdir = logdir
        self.log_period = log_period
        self.updaters = updaters

    def on_train_begin(self, is_restored: bool):
        self.logdir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(logdir=self.logdir)
        self.writer.add_text(
            'restore_args' if is_restored else 'args',
            ' '.join(sys.argv[1:]),
            0,
        )
        for updater in self.updaters:
            updater.attach_subscriber(
                ModuleLossWriter(
                    self.writer,
                    tag_template=os.path.join('losses', updater.module.scope, '{key}'),
                    log_period=self.log_period,
                ),
            )

        for name, channel in channels.items():
            channel.attach_subscriber(
                MetricsWriter(self.writer, tag_template=os.path.join(name, '{key}')),
            )

    def get_config(self):
        return {'period': self.log_period, 'logdir': format_path(self.logdir)}


class ModuleLossWriter(Subscriber):

    def __init__(self, writer, tag_template, log_period: int = 1):
        self.writer = writer
        self.tag_template = tag_template
        self.log_period = log_period

    def update(self, step, losses):
        if step % self.log_period == 0:
            for key, val in losses.items():
                self.writer.add_scalar(
                    tag=self.tag_template.format(key=key),
                    scalar_value=val,
                    global_step=step,  # TODO enerator step?
                )


class MetricsWriter(Subscriber):

    def __init__(self, writer, tag_template):
        self.writer = writer
        self.tag_template = tag_template

    def update(self, step, vals):
        for key, val in vals.items():
            self.writer.add_scalar(
                tag=self.tag_template.format(key=key),
                scalar_value=val,
                global_step=step,  # TODO batch ? epoch?
            )
