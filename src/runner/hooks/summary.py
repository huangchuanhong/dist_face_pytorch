from tensorboardX import SummaryWriter

from .hook import Hook

class SummaryHook(Hook):
    def __init__(self, logdir='.', interval=10, on=True, summary_type=None):
        self.interval = interval
        self.on = on
        self.summary_type = summary_type
        self.writer = SummaryWriter(logdir=logdir)

    def _summary(self, name, type, value, iter=None):
        assert(type in ['scalar', 'hist', 'image'])
        if type == 'scalar':
            self.writer.add_scalar(name, value, iter)
        elif type == 'hist':
            self.writer.add_histogram(name, value, iter)
        elif type == 'image':
            self.writer.add_image(name, value, iter)

    def after_train_iter(self, runner):
        if not self.on:
            return
        if self.every_n_inner_iters(runner, self.interval):
            for name, type in self.summary_type.items():
                if name in runner.summary_vars.keys():
                    self._summary(name, type, runner.summary_vars[name], runner.iter)

