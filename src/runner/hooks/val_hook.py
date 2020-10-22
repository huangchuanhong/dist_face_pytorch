from .hook import Hook

class ValHook(Hook):

    def __init__(self,
                 interval=-1,
                 **kwargs):
        self.interval = interval
        self.args = kwargs

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        runner.val()
