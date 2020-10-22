from .hook import Hook

class CheckpointHook(Hook):

    def __init__(self,
                 interval=-1,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir,
            save_optimizer=self.save_optimizer,
            **self.args
        )

    def after_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir,
            save_optimizer=self.save_optimizer,
            **self.args
        )

