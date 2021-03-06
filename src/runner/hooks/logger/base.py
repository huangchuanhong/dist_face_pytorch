from abc import ABCMeta, abstractmethod

from ..hook import Hook

class LoggerHook(Hook):

    __metaclass__ = ABCMeta

    def __init__(self, interval=10, ignore_last=False, reset_flag=False):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag

    @abstractmethod
    def log(self, runner):
        pass

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_train_epoch(self, runner):
        runner.train_log_buffer.clear() # clear logs of last epoch

    def before_val_epoch(self, runner):
        runner.val_log_buffer.clear()

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            runner.train_log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.train_log_buffer.average(self.interval)

        if runner.train_log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.train_log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.train_log_buffer.ready:
            self.log(runner)

    def after_val_epoch(self, runner):
        runner.val_log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.val_log_buffer.clear_output()
