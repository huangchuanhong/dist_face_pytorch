import datetime

from .base import LoggerHook

class TextLoggerHook(LoggerHook):

    def __init__(self, interval=10, ignore_last=False, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def before_run(self, runner):
        super(TextLoggerHook, self).before_run(runner)
        self.start_iter = runner.iter

    def log(self, runner):
        if runner.mode == 'train':
            self._train_log(runner)
        elif runner.mode == 'val':
            self._val_log(runner)

    def _train_log(self, runner):
        lr_str = ', '.join(
            ['{:.5f}'.format(lr) for lr in runner.current_lr()]
        )
        if isinstance(runner.train_dataloader, list):
            epoch_total_iter = sum([len(dataloader) for dataloader in runner.train_dataloader])
        else:
            if runner.train_dataloader:
                epoch_total_iter = len(runner.train_dataloader)
            else:
                epoch_total_iter = runner.dataloader_lens
        log_str = 'Epoch [{}][{}/{}]\t lr: {}, '.format(
            runner.epoch + 1, runner.inner_iter + 1,
            epoch_total_iter, lr_str
        )

        if 'time' in runner.train_log_buffer.output:
            self.time_sec_tot += (
                runner.train_log_buffer.output['time'] * self.interval
            )
            time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter + 1
            )
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += 'eta: {}, '.format(eta_str)
            log_str += (
                'time: {log[time]:.3f}, data_time: {log[data_time]:.3f}, '.
                format(log=runner.train_log_buffer.output)
            )
        log_items = []
        for name, val in runner.train_log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)

    def _val_log(self, runner):
        log_str = 'Epoch({}) [{}]\t'.format(runner.mode, runner.epoch + 1)

        if 'time' in runner.val_log_buffer.output:
            self.time_sec_tot += (
                runner.val_log_buffer.output['time'] * self.interval
            )
            time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter + 1
            )
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += 'eta: {}, '.format(eta_str)
            log_str += (
                'time: {log[time]:.3f}, data_time: {log[data_time]:.3f}, '.
                format(log=runner.val_log_buffer.output)
            )
        log_items = []
        for name, val in runner.val_log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)

