# mmcv runner
import torch
import time
import logging
import os.path as osp
import os
import numpy as np
import copy
import torch.distributed as dist

from . import hooks
from .utils import init_logger, get_host_info
from ..utils import obj_from_dict
from .checkpoint import load_checkpoint, save_checkpoint
from .hooks import Hook, IterTimerHook, LrUpdaterHook, CheckpointHook, TextLoggerHook, \
    OptimizerHook, ValHook, lr_updater, SummaryHook
from .priority import get_priority
from .log_buffer import LogBuffer


class Runner(object):
    def __init__(self,
                 model,
                 batch_processor,
                 optimizer,
                 work_dir,
                 log_level=logging.INFO,
                 logger=None):
        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = self.init_optimizer(optimizer)
        self.work_dir = work_dir

        if logger is None:
            self.logger = init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.train_log_buffer = LogBuffer()
        self.val_log_buffer = LogBuffer()

        self.mode = 'train'
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0


    @property
    def epoch(self):
        return self._epoch

    @property
    def hooks(self):
        return self._hooks

    @property
    def iter(self):
        return self._iter

    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        return self._max_iters


    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            if 'bn_bias_wd' in optimizer:
                bn_bias_wd = optimizer.pop('bn_bias_wd')
            else:
                bn_bias_wd = True
            if not bn_bias_wd:
                wd_params_ = []
                wo_wd_params_ = []
                for name, param in self.model.named_parameters():
                    if 'bn' in name:
                        wo_wd_params_.append(param)
                        continue
                    if 'bias' in name:
                        wo_wd_params_.append(param)
                        continue
                    wd_params_.append(param)
                if 'weight_decay' in optimizer:
                    wd = optimizer.pop('weight_decay')
                optimizer = obj_from_dict(
                    optimizer, torch.optim, 
                    dict(params=[dict(params=wd_params_, weight_decay=wd),
                                 dict(params=wo_wd_params_, weight_decay=0)]))
            else:
                optimizer = obj_from_dict(
                     optimizer, torch.optim, dict(params=self.model.parameters()))
            print('optimizer={}'.format(optimizer))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}_iter_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = osp.join(out_dir, filename_tmpl.format(self.epoch + 1, self.inner_iter))
        linkname = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filename, optimizer=optimizer, meta=meta)
        if os.path.lexists(linkname):
            os.remove(linkname)
        os.symlink(os.path.abspath(filename), linkname)
   

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert(issubclass(hook_type, Hook))
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def register_hook(self, hook, priority='NORMAL'):
        assert(isinstance(hook, Hook))
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')


    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                val_config=None,
                                log_config=None,
                                summary_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        if val_config is not None:
            self.register_hook(self.build_hook(val_config, ValHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
        if summary_config is not None:
            self.register_hook(self.build_hook(summary_config, SummaryHook))


    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch'] - 1
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d, inner_iter %d', self.epoch + 1, self.iter, self.inner_iter)
        #self.logger.info('resumed epoch %d, iter %d, inner_iter %d', self.epoch, self.iter, self.inner_iter)
  
    def val(self, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.call_hook('before_val_epoch')
        for i, data_batch in enumerate(self.val_dataloader):
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, mode='val'
                )
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.val_log_buffer.update(outputs['log_vars'],
                                           outputs['num_samples'])
            self.call_hook('after_val_iter')
        self.call_hook('after_val_epoch')
        self.model.train()
        self.mode = 'train'

    def nbase_1top_train(self, data_loaders, dataloader_lens, batch_size, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.train_dataloader = data_loaders['train']
        self.val_dataloader = data_loaders['val']
        self.dataloader_lens = dataloader_lens
        self._max_iters = self._max_epochs * self.dataloader_lens
        self.call_hook('before_train_epoch')
        rank = dist.get_rank()
        if rank == 0:
            for i in range(self.dataloader_lens):
                self._inner_iter = i
                self.call_hook('before_train_iter')
                outputs = self.batch_processor(
                    self.model, None, mode='train', num_samples=batch_size,
                )
                if not isinstance(outputs, dict):
                    raise TypeError('batch_processor() must return a dict')
                if 'log_vars' in outputs:
                    self.train_log_buffer.update(outputs['log_vars'],
                                                 outputs['num_samples'])
                self.outputs = outputs
                self.call_hook('after_train_iter')
                self._iter += 1
        else:
            for i, data_batch in enumerate(self.train_dataloader):
                if i >= self.dataloader_lens:
                    break
                self._inner_iter = i
                self.call_hook('before_train_iter')
                outputs = self.model(data_batch)
                self.call_hook('after_train_iter')
                self._iter += 1
        self.call_hook('after_train_epoch')
        self._epoch += 1
        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)
        if hasattr(self.train_dataloader, 'batch_sampler') and hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
            self.train_dataloader.batch_sampler.set_epoch(self.epoch)

    def nbase_mtop_train(self, data_loaders, dataloader_lens, batch_size, top_model_ranks, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.train_dataloader = data_loaders['train']
        self.val_dataloader = data_loaders['val']
        self.dataloader_lens = dataloader_lens
        self._max_iters = self._max_epochs * self.dataloader_lens
        self.call_hook('before_train_epoch')
        rank = dist.get_rank()
        if rank in top_model_ranks:
            for i in range(self.dataloader_lens):
                self._inner_iter = i
                self.call_hook('before_train_iter')
                outputs = self.batch_processor(
                    self.model, None, mode='train', num_samples=batch_size,
                )
                if not isinstance(outputs, dict):
                    raise TypeError('batch_processor() must return a dict')
                if 'log_vars' in outputs:
                    self.train_log_buffer.update(outputs['log_vars'],
                                                 outputs['num_samples'])
                self.outputs = outputs
                self.call_hook('after_train_iter')
                self._iter += 1
        else:
            for i, data_batch in enumerate(self.train_dataloader):
                if i >= self.dataloader_lens:
                    break
                self._inner_iter = i
                self.call_hook('before_train_iter')
                outputs = self.model(data_batch)
                self.call_hook('after_train_iter')
                self._iter += 1
        self.call_hook('after_train_epoch')
        self._epoch += 1
        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)
        if hasattr(self.train_dataloader, 'batch_sampler') and hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
            self.train_dataloader.batch_sampler.set_epoch(self.epoch)

    def train(self, data_loaders, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.train_dataloader = data_loaders['train']
        self.val_dataloader = data_loaders['val']
        self._max_iters = self._max_epochs * len(self.train_dataloader)
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(self.train_dataloader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, mode='train'
            )
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.train_log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            if 'summary_vars' in outputs:
                self.summary_vars = outputs['summary_vars']
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)
        if hasattr(self.train_dataloader, 'batch_sampler') and hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
            self.train_dataloader.batch_sampler.set_epoch(self.epoch)

    def run(self, max_epochs, data_loaders, **kwargs):
        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'None'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('max: {} epochs'.format(max_epochs))
        self.call_hook('before_run')

        while self.epoch < max_epochs: 
            self.train(data_loaders, **kwargs)

        time.sleep(1) # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def nbase_1top_run(self, max_epochs, data_loaders, dataloader_lens, batch_size, **kwargs):
        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'None'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('max: {} epochs'.format(max_epochs))
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            self.nbase_1top_train(data_loaders, dataloader_lens, batch_size, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def nbase_mtop_run(self, max_epochs, data_loaders, dataloader_lens, batch_size, top_model_ranks, **kwargs):
        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'None'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('max: {} epochs'.format(max_epochs))
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            self.nbase_mtop_train(data_loaders, dataloader_lens, batch_size, top_model_ranks, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')



