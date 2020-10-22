import time
import logging
import os.path as osp
import os
from getpass import getuser
from socket import gethostname

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def init_logger(log_dir, level=logging.INFO):
    _format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        format=_format, level=level
    )
    _logger = logging.getLogger(__name__)
    if log_dir:
        if not osp.isdir(log_dir):
            os.makedirs(log_dir)
        def add_file_handler(_logger, filename=None, mode='w', level=logging.INFO):
            file_handler = logging.FileHandler(filename, mode)
            file_handler.setFormatter(
                logging.Formatter(_format)
            )
            file_handler.setLevel(level)
            _logger.addHandler(file_handler)

        filename = '{}.log'.format(get_time_str())
        log_file = osp.join(log_dir, filename)
        add_file_handler(_logger, log_file, level=level)
    return _logger

def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


