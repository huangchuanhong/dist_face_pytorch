from ..utils import obj_from_dict
from . import dataset as dataset_factory
from . import dataloaders as dataloader_factory

def get_dataloader(data_config, dataloader_config):
    print('data_config={}'.format(data_config))
    dataset = obj_from_dict(data_config, dataset_factory)
    dataloader_config['dataset'] = dataset
    dataloader = obj_from_dict(dataloader_config, dataloader_factory)
    return dataloader
