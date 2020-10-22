# Note: Only FaceModel can work properly
# MultiDeviceModel: Pytorch can not do both data parallel and model parallel in one project with DDP
# FaceModel1Dist: trash
# NBase1TopModel: Designed to Do base_model in several devices(ranks 1~N) using DDP and Do
#                 top_model in a single device(rank 0) using model parallel. But problem is:
#                 default process_group need DDP range from rank 0 to rank N(how ever we only want
#                 it from rank 1 to rank N), I don't know how to solve this.

from .face_model import FaceModel
from .onebase_1top_dp_model import OneBase1TopDPModel
from .nbase_ddp_1top_mp_model import NBaseDDP1TopMPModel
from .pairwise_model import OneDevicePairwiseModel, DistPairwiseModel
from .nbase_ddp_mtop_mp_model import NBaseDDPMTopMPModel

__all__ = ["FaceModel",
           'OneBase1TopDPModel', 'NBaseDDP1TopMPModel',
           'OneDevicePairwiseModel',
           'DistPairwiseModel',
           'NBaseDDPMTopMPModel']
