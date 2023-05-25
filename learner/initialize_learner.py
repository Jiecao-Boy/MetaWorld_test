import hydra 

from torch.nn.parallel import DistributedDataParallel as DDP

# from .byol import BYOLLearner
# from .vicreg import VICRegLearner
# from .behavior_cloning import ImageTactileBC
# from .bet import BETLearner
# from .bc_gmm import BCGMM
# from .simclr import SIMCLRLearner
# from .mocov3 import MOCOLearner

from learner.VisualBC import VisualBC

# from tactile_learning.utils import *
# from tactile_learning.models import  *

from utils import *
from model import *

def init_learner(cfg, device, rank=0):
    if cfg.learner_type == 'bc':
        return init_bc(cfg, device, rank)
    return None

def init_bc(cfg, device, rank):
    image_encoder = hydra.utils.instantiate(cfg.encoder.image_encoder).to(device)
    image_encoder = DDP(image_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # tactile_encoder = hydra.utils.instantiate(cfg.encoder.tactile_encoder).to(device)
    # tactile_encoder = DDP(tactile_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    last_layer = hydra.utils.instantiate(cfg.encoder.last_layer).to(device)
    last_layer = DDP(last_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # optim_params = list(image_encoder.parameters()) + list(tactile_encoder.parameters()) + list(last_layer.parameters())
    optim_params = list(image_encoder.parameters()) + list(last_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params = optim_params)

    # learner = ImageTactileBC(
    #     image_encoder = image_encoder, 
    #     tactile_encoder = tactile_encoder,
    #     last_layer = last_layer,
    #     optimizer = optimizer,
    #     loss_fn = cfg.learner.loss_fn,
    #     representation_type = cfg.learner.representation_type,
    #     freeze_encoders = cfg.learner.freeze_encoders
    # )
    learner = VisualBC(
        image_encoder = image_encoder, 
        last_layer = last_layer,
        optimizer = optimizer,
        loss_fn = cfg.learner.loss_fn,
        representation_type = cfg.learner.representation_type,
        freeze_encoders = cfg.learner.freeze_encoders
    )
    learner.to(device) 
    
    return learner