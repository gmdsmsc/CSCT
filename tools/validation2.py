import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)
import torch.nn.functional as F
try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


def compute_centroids(frames):
    # frames: [B, T, 1, H, W] — grayscale
    B, T, _, H, W = frames.shape
    centroids = []

    for t in range(T):
        frame = frames[:, t, 0]  # [B, H, W]
        norm = frame.sum(dim=[1,2], keepdim=True) + 1e-6
        x_coords = torch.linspace(0, W-1, W, device=frames.device).view(1, 1, W).expand(B, H, W)
        y_coords = torch.linspace(0, H-1, H, device=frames.device).view(1, H, 1).expand(B, H, W)

        cx = (frame * x_coords).sum(dim=[1,2], keepdim=True) / norm
        cy = (frame * y_coords).sum(dim=[1,2], keepdim=True) / norm
        centroids.append(torch.cat([cx, cy], dim=2))  # [B, 1, 2]

    return torch.cat(centroids, dim=1)  # [B, T, 2]


def semantic_position_loss(pred_frames, gt_frames):
    pred_c = compute_centroids(pred_frames)
    gt_c   = compute_centroids(gt_frames)

    # Penalize deviation in displacement vectors
    pred_disp = pred_c[:, 1:] - pred_c[:, :-1]
    gt_disp   = gt_c[:, 1:] - gt_c[:, :-1]

    return F.mse_loss(pred_disp, gt_disp)


def get_pos(frame, threshold=0.5):
    mask = (frame.squeeze(0) > threshold).float()
    if mask.sum() == 0:
        return 0.0
    coords = torch.nonzero(mask)
    y_mean = coords[:, 0].float().mean()
    return y_mean.item()




def check_static2(model, device, dataloader):
    for batch_x, batch_y, labels_x, labels_y, subset_no in dataloader:
        import numpy as np
        batch_x, labels_x, labels_y = batch_x.to(device), labels_x.to(device), labels_y.to(device)
        batch_y = batch_y.to(device)
        pred = model._predict(batch_x, labels_x, labels_y)
        
        print(pred.shape, batch_y.shape)
        #loss4 = semantic_position_loss(pred, batch_y)


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'batch_size', 'val_batch_size'])
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]
    if not config['inference'] and not config['test']:
        config['test'] = True

    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' testing  ' + '<'*35)
    print(args)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()

#    mse = exp.inference()

    import torch
    import os.path as osp


    best_model_path = osp.join(exp.path, 'checkpoint.pth')
    exp._load_from_state_dict(torch.load(best_model_path))

    model = exp.method



    from openstl.datasets.dataloader_flappy import load_data
    
    train_load, validate_load, test_load = load_data(1, 1, "E:/python_home/flappy_bird")

    check_static2(model, 'cuda', validate_load)
