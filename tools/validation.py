import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False



class PygameRecord:
    def __init__(self, filename: str, fps: int):
        self.filename = filename
        self.fps = fps
        self.frames = []

    def add_frame(self):
        surface = pygame.display.get_surface()
        array = pygame.surfarray.array3d(surface)
        array = np.moveaxis(array, 0, 1)  # Convert to (height, width, channels)
        image = Image.fromarray(np.uint8(array))
        self.frames.append(image)

    def save(self):
        self.frames[0].save(
            self.filename,
            save_all=True,
            append_images=self.frames[1:],
            loop=0,
            duration=int(1000 / self.fps)
        )

    
from image_queue import Game
import pygame


def check(model, device, dataset):

    game = Game(device, dataset, model)
    # Initialize Pygame
    pygame.init()

    font = pygame.font.SysFont(None, 48)
    recorder = PygameRecord("output.gif", 30)
    current_key = None

    # Set up display
    WIDTH, HEIGHT = 500, 500
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    frame_count = 0

    running = True
    while running:
        if pygame.key.get_pressed()[pygame.K_SPACE]:
            game.reset()
            frame_count = 0
        #pygame.time.delay(20)
        game.make_next(pygame.key.get_pressed())
        
        if frame_count > 2:
            pygame_surface = game.get_current()
            screen.blit(pygame_surface, (0, 0))    


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


            if event.type == pygame.KEYDOWN:
                current_key = pygame.key.name(event.key)
            elif event.type == pygame.KEYUP:
                current_key = None  # Clear when released


        # Show current key name
        if current_key:
            key_surface = font.render(f"Key: {current_key}", True, (255, 255, 255))
            text_rect = key_surface.get_rect(center=(480 // 2, 50))
            screen.blit(key_surface, text_rect)

        recorder.add_frame()

        # Update the display
        pygame.display.update()



        frame_count += 1

    # Quit Pygame
    pygame.quit()
    recorder.save()



def check_static(model, device, dataset):
    import numpy as np


    batch_x, batch_y, labels_x, labels_y, subset_no = next(iter(DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)))


    pred = model._predict(batch_x.to(device), labels_x.to(device), labels_y.to(device))

    images = np.concatenate([batch_x.detach().cpu().numpy(), pred.detach().cpu().numpy()], axis=1).squeeze(0)

    import matplotlib.pyplot as plt

    # Set up the figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 2))

    for i in range(4):
        axes[i].imshow(images[i].squeeze(0), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')

    plt.tight_layout()
    plt.show()

from torch import nn
import torch
import torch.nn as nn
from pytorch_msssim import ssim

# Define SSIM loss
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, data_range=1.0, size_average=self.size_average)

import torch
import torch.nn as nn

class TemporalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, gt):
        """
        pred, gt: tensors of shape [B, T, C, H, W]
        Computes L1 loss on temporal differences between consecutive frames.
        """
        # Compute temporal differences along the time dimension
        pred_diff = pred[:, :-1] - pred[:, 1:]  # shape [B, T-1, C, H, W]
        gt_diff   = gt[:, :-1]   - gt[:, 1:]    # shape [B, T-1, C, H, W]
        
        # Flatten time dimension into batch dimension for L1Loss
        B, T_minus1, C, H, W = pred_diff.shape
        pred_diff = pred_diff.reshape(B * T_minus1, C, H, W)
        gt_diff   = gt_diff.reshape(B * T_minus1, C, H, W)
        
        return self.l1(pred_diff, gt_diff)



def PSNR(pred, gt, max_val=1.0, eps=1e-8):
    """
    pred, gt: tensors of shape [B, T, C, H, W]
    Computes PSNR on temporal differences between consecutive frames.
    """
    # Compute temporal differences
    pred_diff = pred[:, :-1] - pred[:, 1:]  # [B, T-1, C, H, W]
    gt_diff   = gt[:, :-1]   - gt[:, 1:]    # [B, T-1, C, H, W]

    # Flatten time into batch
    B, T_minus1, C, H, W = pred_diff.shape
    pred_diff = pred_diff.reshape(B * T_minus1, C, H, W)
    gt_diff   = gt_diff.reshape(B * T_minus1, C, H, W)

    # Compute MSE
    mse = torch.mean((pred_diff - gt_diff) ** 2, dim=[1, 2, 3])  # shape [B * T-1]

    # Compute PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + eps))  # shape [B * T-1]

    # Return mean PSNR over all temporal slices
    return psnr.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowLoss(nn.Module):
    def __init__(self, use_motion_weighting=True, slot_trajectory_hook=None):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.use_motion_weighting = use_motion_weighting
        self.slot_trajectory_hook = slot_trajectory_hook

        # Sobel kernels
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([[1, 0, -1],
                                                  [2, 0, -2],
                                                  [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y.weight.data = torch.tensor([[1, 2, 1],
                                                  [0, 0, 0],
                                                  [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def compute_flow(self, frames):
        B, T, C, H, W = frames.shape
        flows = []
        for t in range(T - 1):
            I1 = frames[:, t]
            I2 = frames[:, t + 1]

            # Grayscale conversion
            if C == 3:
                I1_gray = 0.2989 * I1[:, 0] + 0.5870 * I1[:, 1] + 0.1140 * I1[:, 2]
                I2_gray = 0.2989 * I2[:, 0] + 0.5870 * I2[:, 1] + 0.1140 * I2[:, 2]
            else:
                I1_gray = I1[:, 0]
                I2_gray = I2[:, 0]

            I1_gray = I1_gray.unsqueeze(1)
            I2_gray = I2_gray.unsqueeze(1)

            # Spatial gradients
            Ix = self.sobel_x(I1_gray)
            Iy = self.sobel_y(I1_gray)
            It = I2_gray - I1_gray

            epsilon = 1e-3
            denom = Ix**2 + Iy**2 + epsilon
            u = -Ix * It / denom
            v = -Iy * It / denom

            flow = torch.cat([u, v], dim=1)
            flows.append(flow)

        return torch.stack(flows, dim=1)  # [B, T-1, 2, H, W]

    def forward(self, pred, gt, slots=None):
        """
        pred, gt: [B, T, C, H, W]
        slots: optional [B, T, num_slots, D]
        """
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        pred_flow = self.compute_flow(pred)
        gt_flow = self.compute_flow(gt)

        # Motion weighting
        if self.use_motion_weighting:
            motion_mag = (gt[:, 1:] - gt[:, :-1]).abs().mean(dim=2, keepdim=True)  # [B, T-1, 1, H, W]
            motion_mag = motion_mag.expand_as(gt_flow)
            flow_loss = self.l1(pred_flow, gt_flow) * motion_mag
        else:
            flow_loss = self.l1(pred_flow, gt_flow)

        flow_loss = flow_loss.mean()

        # Optional slot trajectory regularization
        slot_loss = 0.0
        if self.slot_trajectory_hook and slots is not None:
            slot_loss = self.slot_trajectory_hook(slots)

        return flow_loss + slot_loss


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


criterion_ssim = SSIMLoss()
criterion_temporal = TemporalLoss()
criterion_optical = OpticalFlowLoss()
criterion_mse = nn.MSELoss()

def get_pos(frame, threshold=0.5):
    mask = (frame.squeeze(0) > threshold).float()
    if mask.sum() == 0:
        return 0.0
    coords = torch.nonzero(mask)
    y_mean = coords[:, 0].float().mean()
    return y_mean.item()

import matplotlib.pyplot as plt

def check_speed(frames):
    first_frame = frames[0, 0]       # shape: (C, H, W)
    last_frame  = frames[0, -1]      # shape: (C, H, W)

    start_pos = get_pos(first_frame)
    end_pos = get_pos(last_frame)

    print(end_pos - start_pos)
    

    frames = [first_frame, last_frame]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, frame, title in zip(axes, frames, ['First Frame', 'Last Frame']):
        img = frame.detach().cpu().numpy().transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img.squeeze(-1)
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def check_static2(model, device, dataloader):
    loss1_all = []
    loss2_all = []
    loss3_all = []
    loss4_all = []
    loss5_all = []
    loss6_all = []
    for batch_x, batch_y, labels_x, labels_y, subset_no in dataloader:
        import numpy as np
        batch_x, labels_x, labels_y = batch_x.to(device), labels_x.to(device), labels_y.to(device)
        batch_y = batch_y.to(device)
        pred = model._predict(batch_x, labels_x, labels_y)
        
        loss1 = criterion_ssim(pred, batch_y)
        loss2 = criterion_temporal(pred, batch_y)
        loss3 = criterion_optical(pred, batch_y)
        loss4 = semantic_position_loss(pred, batch_y)
        loss5 = PSNR(pred, batch_y)
        loss6 = criterion_mse(pred, batch_y)

        #print(loss1.item(), loss2.item(), loss3.item())
        #check_speed(batch_y)

        loss1_all.append(loss1.item())
        loss2_all.append(loss2.item())
        loss3_all.append(loss3.item())
        loss4_all.append(loss4.item())
        loss5_all.append(loss5.item())
        loss6_all.append(loss6.item())
    print(np.mean(loss1_all), np.mean(loss2_all), 
          np.mean(loss3_all), np.mean(loss4_all), 
          np.mean(loss5_all), np.mean(loss6_all))


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
    
    train_load, validate_load, test_load = load_data(1, 1, "./data")

    check_static2(model, 'cuda', validate_load)
