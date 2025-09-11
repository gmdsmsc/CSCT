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

def visualise_frames(model, device, dataset):
    import matplotlib.pyplot as plt

    batch_x, batch_y, labels_x, labels_y, subset_no = next(iter(DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)))
    pred = model._predict(batch_x.to(device), labels_x.to(device), labels_y.to(device))

    # Extract the first sample from batch
    batch_y_np = batch_y[0].detach().cpu().numpy()  # shape: (T, C, H, W)
    pred_np = pred[0].detach().cpu().numpy()        # shape: (T, C, H, W)

    # Squeeze channel if it's grayscale (C=1)
    batch_y_np = batch_y_np.squeeze(1)  # shape: (T, H, W)
    pred_np = pred_np.squeeze(1)        # shape: (T, H, W)

    # Select specific frames to visualize
    selected_frames = [0, 4, 9]
    fig, axes = plt.subplots(2, len(selected_frames), figsize=(4 * len(selected_frames), 6))

    for i, t in enumerate(selected_frames):
        if t >= batch_y_np.shape[0]:
            continue  # Skip if frame index is out of bounds

        # Top row: Ground truth
        axes[0, i].imshow(batch_y_np[t], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Target t={t}')

        # Bottom row: Prediction
        axes[1, i].imshow(pred_np[t], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Pred t={t}')

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


import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        # Sobel kernels for computing spatial gradients
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, gt):
        """
        pred, gt: tensors of shape [B, T, C, H, W]
        Computes L1 loss on approximate optical flow between consecutive frames.
        """
        B, T, C, H, W = pred.shape
        device = pred.device

        # Move Sobel kernels to device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        def compute_flow(frames):
            flows = []
            for t in range(T-1):
                I1 = frames[:, t]   # [B, C, H, W]
                I2 = frames[:, t+1]

                # Convert to grayscale if needed
                if I1.shape[1] == 3:
                    I1_gray = 0.2989*I1[:,0] + 0.5870*I1[:,1] + 0.1140*I1[:,2]
                    I2_gray = 0.2989*I2[:,0] + 0.5870*I2[:,1] + 0.1140*I2[:,2]
                else:
                    I1_gray = I1[:,0]
                    I2_gray = I2[:,0]

                I1_gray = I1_gray.unsqueeze(1)
                I2_gray = I2_gray.unsqueeze(1)

                # Compute spatial gradients
                Ix = F.conv2d(I1_gray, sobel_x, padding=1)
                Iy = F.conv2d(I1_gray, sobel_y, padding=1)

                # Temporal gradient
                It = I2_gray - I1_gray

                # Optical flow (simple least squares approximation)
                epsilon = 1e-6
                u = -Ix * It / (Ix**2 + Iy**2 + epsilon)
                v = -Iy * It / (Ix**2 + Iy**2 + epsilon)

                flow = torch.cat([u, v], dim=1)  # [B, 2, H, W]
                flows.append(flow)

            return torch.stack(flows, dim=1)  # [B, T-1, 2, H, W]

        pred_flow = compute_flow(pred)
        gt_flow   = compute_flow(gt)

        # Flatten time dimension for L1 loss
        B, T_minus1, C_flow, H, W = pred_flow.shape
        pred_flow = pred_flow.reshape(B * T_minus1, C_flow, H, W)
        gt_flow   = gt_flow.reshape(B * T_minus1, C_flow, H, W)

        return self.l1(pred_flow, gt_flow)

criterion = SSIMLoss()
criterion2 = TemporalLoss()
criterion3 = OpticalFlowLoss()



def check_static2(model, device, dataset):
    import numpy as np
    batch_x, batch_y, labels_x, labels_y, subset_no = next(iter(DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)))
    batch_x, labels_x, labels_y = batch_x.to(device), labels_x.to(device), labels_y.to(device)
    pred = model._predict(batch_x, labels_x, labels_y)
    
    loss1 = criterion(pred, batch_x)
    loss2 = criterion2(pred, batch_x)
    loss3 = criterion3(pred, batch_x)

    print(loss1.item(), loss2.item(), loss3.item())


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



    from openstl.datasets.dataloader_flappy import CustomDatasetAll
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Example usage
    from pathlib import Path
    _thisdir = Path(__file__).parent.parent.absolute()
    file_path = _thisdir / 'benchmarking'/ 'create_data' / 'fixed_gravity'

    dataset = CustomDatasetAll(file_path, 20, 10, transform=transform)


    check(model, 'cuda', dataset)
