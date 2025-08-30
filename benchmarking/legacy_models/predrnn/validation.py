

from pytorch_msssim import ssim
from torch import nn

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

def validate(model, device, dataloader):
    loss1_all = []
    loss2_all = []
    loss3_all = []
    loss4_all = []
    loss5_all = []
    loss6_all = []
    for batch_x, batch_y, labels_x, labels_y, subset_no in dataloader:
        import numpy as np
        batch_x = batch_x.to(device)
        labels_x = labels_x.to(device)
        labels_y = labels_y.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x, labels_x, labels_y)
        
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
    from model import PredRNNPredictor
    device = "cuda"
    model = PredRNNPredictor().to(device)
    model.load_state_dict(torch.load("predrnn_model.pth"))

    from openstl.datasets.dataloader_flappy import load_data

    # Example usage
    train_data, vali_data, test_data = load_data(1, 1, "./data", num_workers=0)


    validate(model, device, test_data)