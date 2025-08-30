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
    
from image_queue import Game
import pygame


def get_pos(frame, threshold=0.5):
    mask = (frame.squeeze(0) > threshold).float()
    if mask.sum() == 0:
        return 0.0
    coords = torch.nonzero(mask)
    y_mean = coords[:, 0].float().mean()
    return y_mean.item()


def extract_position(frame):
    print(get_pos(torch.tensor(frame)))


def extract_positions(model, device, dataset):

    game = Game(device, dataset, model)
    # Initialize Pygame
    pygame.init()

    font = pygame.font.SysFont(None, 48)
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
            extract_position(game.queue.get_current_image())
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

        # Update the display
        pygame.display.update()

        frame_count += 1

    # Quit Pygame
    pygame.quit()



        

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
    dataset = CustomDatasetAll("E:/python_home/flappy_bird", 20, 10, transform=transform)

    extract_positions(model, 'cuda', dataset)
