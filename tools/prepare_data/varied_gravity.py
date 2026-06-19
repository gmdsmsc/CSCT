import pygame
import random
import sys
import pygame.surfarray as surfarray
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import time

random.seed(42)

WIDTH, HEIGHT = 480, 480

def get_next_filename(output_directory):
    output_path = Path(output_directory)
    i = 0
    target_path = output_path / f"Data-{i}.npy"
    while target_path.exists():
        i += 1
        target_path = output_path / f"Data-{i}.npy"
    return target_path

def get_filename_next_grav_jump(output_directory, gravity, jump_size):
    output_path = Path(output_directory)
    i = 0
    target_path = output_path / f"Data_{gravity}_{jump_size}({i}).npy"
    while target_path.exists():
        i += 1
        target_path = output_path / f"Data_{gravity}_{jump_size}({i}).npy"
    return target_path

def get_filename(output_directory, gravity, jump_size):
    output_path = Path(output_directory)
    return output_path / f"Data_{gravity}_{jump_size}.npy"


def write_frame(stored_frames, window, action, output_directory=None):
    if output_directory is not None:
        frame_array = surfarray.pixels3d(window)
        grayscale_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale_array, (84, 84))
        resized = resized.reshape(7056,)
        resized = np.append(resized, action)
        stored_frames.append(resized)
    return stored_frames

def write_positions(positions, name):
    output_directory = Path(__file__).parent
    pos_path = output_directory / name.replace('.npy', '.txt')
    with open(pos_path, 'w') as f:
        for pos in positions:
            f.write(f"{pos}\n")

def make(gravity, start_position):
    stored_frames = []
    
    pygame.init()
    win = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Bird settings
    bird = pygame.Rect(WIDTH/2 - 15, 30 + start_position, 30, 30)
    bird_movement = 0

    frame_count = 0
    running = True

    def gaussian_int_0_20(mu=10, sigma=4):
        while True:
            val = round(random.gauss(mu, sigma))
            yield max(0, min(20, val))  # clip to [0, 20]

    gen = gaussian_int_0_20()

    # Jump simulation variables
    jump_active = False

    positions = []
    while running:
        win.fill((0, 0, 0))  # Sky blue background

        # === Random Chance to Trigger Jump ===
        if not jump_active and random.random() < 0.06:  # ~1% chance per frame
            sim_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})
            pygame.event.post(sim_event)

            jump_active = True
            jump_start_frame = frame_count
            jump_frame_length = next(gen)


        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        #         bird_movement = -jump_size

        # === Maintain Jump State ===
        if jump_active and frame_count - jump_start_frame >= jump_frame_length:
            jump_active = False  # End jump

        action = 1 if jump_active else 0

        # Bird physics
        bird_movement += gravity
        bird.y += bird_movement

        positions.append(bird.y)

        if bird.bottom > HEIGHT:
            break
            bird.bottom = HEIGHT
            bird_movement = 0
        if bird.top < 0:
            bird.top = 0
            bird_movement = 0

        pygame.draw.rect(win, (255, 255, 255), bird)

        frame_count += 1
        
        #clock.tick(60)

        output_directory = ''
        stored_frames = write_frame(stored_frames, win, action, output_directory)
        if len(stored_frames) >= 200:
            break    

    array = np.stack(stored_frames)
    target_path = get_filename_next_grav_jump(output_directory, gravity, 0)
    np.save(target_path, array)
    write_positions(positions, target_path.name)

    pygame.quit()


if __name__ == "__main__":
    from itertools import product

    gravities = [i / 10 for i in range(5,20)]
    start_positions = list(range(-2, 2))

    from multiprocessing import Pool

    combinations = list(product(gravities, start_positions))
    repeated_combinations = [item for item in combinations]


    #for start_position in start_positions:
    #    make(*start_position)

    with Pool() as pool:
        pool.starmap(make, repeated_combinations)
    sys.exit()

