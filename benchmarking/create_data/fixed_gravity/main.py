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
    return target_path, i


def write_frame(stored_frames, window, action, output_directory=None):
    if output_directory is not None:
        frame_array = surfarray.pixels3d(window)
        grayscale_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale_array, (84, 84))
        resized = resized.reshape(7056,)
        resized = np.append(resized, action)
        stored_frames.append(resized)
    return stored_frames


def write_positions(positions, idx):
    output_directory = Path(__file__).parent
    pos_path = output_directory / f"positions_{idx}.txt"
    with open(pos_path, 'w') as f:
        for pos in positions:
            f.write(f"{pos}\n")


def make(start_pos):
    stored_frames = []
    
    pygame.init()
    win = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Bird settings
    bird = pygame.Rect(WIDTH/2 - 15, start_pos, 30, 30)
    gravity = 0.25
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


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_movement = -6

        # === Maintain Jump State ===
        if jump_active and frame_count - jump_start_frame >= jump_frame_length:
            jump_active = False  # End jump

        action = 1 if jump_active else 0

        # Bird physics
        bird_movement += gravity
        bird.y += bird_movement

        positions.append(bird.y)

        if bird.bottom > HEIGHT:
            bird.bottom = HEIGHT
            bird_movement = 0
        if bird.top < 0:
            bird.top = 0
            bird_movement = 0

        pygame.draw.rect(win, (255, 255, 255), bird)

        frame_count += 1
        
        #clock.tick(60)

        output_directory = Path(__file__).parent

        stored_frames = write_frame(stored_frames, win, action, output_directory)
        if len(stored_frames) >= 50:
            array = np.stack(stored_frames)
            filename, idx = get_next_filename(output_directory)
            write_positions(positions, idx)
            np.save(filename, array)
            pygame.quit()
            break    


if __name__ == "__main__":
    start_positions = [(random.randint(0, HEIGHT),) for _ in range(500)]

    from multiprocessing import Pool

    #for start_position in start_positions:
    #    make(*start_position)

    with Pool() as pool:
        pool.starmap(make, start_positions)
    sys.exit()

