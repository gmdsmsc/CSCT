import numpy as np
from torch.utils.data import DataLoader
import pygame
import torch
from PIL import Image
from collections import deque



def get_action(keys):
    up_pressed = keys[pygame.K_UP]
    down_pressed = keys[pygame.K_DOWN]
    if up_pressed and down_pressed:
        return 0
    elif up_pressed:
        return 1
    elif down_pressed:
        return 2 
    return 0

def image_array_to_pygame_surface_old(image_array):
    print(image_array.dtype)
    image_array = np.uint8(255 * image_array)
    pil_image = Image.fromarray(image_array)
    scaled_image = pil_image.resize((500, 500), 
                    Image.NEAREST).convert("RGB")
    return pygame.image.frombuffer(scaled_image.tobytes(), 
                                   scaled_image.size, 
                                   scaled_image.mode)


def image_array_to_pygame_surface_old2(image_array):
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    image_array = np.uint8(255 * image_array)
    pil_image = Image.fromarray(image_array)
    scaled_image = pil_image.resize((500, 500), 
                    Image.LANCZOS).convert("RGB")
    return pygame.image.frombuffer(scaled_image.tobytes(), 
                                   scaled_image.size, 
                                   scaled_image.mode)


def resize_nearest(image_array, target_size=(500, 500)):
    original_h, original_w = image_array.shape[:2]
    target_h, target_w = target_size

    # Compute scale factor
    scale_y = original_h / target_h
    scale_x = original_w / target_w

    # Use nearest-neighbor sampling
    indices_y = (np.arange(target_h) * scale_y).astype(int)
    indices_x = (np.arange(target_w) * scale_x).astype(int)

    resized = image_array[indices_y[:, None], indices_x]
    return resized

def image_array_to_pygame_surface(image_array):
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    image_array = np.uint8(255 * image_array)

    # Ensure image has 3 channels (RGB)
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    resized = resize_nearest(image_array)

    return pygame.image.frombuffer(resized.tobytes(), resized.shape[1::-1], "RGB")


def gen_images(iterator):
    for batch_x, batch_y, label_x, label_y, subset_no in iterator:
        
        images = batch_x.squeeze(0).numpy()[:3].squeeze(1)
        labels = label_x.squeeze(0).numpy()[:3]
        
        for i, image in enumerate(images):
            action = labels[i]
            yield image, action


class GameQueue:
    def __init__(self):
        self.queue = deque([], maxlen=10)

    def add_image(self, input_image, input_label):
        paired_input = input_image, input_label
        self.queue.append(paired_input)

    def get_images(self):
        images, labels = list(zip(*self.queue))
        images, labels = np.array(images), np.array(labels)
        return images, labels
    
    def get_current_image(self):
        images, labels = list(zip(*self.queue))
        return np.array(images[-1])
    
    def clear(self):
        self.queue.clear()

    def get_queue_size(self):
        return len(self.queue)

    def clear_queue(self):
        self.queue.clear()

    def queue_image(self, image, action):
        self.add_image(image, np.array([action]))
   


class Game:
    def __init__(self, device, dataset, model):
        self.device = device
        self.model = model
        self.dataset = dataset       
        self.image_generator = gen_images(iter(DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)))
        self.queue = GameQueue()

    def reset(self):
        self.queue.clear_queue()
        self.make_next(None)

    def make_next(self, keys):
        if self.queue.get_queue_size() == 10:
            image, action = self.gen_next(keys)
        else:
            image, action = next(self.image_generator)
        self.queue.queue_image(image, action)

    def gen_next(self, keys):
        action = get_action(keys)
        actions = np.array([[action]])

        images, labels = self.queue.get_images()
        images = torch.tensor(images).to(self.device).unsqueeze(1).unsqueeze(0)
        labels = torch.tensor(labels.T).to(self.device)
        actions = torch.tensor(actions).to(self.device).repeat(1, 10)
        
        prediction = self.model._predict(images, labels, actions)
        prediction = prediction[:, :1, :]
        image = prediction.detach().cpu().numpy().squeeze(0).squeeze(0).squeeze(0)
        return image, action

    def get_current(self):
        image = self.queue.get_current_image()
        return image_array_to_pygame_surface(image)


