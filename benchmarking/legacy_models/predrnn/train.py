import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import torch

'''
    Added fusion gate. Not much improvement.
 '''

def display_image(image_array):
    image = Image.fromarray(np.uint8(255.0 * image_array), mode='L')
    image.show()


def display_images(image_array1, image_array2):
    img1 = Image.fromarray(np.uint8(255.0 * image_array1), mode='L')
    img2 = Image.fromarray(np.uint8(255.0 * image_array2), mode='L')

    # Create a new image with width = sum of both widths and height = max of both heights
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)

    # Create a new blank image with white background
    new_img = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    # Paste the images
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    # Show or save the result
    new_img.show()

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


class Game:
    def __init__(self, device, queue, model):
        self.device = device
        self.queue = queue
        self.model = model

    def next(self, keys):
        images, labels = self.queue.get_images()
        action = get_action(keys)
        labels = np.append(labels, action)

        images = torch.tensor(images).to(self.device).unsqueeze(1).unsqueeze(0)
        labels = torch.tensor(labels).to(self.device)

        prediction = self.model(images, labels)

        image = prediction.detach().cpu().numpy().squeeze(0).squeeze(0)
        action = np.array([action])

        self.queue.add_image(image, action)

        return np.uint8(255.0 * image.squeeze(0))




from image_queue import GameQueue
import pygame

def image_array_to_pygame_surface(image_array):
    pil_image = Image.fromarray(image_array)
    scaled_image = pil_image.resize((500, 500), 
                    Image.Resampling.LANCZOS).convert("RGB")
    return pygame.image.frombuffer(scaled_image.tobytes(), 
                                   scaled_image.size, 
                                   scaled_image.mode)


def check(model, device, dataset):

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    batch_x, batch_y, label_x, label_y, subset_no = next(iter(dataloader))
    first_three_images = batch_x.squeeze(0).numpy()[:3].squeeze(1)
    first_three_labels = batch_y.squeeze(0).numpy()[:3]

    game_queue = GameQueue()

    game = Game(device, game_queue, model)

    # Initialize Pygame
    pygame.init()
    # Set up display
    WIDTH, HEIGHT = 500, 500
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    running = True
    while running:
        if pygame.key.get_pressed()[pygame.K_SPACE]:
            input_images, input_labels = next(iter(dataloader))
            first_three_images = input_images.squeeze(0).numpy()[:3].squeeze(1)
            first_three_labels = input_labels.squeeze(0).numpy()[:3]
            game_queue.add_image(first_three_images, first_three_labels)
        pygame.time.delay(60)
        image_array = game.next(pygame.key.get_pressed())
        print(image_array.shape, image_array.dtype)
        pygame_surface = image_array_to_pygame_surface(image_array)
        screen.blit(pygame_surface, (0, 0))    
        # Update the display
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Quit Pygame
    pygame.quit()


def validation_loss(dataloader, loss_fn, model, device):
    total_loss = 0
    for batch_x, batch_y, label_x, label_y, subset_no in dataloader:
        batch_x = batch_x.to(device)  # First 3 frames
        batch_y = batch_y.to(device)  # 4th frame
        label_x = label_x.to(device)  # Labels for conditioning
        label_y = label_y.to(device)  # Labels for conditioning
        outputs = model(batch_x, label_x, label_y)
        loss = loss_fn(outputs, batch_y)
        total_loss += loss.item()
    print('Validation Loss: {:.4f}'.format(total_loss / len(dataloader)))
    return total_loss / len(dataloader)

# Training loop
def train_model(model, loss_fn, dataloader, vali_loader, epochs=10, device="cuda"):
    model.to(device)
    model.train()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001)

    with open("vali_loss.txt", "w") as f:

        for epoch in range(epochs):
            total_loss = 0

            for batch_x, batch_y, label_x, label_y, subset_no in dataloader:
                batch_x = batch_x.to(device)  # First 3 frames
                batch_y = batch_y.to(device)  # 4th frame
                label_x = label_x.to(device)  # Labels for conditioning
                label_y = label_y.to(device)  # Labels for conditioning

                optimizer.zero_grad()  # Reset gradients

                # Forward pass (tracking gradients!)
                outputs = model(batch_x, label_x, label_y)

                # Compute loss
                loss = loss_fn(outputs, batch_y)
                total_loss = total_loss + loss.item()

                # Backward + optimize
                loss.backward()
                optimizer.step()                  

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

            #val_loss = validation_loss(vali_loader, loss_fn, model, device)
            #f.write(str(val_loss) + '\n')

            scheduler.step(avg_loss)  # Adjust LR based on validation loss

        print("Training complete!")


from model19 import PredRNNPredictor


if __name__ == '__main__':
    
    file_path = "E:/python_home/cscst/Data/Data-1.npy"  # Path to your numpy file

    batch_size = 1

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    from openstl.datasets.dataloader_flappy import load_data

    # Example usage
    train_data, vali_data, test_data = load_data(batch_size, batch_size, "E:/python_home/flappy_bird", num_workers=0)


#    model = MobileViTModel()
    model = PredRNNPredictor()

    model.train()


    # for name, param in model.named_parameters():
    #     print(name, param.grad)

    # Define loss function
    
    from torch import nn
    loss_fn = nn.MSELoss()
    
    # Example usage:

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Example usage:
    train_model(model, loss_fn, train_data, test_data, epochs=30)

    torch.save(model.state_dict(), 'predrnn_model.pth')
    #check(model, 'cuda', test_data.dataset)

    