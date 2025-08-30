import torch
from torch.utils.data import random_split, Dataset, Subset
from pathlib import Path, PurePath
import numpy as np
from openstl.datasets.utils import create_loader
import torchvision.transforms as transforms


def partition_list(n, partition_size=5, overlap=0):
    step = partition_size - overlap
    return [list(range(i, i + partition_size))
        for i in range(0, n - partition_size + 1, step)]

class CustomDataset(Dataset):
    def __init__(self, file_path, pre_seq_length, after_seq_length, transform=None, overlap=0):
        classified_images = np.load(file_path)
        self.pre_seq_length = pre_seq_length
        self.image_list = classified_images[:, :-1].reshape(-1, 84, 84).astype(np.uint8)
        self.after_seq_length = after_seq_length
        self.labels = classified_images[:, -1].astype(int)
        self.transform = transform

        image_count = pre_seq_length + after_seq_length
        self.index_list = partition_list(len(self.image_list), image_count, overlap)


    def __len__(self):
        return len(self.index_list) # Each set consists of image_count images

    def __getitem__(self, idx):
        indexes = self.index_list[idx]

        images = [self.transform(self.image_list[i].T) if self.transform else 
            torch.tensor(self.image_list[i].T, dtype=torch.float32)
            for i in indexes]

        labels = [
            torch.tensor(self.labels[i])
            for i in indexes]

        input_images = torch.stack(images[:self.pre_seq_length])
        output_images = torch.stack(images[self.pre_seq_length:])
        input_labels = torch.stack(labels[:self.pre_seq_length])
        output_labels = torch.stack(labels[self.pre_seq_length:])

        return input_images, output_images, input_labels, output_labels

import numpy as np
from pathlib import Path
import re

def find_list_containing_element(lengths, target_index):
    current_index = 0
    for i, length in enumerate(lengths):
        if current_index + length > target_index:
            return i, target_index - current_index  # Dataset index, element offset within it
        current_index += length
    raise IndexError(f"Target index {target_index} out of range (max {current_index - 1})")

def get_metadata(filename):
    name, gravity, jump_size = re.sub(r"\(\d+\)", "", filename.name.replace('.npy', '')).split('_')
    return float(gravity), int(jump_size)

class CustomDatasetAll(Dataset):
    def __init__(self, dir_path, pre_seq_length, aft_seq_length, overlap=0, transform=None):
        self.dir_path = Path(dir_path)
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        assert pre_seq_length + aft_seq_length > overlap
        self.overlap = overlap
        self.transform = transform

        self.npy_files = list(self.dir_path.glob("*.npy"))

        # Store lengths instead of full datasets
        self.lengths = []
        for file_path in self.npy_files:
            temp_dataset = CustomDataset(file_path, pre_seq_length, aft_seq_length, transform, overlap=overlap)
            self.lengths.append(len(temp_dataset))

        # Optional caching (single dataset reuse)
        self._last_index = None
        self._last_dataset = None

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        list_index, element_index = find_list_containing_element(self.lengths, idx)

        # Load only the relevant dataset (reuse if cached)
        if list_index != self._last_index:
            file_path = self.npy_files[list_index]
            self._last_dataset = CustomDataset(file_path, self.pre_seq_length, self.aft_seq_length, self.transform, overlap=self.overlap)
            self._last_index = list_index

        #gravity, jump_size = get_metadata(self.npy_files[list_index])
        return *self._last_dataset[element_index], list_index



class CustomSubset(Subset):
    def __init__(self, dataset, start, end):
        start_index = sum(dataset.lengths[:start])
        end_index = sum(dataset.lengths[:end])
        super().__init__(dataset, range(start_index, end_index))


def load_data(batch_size, val_batch_size, data_root, num_workers=4, data_name='flappy',
              pre_seq_length=10, aft_seq_length=10, overlap=0, in_shape=[1, 4, 84, 84],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CustomDatasetAll(data_root, pre_seq_length, aft_seq_length, transform=transform)
    print(data_root, pre_seq_length, aft_seq_length, overlap, len(dataset))


    num_datasets = len(dataset.lengths)
    train_size = int(0.8 * num_datasets)

    train_set = CustomSubset(dataset, 0, train_size)
    test_set = CustomSubset(dataset, train_size, num_datasets)

    
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=True, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=True, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test



if __name__ == '__main__':

    # train_set, vali_set, test_set = load_data(1, 1, None)

    # batch = next(iter(train_set))

    # x, y, label_x, label_y = batch

    # print(x.min(), x.max(), x.dtype)

    #dataset = CustomDatasetAll("E:/python_home/flappy_bird", 40, 4, overlap=42)
    
    load_data(1, 1, "E:/python_home/flappy_bird", pre_seq_length=40, aft_seq_length=4, overlap=42, use_prefetcher=False)
    

   
    #train_size = int(0.8 * num_datasets)

    #train_set = CustomSubset(dataset, 0, train_size)
    #test_set = CustomSubset(dataset, train_size, num_datasets)
