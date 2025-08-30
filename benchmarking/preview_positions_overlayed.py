import torch
import pandas as pd

def get_pos(frame, threshold=0.5):
    frame = frame.T
    mask = (frame > threshold).float()
    if mask.sum() == 0:
        return 0.0
    coords = torch.nonzero(mask)
    y_mean = coords[:, 0].float().mean()
    return y_mean.item()

def animate_frames(frames):
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap='gray', animated=True)
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    ani = FuncAnimation(
        fig, update, frames=len(frames), interval=20, blit=True, repeat=True
    )
    plt.show()

def extract_positions(dataloader, target_subset=None):
    if target_subset is None:
        current_subset = None
        batches = []
        for batch_x, batch_y, batch_label_x, batch_label_y, subset_no in dataloader:
            if subset_no.item() != current_subset and current_subset is not None:
                break
            else:
                current_subset = subset_no.item()
                combined = torch.cat([batch_x, batch_y], dim=1)
                batches.append(combined)
        return torch.cat(batches, dim=1)  # Concatenate along the sequence dimension
    batches = []
    for batch_x, batch_y, batch_label_x, batch_label_y, subset_no in dataloader:
        if target_subset == subset_no:
            current_subset = subset_no.item()
            combined = torch.cat([batch_x, batch_y], dim=1)
            batches.append(combined)
    return torch.cat(batches, dim=1)  # Concatenate along the sequence dimension

    
def get_raw_positions(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()  
    positions = [float(line.strip()) for line in lines]
    return pd.DataFrame(positions, columns=['raw_position'])


def normalize_to_first_last(col):
    first = col.iloc[0]
    last = col.iloc[-1]
    return (col - first) / (last - first)


if __name__ == '__main__':
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from openstl.datasets.dataloader_flappy import load_data
    import numpy as np

    import pathlib
    _thisdir = pathlib.Path(__file__).parent.absolute()
    data_root = _thisdir / 'create_data' / 'fixed_gravity'
    path = data_root / "Data-141.npy"
    data = np.load(path)

    raw_path = data_root / "positions_141.txt"
    df_raw = get_raw_positions(raw_path)

    image_frames = data[:, :-1].reshape(-1, 84, 84).astype(np.uint8) / 255

    #animate_frames(image_frames)    

    positions = [get_pos(frame) for frame in torch.tensor(image_frames)]
    df = pd.DataFrame(positions, columns=['position'])
    df = pd.concat([df, df_raw], axis=1)
    
    df['raw_position'] = normalize_to_first_last(df['raw_position'])
    df['position'] = normalize_to_first_last(df['position'])

    sns.lineplot(data=df, x=df.index, y='raw_position', label='Raw Pygame Position')
    sns.lineplot(data=df, x=df.index, y='position', label='Semantically Detected Position')
    plt.title('Position Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Normalised Position')
    plt.legend()
    plt.show()
