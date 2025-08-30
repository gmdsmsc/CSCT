import torch
import pandas as pd
import random
random.seed(42)

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
    positions = [-1 * float(line.strip()) for line in lines]
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
    data_root = _thisdir / 'create_data' / 'varied_gravity'

    position_files = list(data_root.glob("*.txt"))

    sample_pos_files = random.sample(position_files, 20)

    dataframes = []
    for pos_file in sample_pos_files:
        df_raw = get_raw_positions(pos_file)
        dataframes.append(df_raw)

    df = pd.concat(dataframes, axis=1)
    df.columns = [f"value_{i}" for i in range(df.shape[1])]

    df_melted = df.reset_index().melt(id_vars='index', var_name='Variable', value_name='Value')
    
    sns.lineplot(data=df_melted, x='index', y='Value', hue='Variable')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title("Line Plot of All Columns (Melted)")
    plt.legend().remove()
    plt.show()
