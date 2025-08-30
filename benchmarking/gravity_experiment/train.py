import torch
import torch.nn as nn
import torch.optim as optim
from openstl.datasets.dataloader_flappy import load_data
from model import GravityTransformer
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import re


def train(config, dataloader, device):

    model = GravityTransformer(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        current_subset = None
        for x_in, x_out, _, _, subset, _ in dataloader:
            x_in = x_in.to(device)
            if current_subset != subset:
                model.reset_memory()           
            pred, _, _ = model(x_in)  # [B, aft_seq, C, H, W]
            # Losses
            loss = criterion(pred, x_out.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(total_loss / len(dataloader))

        model.reset_memory()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
    return model


def collect_latents(model, dataloaders, device, seq_len=15):
    model.eval()
    all_latents = {}
    all_trues = {}
    count = 0

    with torch.no_grad():
        for dataloader in dataloaders:
            for x_in, _, _, _, subset_no, metadata in dataloader:
                gravity, _ = metadata
                x_in = x_in.to(device)
                _, _, latent_gravity = model(x_in)  # shape e.g. [B, seq_len, feature_dim]
                
                #latent_gravity = latent_gravity.unsqueeze(1)[:, :seq_len, :]

                latent_gravity = latent_gravity.reshape(latent_gravity.size(0), -1).cpu()
                all_latents[subset_no.item()] = latent_gravity # only stores the last one
                all_trues[subset_no.item()] = gravity.cpu() # only stores the last one

    all_latents = torch.cat(tuple(all_latents.values()), dim=0)
    all_trues = torch.cat(tuple(all_trues.values()), dim=0)
    return all_latents, all_trues


def gravity_check(model, dataloaders, device, seq_len=15, plot=True, n_components=2):
    # Collect latent representations and gravity labels
    latents, gravities = collect_latents(model, dataloaders, device, seq_len=seq_len)
        
    # Convert to numpy
    X = latents.numpy()
    y = gravities.numpy().reshape(-1)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Explained variance
    explained = pca.explained_variance_ratio_
    print(f"PCA explained variance (first {n_components} components): {explained}")

    if plot and n_components >= 2:
        plt.figure(figsize=(6,6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA of Latents Colored by Gravity")
        plt.grid(True)
        plt.colorbar(scatter, label='Gravity')
        plt.show()

    return X_pca, explained, y

def get_metadata(filename):
    name, gravity, jump_size = re.sub(r"\(\d+\)", "", filename.replace('.npy', '')).split('_')
    return float(gravity), int(jump_size)

if __name__ == "__main__":
    model_config = {
        # image h w c
        'height': 84,
        'width': 84,
        'num_channels': 1,
        # video length in and out
        'pre_seq': 5,
        'aft_seq': 15,
        # patch size
        'patch_size': 21,
        'dim': 224, 
        'heads': 8,
        'dim_head': 32,
        # dropout
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        # depth
        'depth': 1,
        'Ndepth': 6, # For FullAttention-24, for BinaryST, BinaryST, FacST, FacTS-12, for TST,STS-8, for TSST, STTS-6
        'data_dir':  "benchmarking/create_data/variable_gravity",
        'batch_size': 1,
        'val_batch_size': 1,
        'lr': 1e-3,
        'epochs': 50,
        'slot_dim': 224,
        'num_slots': 16,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, _ = load_data(1, 1, model_config['data_dir'], 
                                        pre_seq_length=5, aft_seq_length=15,
                                        filename_parser=get_metadata,)
    model = train(model_config, train_loader, device)
    gravity_check(model, [train_loader, test_loader], device)

