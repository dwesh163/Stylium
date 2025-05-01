"""
Module for training the style transfer model.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision import utils as vutils
from model import ContentLoss, StyleLoss  # Assuming these are used elsewhere in the model

def train_model(
    model, train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 10, learning_rate: float = 0.001,
    content_weight: float = 1.0, style_weight: float = 1000.0
):
    """
    Train the style transfer model.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        content_weight: Weight for content loss.
        style_weight: Weight for style loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    content_criterion = nn.MSELoss().to(device)
    style_criterion = nn.MSELoss().to(device)

    train_losses, val_losses = [], []

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

        for i, (content_images, _) in progress_bar:
            content_images = content_images.to(device)
            batch_size = content_images.size(0)
            style_indices = torch.randint(0, len(model.styles), (batch_size,))

            optimizer.zero_grad()
            batch_loss = 0.0

            for j in range(batch_size):
                content_img = content_images[j:j+1]
                style_idx = style_indices[j].item()

                output = model(content_img, style_idx)
                content_loss = content_criterion(output, content_img)
                style_loss = torch.tensor(0.0).to(device)  # Placeholder

                loss = content_weight * content_loss + style_weight * style_loss
                batch_loss += loss.item()
                loss.backward()

            optimizer.step()
            avg_loss = batch_loss / batch_size
            total_train_loss += avg_loss
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        avg_epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_epoch_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for content_images, _ in val_loader:
                content_images = content_images.to(device)
                batch_size = content_images.size(0)
                style_indices = torch.randint(0, len(model.styles), (batch_size,))
                batch_loss = 0.0

                for j in range(batch_size):
                    content_img = content_images[j:j+1]
                    style_idx = style_indices[j].item()
                    output = model(content_img, style_idx)

                    content_loss = content_criterion(output, content_img)
                    style_loss = torch.tensor(0.0).to(device)  # Placeholder

                    loss = content_weight * content_loss + style_weight * style_loss
                    batch_loss += loss.item()

                total_val_loss += batch_loss / batch_size

        avg_epoch_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_epoch_val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_epoch_train_loss:.4f}, "
              f"Val Loss: {avg_epoch_val_loss:.4f}, "
              f"Time: {elapsed:.2f}s")

        # Visualization
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                val_images = next(iter(val_loader))[0][:4].to(device)
                output_batches = []

                for style_idx in range(min(4, len(model.styles))):
                    styled_images = [model(img.unsqueeze(0), style_idx) for img in val_images]
                    output_batches.append(torch.cat(styled_images, 0))

                all_outputs = torch.cat(output_batches, 0)
                grid = vutils.make_grid(all_outputs, nrow=4, normalize=True)
                img = grid.cpu().numpy().transpose((1, 2, 0))
                img = np.clip(img, 0, 1)

                plt.figure(figsize=(15, 15))
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(f'results/epoch_{epoch+1}.png')
                plt.close()

        # Save model periodically
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.save(f'models/style_transfer_model_epoch_{epoch+1}.pth')

    # Final model save
    model.save()

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig('results/loss_curves.png')
    plt.close()

    print("Training complete.")
    return model
