import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Preprocessed_Pytorch import cvngenerator
from Mobilenet_Regression import MobileNetRegression
from Googlenet_Regression import GoogLeNetRegression

def train(args):
    path = args.path
    name = args.name
    model_type = args.model
    input_shape = (2, 100, 80)
    batch_size = 128
    epochs = 100
    learning_rate = 1e-4
    checkpoint_interval = 500
    validation_check_interval = 500
    early_stop_patience = 10000000

    filenames = sorted(os.listdir(path))
    split = 4 * len(filenames) // 5
    train_files = filenames[:split]
    valid_files = filenames[split:]

    train_dataset = cvngenerator(path, train_files, input_shape=input_shape)
    valid_dataset = cvngenerator(path, valid_files, input_shape=input_shape)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ CUDA is available!")
        print("   Device count:", torch.cuda.device_count())
        print("   Using device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available. Using CPU.")

    if model_type == "mobilenet":
        model = MobileNetRegression().to(device)
    elif model_type == "googlenet":
        model = GoogLeNetRegression().to(device)
    else:
        raise ValueError("Invalid model type. Choose 'mobilenet' or 'googlenet'.")

    print("🧠 Model architecture:")
    print(model)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"🚀 Training started with learning rate = {learning_rate}, batch size = {batch_size}, epochs = {epochs}\n")

    save_dir = "/home/zhongyi/nova/pytorch_version/models"
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_steps = len(train_loader)

        for step, (inputs, targets) in enumerate(train_loader):
            global_step += 1
            inputs = inputs.to(device)
            input_x = inputs[:, 0:1, :, :]
            input_y = inputs[:, 1:2, :, :]
            targets = targets.to(device).unsqueeze(1)

            outputs = model(input_x, input_y)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if global_step % validation_check_interval == 0:
                model.eval()
                val_loss = 0.0
                total_absolute_error = 0.0
                total_targets = 0.0
                with torch.no_grad():
                    for inputs_val, targets_val in valid_loader:
                        inputs_val = inputs_val.to(device)
                        input_x_val = inputs_val[:, 0:1, :, :]
                        input_y_val = inputs_val[:, 1:2, :, :]
                        targets_val = targets_val.to(device).unsqueeze(1)

                        outputs_val = model(input_x_val, input_y_val)
                        loss_val = criterion(outputs_val, targets_val)
                        val_loss += loss_val.item()
                        total_absolute_error += torch.sum(torch.abs(outputs_val - targets_val)).item()
                        total_targets += torch.sum(targets_val).item()

                accuracy = 1.0 - total_absolute_error / (total_targets + 1e-8)
                print(f"[Epoch {epoch+1}/{epochs} Step {step+1}/{total_steps}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")

                if val_loss < best_val_loss:
                    print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f})")
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement in validation loss. Patience: {patience_counter}/{early_stop_patience}")

                if patience_counter >= early_stop_patience:
                    print("Early stopping triggered! Training stopped.")
                    return

            if global_step % checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_checkpoint.pt"))
                print(f"\n💾 Saved checkpoint at step {global_step}\n")

        print(f"✅ Epoch {epoch+1}/{epochs} finished. Total Train Loss: {running_loss:.4f}\n")

    final_model_path = os.path.join(save_dir, f"{name}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved to {final_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="a descriptive name for the model")
    parser.add_argument("--path", type=str, required=True, help="path to the folder containing h5 files")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["mobilenet", "googlenet"], help="which model to use: mobilenet or googlenet")
    args = parser.parse_args()
    train(args)
