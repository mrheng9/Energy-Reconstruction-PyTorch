import os
import random  
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm  
from Preprocessed_Pytorch import cvngenerator
from Mobilenet_Regression import MobileNetRegression
from Googlenet_Regression import GoogLeNetRegression
from Resnet_Regression import ResNetRegression
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

def train(args):
    path = args.path
    name = args.name
    mode = args.mode
    model_type = args.model
    weighted = args.weighted
    epochs = args.epochs
    batch_size = args.batch_size
    input_shape = (2, 100, 80)
    learning_rate = 1e-4

    early_stop_patience = 10

    filenames = sorted(os.listdir(path))
    filenames = random.sample(filenames, len(filenames) // 8)  
    split = 4 * len(filenames) // 5
    train_files = filenames[:split]
    valid_files = filenames[split:]

    train_dataset = cvngenerator(path, train_files, input_shape=input_shape, weighted=weighted, mode=mode)
    valid_dataset = cvngenerator(path, valid_files, input_shape=input_shape, weighted=weighted, mode=mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=24,pin_memory=True)

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
    elif model_type == "resnet":
        model = ResNetRegression().to(device)
    else:
        raise ValueError("Invalid model type. Choose 'mobilenet' or 'googlenet'.")

    print("🧠 Model architecture:")
    print(model)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"🚀 Training started with learning rate = {learning_rate}, batch size = {batch_size}, epochs = {epochs}\n")
    if weighted:
        print("⚖️ Using weighted training")

    save_dir = "/home/houyh/nova/pytorch_version/models"
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    # Include the mode in the model filename
    model_name = f"{name}_{mode}"
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pt")
  
    scaler = GradScaler()  
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_steps = len(train_loader)

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
        
        for step, batch in enumerate(train_loader_tqdm):
            global_step += 1
            if weighted:
                inputs, targets, weights = batch
                weights = weights.to(device)
            else:
                inputs, targets = batch
            
            inputs = inputs.to(device)
            input_x = inputs[:, 0:1, :, :]
            input_y = inputs[:, 1:2, :, :]
            targets = targets.to(device).unsqueeze(1)

            with autocast():
                outputs = model(input_x, input_y)
                if weighted:
                    loss = (torch.abs(outputs - targets) * weights.unsqueeze(1)).mean()
                else:
                    loss = criterion(outputs, targets)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # 更新 tqdm 描述信息
            train_loader_tqdm.set_postfix({"Train Loss": loss.item()})
        
        print(f"✅ Epoch {epoch+1}/{epochs} finished. Total Train Loss: {running_loss:.4f}\n")

        # if global_step % validation_check_interval == 0:
        model.eval()
        val_loss = 0.0
        total_absolute_error = 0.0
        total_targets = 0.0

        # 使用 tqdm 包裹验证数据加载器
        valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)
        with torch.no_grad():
            for batch_val in valid_loader_tqdm:
                if weighted:
                    inputs_val, targets_val, weights_val = batch_val
                    weights_val = weights_val.to(device)
                else:
                    inputs_val, targets_val = batch_val
                    
                inputs_val = inputs_val.to(device)
                input_x_val = inputs_val[:, 0:1, :, :]
                input_y_val = inputs_val[:, 1:2, :, :]
                targets_val = targets_val.to(device).unsqueeze(1)

                outputs_val = model(input_x_val, input_y_val)
                if weighted:
                    loss_val = (torch.abs(outputs_val - targets_val) * weights_val.unsqueeze(1)).mean()
                else:
                    loss_val = criterion(outputs_val, targets_val)
                    
                val_loss += loss_val.item()
                total_absolute_error += torch.sum(torch.abs(outputs_val - targets_val)).item()
                total_targets += torch.sum(targets_val).item()

                # 更新 tqdm 描述信息
                valid_loader_tqdm.set_postfix({"Val Loss": loss_val.item()})

        accuracy = 1.0 - total_absolute_error / (total_targets + 1e-8)
        print(f"[Epoch {epoch+1}/{epochs} Step {step+1}/{total_steps}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")

        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f})")
            best_val_loss = val_loss
            patience_counter = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_best.pt"))
            print(f"💾 Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered! Training stopped.")
            break

        # if global_step % checkpoint_interval == 0:
        #     torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_checkpoint.pt"))
        #     print(f"\n💾 Saved checkpoint at step {global_step}\n")
        # Total Train Loss: {running_loss:.4f}\n
        print(f"✅ Epoch {epoch+1}/{epochs} finished. ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--name", type=str, required=True, help="a descriptive name for the model")
    parser.add_argument("--path", type=str, required=True, help="path to the folder containing h5 files")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["mobilenet", "googlenet","resnet"], help="which model to use: resnet,mobilenet or googlenet")
    parser.add_argument("--mode", type=str, default="nue", choices=["nue", "electron"], help="data mode: nue or electron")
    parser.add_argument("--weighted", action="store_true", help="use weighted training")
    args = parser.parse_args()
    train(args)