import torch
import numpy as np
import pickle
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
from Preprocessed_Pytorch import cvngenerator
from Mobilenet_Regression import MobileNetRegression
from Googlenet_Regression import GoogLeNetRegression
from Resnet_Regression import ResNetRegression
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Load model
    if args.model == "mobilenet":
        model = MobileNetRegression().to(device)
    elif args.model == "googlenet":
        model = GoogLeNetRegression().to(device)
    elif args.model == "resnet":
        model = ResNetRegression().to(device)
    else:
        raise ValueError("Invalid model type. Choose 'mobilenet' or 'googlenet'")

    checkpoint = torch.load(args.modelpath, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ Loaded model ({args.model}) from {args.modelpath}")

    # Load data
    file_list = sorted(os.listdir(args.path))
    dataset = cvngenerator(args.path, file_list, input_shape=(2, 100, 80))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    args.steps = len(loader)

    y_true, y_pred = [], []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(loader):
            if step >= args.steps:
                break
            print(f"[Step {step+1}/{args.steps}] Running inference...")
            inputs = inputs.to(device)
            x = inputs[:, 0:1, :, :]
            y = inputs[:, 1:2, :, :]
            targets = targets.cpu().numpy()
            outputs = model(x, y).cpu().numpy()
            y_true.append(targets)
            y_pred.append(outputs)

    y_arr = np.concatenate(y_true).flatten()
    yhat_arr = np.concatenate(y_pred).flatten()
    resolution = yhat_arr / y_arr
    iqr = np.percentile(resolution, 75) - np.percentile(resolution, 25)
    print(f"📊 Resolution IQR: {iqr:.4f}")

    # Save predictions
    output = {
        "y": y_arr,
        "yhat": yhat_arr,
        "resolution": resolution
    }

    save_path = f"/home/houyh/nova/pytorch_version/pkl_file/{args.name}_predictions.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(output, f)

    print(f"💾 Saved prediction results to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, required=True, help="Path to trained PyTorch model (.pt)")
    parser.add_argument("--path", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--name", type=str, default="test2")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["mobilenet", "googlenet","resnet"], help="Model type: mobilenet or googlenet")
    args = parser.parse_args()

    evaluate(args)
