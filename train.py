import os
import random
import itertools
import argparse
from tqdm import tqdm

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.util import make_output_folder
from utils.dataset import ImageFolderWithoutTarget, ImageFolderWithPath
from utils.networks import AutoEncoder, PDN, load_weights
from utils.train_utils import calculate_loss, teacher_normalization, map_normalization, test

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

RESIZE_SIZE = 256 # fix
OUT_CHANNELS = 384

# Transform
default_transform = transforms.Compose([
transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
transforms.ToTensor(),
transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness = 0.2),
    transforms.ColorJitter(contrast = 0.2),
    transforms.ColorJitter(saturation = 0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))


def get_argparse():
    parser = argparse.ArgumentParser(description="Train EfficientAD")
    parser.add_argument("-d", "--dataset", default = "./dataset")
    parser.add_argument("-s", "--subdataset")
    parser.add_argument("-m", "--model_size", default = "small")
    parser.add_argument("-e", "--epochs", default = 200)

    return parser.parse_args()

def write_info_txt(config, txt_path):
    config_dict = vars(config)

    with open(txt_path, "w") as f:
        for k, v in config_dict.items():
            f.write(f"{k}: {v}\n")

def main():
    config = get_argparse()

    # Config
    model_size = config.model_size
    epochs = config.epochs
    dataset_path = config.dataset
    subdataset = config.subdataset

    # Make output dir & Write train info
    output_dir = os.path.join("./runs", subdataset, "train")
    output_dir = make_output_folder(output_dir)
    write_info_txt(config, os.path.join(output_dir, "train_info.txt"))

    # Dataset
    train_ds = ImageFolderWithoutTarget(
        os.path.join(dataset_path, subdataset, "train"),
        transform = transforms.Lambda(train_transform))

    val_ds = ImageFolderWithoutTarget(
        os.path.join(dataset_path, subdataset, "validation"),
        transform = transforms.Lambda(train_transform))

    test_ds = ImageFolderWithPath(
        os.path.join(dataset_path, subdataset, "test"))

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size = 1, shuffle = True, pin_memory = True)
    val_loader= DataLoader(val_ds, batch_size = 1)

    # Model
    teacher = PDN(model_size = model_size, out_channels = OUT_CHANNELS)
    teacher_weights = f"./models/teacher_{model_size}.pth"
    load_weights(teacher, teacher_weights)

    student = PDN(model_size = model_size, out_channels = 2 * OUT_CHANNELS)
    autoencoder = AutoEncoder()

    # Model frozen
    teacher.eval()
    student.eval()
    autoencoder.eval()

    teacher = teacher.to(device)
    student = student.to(device)
    autoencoder = autoencoder.to(device)

    # teacher normalization
    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader, device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                                 lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 5)

    # Train
    best_loss = np.inf
    loss_history = {"train_loss" : [], "val_loss": [], "auc": []}

    with tqdm(range(epochs), leave = False, dynamic_ncols=True) as pbar:
        for ep in pbar:
            
            # train
            teacher.eval()
            student.train()
            autoencoder.train()
        
            train_loss = 0
            for image_st, image_ae in train_loader:
                train_batch_loss = calculate_loss(teacher, student, autoencoder,
                                                image_st, image_ae, OUT_CHANNELS, 
                                                teacher_mean, teacher_std, device)
                
                optimizer.zero_grad()
                train_batch_loss.backward()
                optimizer.step()

                train_loss += train_batch_loss.item()
            
            train_loss /= len(train_loader)

            # val
            teacher.eval()
            student.eval()
            autoencoder.eval()

            val_loss = 0
            with torch.no_grad():
                for image_st, image_ae in val_loader:
                    val_batch_loss = calculate_loss(teacher, student, autoencoder,
                                            image_st, image_ae, OUT_CHANNELS, 
                                            teacher_mean, teacher_std, device)
                    val_loss += val_batch_loss.item()
                
                val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            
            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(val_loader, teacher, student, autoencoder,
                                                                           OUT_CHANNELS, teacher_mean, teacher_std, device)
            # test
            auc = test(test_ds, teacher, student, autoencoder,
                    default_transform, OUT_CHANNELS, teacher_mean, teacher_std,
                    q_st_start, q_st_end, q_ae_start, q_ae_end, device)

            # Save best model
            if val_loss < best_loss:
                
                best_loss = val_loss
                lr = scheduler.optimizer.param_groups[0]['lr']

                torch.save({"model": teacher, "epoch": ep + 1, "lr": lr}, os.path.join(output_dir, "teacher_best.pt"))
                torch.save({"model": student, "epoch": ep + 1, "lr": lr}, os.path.join(output_dir, "student_best.pt"))
                torch.save({"model": autoencoder, "epoch": ep + 1, "lr": lr}, os.path.join(output_dir, "autoencoder_best.pt"))

                torch.save(teacher_mean, os.path.join(output_dir, "teacher_mean.pt"))
                torch.save(teacher_std, os.path.join(output_dir, "teacher_std.pt"))

                torch.save(q_st_start, os.path.join(output_dir, "q_st_start_best.pt"))
                torch.save(q_st_end, os.path.join(output_dir, "q_st_end_best.pt"))
                torch.save(q_ae_start, os.path.join(output_dir, "q_ae_start_best.pt"))
                torch.save(q_ae_end, os.path.join(output_dir, "q_ae_end_best.pt"))
            
            # Save history
            loss_history["train_loss"].append(train_loss)
            loss_history["val_loss"].append(val_loss)
            loss_history["auc"].append(auc)
            torch.save(loss_history, os.path.join(output_dir, "loss_history.pt"))
            
            pbar.set_description(
                f"Epoch: {ep + 1}:   Current lr: {lr:.5f}     train loss: {train_loss:.7f}     val loss: {val_loss:.7f}  auc: {auc:.5f}      best val loss: {best_loss:.7f}"
            )
    print("End Train!")

if __name__ == "__main__":
    main()