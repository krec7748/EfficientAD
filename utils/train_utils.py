import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torchvision import transforms

def train_transform(image, resize_size):
    # Transform
    default_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    transform_ae = transforms.RandomChoice([
        transforms.ColorJitter(brightness = 0.2),
        transforms.ColorJitter(contrast = 0.2),
        transforms.ColorJitter(saturation = 0.2)
    ])
    return default_transform(image), default_transform(transform_ae(image))

def calculate_loss(teacher, student, autoencoder,
                   image_st, image_ae, out_channels, 
                   teacher_mean, teacher_std ,device):

    image_st = image_st.to(device)
    image_ae = image_ae.to(device)

    with torch.no_grad():

        # Predict teacher
        teacher_output_st = teacher(image_st)
        teacher_output_ae = teacher(image_ae)

        teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

    # Predict student & autoencoder
    student_output_st = student(image_st)[:, :out_channels]
    student_output_ae = student(image_ae)[:, out_channels:]
    ae_output = autoencoder(image_ae)

    # Calculate distance
    distance_st = (teacher_output_st - student_output_st) ** 2  # teacher vs student
    distance_ae = (teacher_output_ae - ae_output) ** 2          # teacher vs autoencoder
    disantce_stae = (ae_output - student_output_ae) ** 2        # autoencoder vs student

    # Calculate loss_st
    d_hard = torch.quantile(distance_st, q = 0.999)
    loss_hard = torch.mean(distance_st[distance_st >= d_hard])
    loss_st = loss_hard

    # Calculate loss_ae
    loss_ae = torch.mean(distance_ae)

    # Calculate loss_stae
    loss_stae = torch.mean(disantce_stae)

    # Total loss
    loss_total = loss_st + loss_ae + loss_stae

    return loss_total

@torch.no_grad()
def teacher_normalization(teacher, train_loader, device):
    
    # Mean
    mean_outputs = []
    for train_img, _ in tqdm(train_loader, desc = "Computing mean of features", leave = False, dynamic_ncols=True):
        train_img = train_img.to(device)
        teacher_output = teacher(train_img)
        
        mean_output = torch.mean(teacher_output, dim = [0, 2, 3]) # shape: (384, )
        mean_outputs.append(mean_output)
    
    channel_mean = torch.mean(torch.stack(mean_outputs, dim = 0), dim = 0) # shape: (N, 384) -> (384, )
    channel_mean = channel_mean[None, :, None, None] # shape: (384, ) -> (1, 384, 1, 1)

    # Std
    mean_distances = []
    for train_img, _ in tqdm(train_loader, desc = "Computing std of features", leave = False, dynamic_ncols=True):
        train_img = train_img.to(device)
        teacher_output = teacher(train_img)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim = [0, 2, 3])
        mean_distances.append(mean_distance)
    
    channel_var = torch.mean(torch.stack(mean_distances, dim = 0), dim = 0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder, out_channels, teacher_mean, teacher_std, device):
    maps_st = []
    maps_ae = []
    for image, _ in tqdm(validation_loader, desc = "Map normalization", leave = False):
        image = image.to(device)
        map_combined, map_st, map_ae = predict(teacher, student, autoencoder,
                                               image, out_channels, teacher_mean, teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)

    q_st_start = torch.quantile(maps_st, q = 0.9)
    q_st_end = torch.quantile(maps_st, q = 0.995)
    q_ae_start = torch.quantile(maps_ae, q = 0.9)
    q_ae_end = torch.quantile(maps_ae, q = 0.995)

    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def predict(teacher, student, autoencoder, image, out_channels, teacher_mean, teacher_std,
            q_st_start = None, q_st_end = None, q_ae_start = None, q_ae_end = None):
    
    teacher_output = (teacher(image) - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)

    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2, dim = 1, keepdim = True)
    map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:]) ** 2, dim = 1, keepdim = True)

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined =  0.5 * map_st + 0.5 * map_ae

    return map_combined, map_st, map_ae


def test(test_set, teacher, student, autoencoder,
         tranform, out_channels, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, device):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc = "Running inference", leave = False, dynamic_ncols=True):
        ori_width, ori_height = image.size
        image = tranform(image)
        image = image[None]
        image = image.to(device)
        map_combined, map_st, map_ae = predict(teacher, student, autoencoder,
                                               image, out_channels, teacher_mean, teacher_std,
                                               q_st_start, q_st_end, q_ae_start, q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (ori_height, ori_width), mode = "bilinear")
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        y_true_image = 0 if defect_class == "good" else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

    auc = roc_auc_score(y_true, y_score)
    return auc
