import argparse
import os
import numpy as np
import torch, cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms 

from utils.util import make_output_folder
from utils.dataset import ImageFolderWithPath
from utils.train_utils import predict

device = "cuda" if torch.cuda.is_available() else "cpu"

RESIZE_SIZE = 256 # fix
OUT_CHANNELS = 384

def get_argparse():
    parser = argparse.ArgumentParser(description="Inference EfficientAD")
    parser.add_argument("-t", "--test_dataset")
    parser.add_argument("-m", "--model_dir")
    return parser.parse_args()

def load_model(model_dir):

    assert os.path.exists(model_dir)

    teacher = torch.load(os.path.join(model_dir, "teacher_best.pt"))["model"]
    student = torch.load(os.path.join(model_dir, "student_best.pt"))["model"]
    autoencoder = torch.load(os.path.join(model_dir, "autoencoder_best.pt"))["model"]

    teacher_mean = torch.load(os.path.join(model_dir, "teacher_mean.pt"))
    teacher_std = torch.load(os.path.join(model_dir, "teacher_std.pt"))

    q_st_start = torch.load(os.path.join(model_dir, "q_st_start_best.pt"))
    q_st_end = torch.load(os.path.join(model_dir, "q_st_end_best.pt"))

    q_ae_start = torch.load(os.path.join(model_dir, "q_ae_start_best.pt"))
    q_ae_end = torch.load(os.path.join(model_dir, "q_ae_end_best.pt"))

    return teacher, student, autoencoder, teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def inference(img:Image.Image,
              teacher, student, autoencoder,
              transform, out_channels,
              teacher_mean, teacher_std,
              q_st_start = None, q_st_end = None,
              q_ae_start = None, q_ae_end = None):
    
    ori_width, ori_height = img.size
    img_tensor = transform(img)
    img_tensor = img_tensor[None]
    img_tensor = img_tensor.to(device)

    map_combined, map_st, map_ae = predict(teacher, student, autoencoder,
                                           img_tensor, out_channels,
                                           teacher_mean, teacher_std,
                                           q_st_start, q_st_end,
                                           q_ae_start, q_ae_end)
    
    map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
    map_combined = torch.nn.functional.interpolate(
        map_combined, (ori_height, ori_width), mode='bilinear')
    map_combined = map_combined[0, 0].cpu().numpy()

    return map_combined

@torch.no_grad()
def main():

    # Config
    config = get_argparse()

    test_dataset_dir = config.test_dataset.replace("\\", "/")
    test_dataset_dir = test_dataset_dir[:-1] if test_dataset_dir[-1] == "/" else test_dataset_dir

    model_dir = config.model_dir.replace("\\", "/")
    model_dir = model_dir[:-1] if model_dir[-1] == "/" else model_dir

    # Make result dir
    result_dir = make_output_folder(test_dataset_dir.replace("./dataset", "./result"))

    # Transform
    default_transform = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # Load datset, model
    test_ds = ImageFolderWithPath(test_dataset_dir)
    teacher, student, autoencoder, teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end = load_model(config.model_dir)

    teacher.eval()
    student.eval()
    autoencoder.eval()

    # Inference
    with torch.no_grad():
        for img, target, path in tqdm(test_ds, desc = "Running Inference", leave = False, dynamic_ncols = True):

            defect_class = os.path.basename(os.path.dirname(path))
            result_subdir = os.path.join(result_dir, defect_class)
            os.makedirs(result_subdir, exist_ok = True)

            # Get anomaly map
            map_combined = inference(img, teacher, student, autoencoder,
                                     default_transform, OUT_CHANNELS,
                                     teacher_mean, teacher_std,
                                     q_st_start, q_st_end, q_ae_start, q_ae_end)
            
            clip_thresh_min = np.min(map_combined)
            clip_thresh_max = clip_thresh_min + 0.5
            clipped_map = np.clip(map_combined, clip_thresh_min, clip_thresh_max)

            map_combined = ((clipped_map - clip_thresh_min) / (clip_thresh_max) * 255).astype(np.uint8)
            
            heatmap_combined = cv2.applyColorMap(map_combined, cv2.COLORMAP_JET)
            
            out_img_array = np.float32(np.asarray(img)) / 255.
            out = np.float32(heatmap_combined) / 255. + cv2.cvtColor(out_img_array, cv2.COLOR_RGB2BGR)
            out = out / np.max(out)
            out = np.uint8(out * 255.0)

            heatmap_combined = np.float32(heatmap_combined) / 255.
            heatmap_combined /= np.max(heatmap_combined)
            heatmap_combined = np.uint8(heatmap_combined * 255.)
    
            cv2.imwrite(os.path.join(result_subdir, os.path.basename(path)), out)
            cv2.imwrite(os.path.join(result_subdir, os.path.splitext(os.path.basename(path))[0] + "_anomaly_map.png"), heatmap_combined)
    
    print("End Inference!")

if __name__ == "__main__":
    main()