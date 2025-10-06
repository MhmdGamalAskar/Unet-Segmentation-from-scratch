import os, glob
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from tqdm import tqdm

from Model import UNET


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

IMAGE_HEIGHT = 160  # 1280 originally, make smaller if you want to train faster
IMAGE_WIDTH = 240  # 1918 originally

INPUT_DIR  = "/Users/mhmdgamal/Downloads/Segmentation_from_scratch_usingU-Net/data/test_img"
OUTPUT_DIR = "/Users/mhmdgamal/Downloads/Segmentation_from_scratch_usingU-Net/saved_images/"
MODEL_PATH = "/Users/mhmdgamal/Downloads/Segmentation_from_scratch_usingU-Net/my_checkpoint.pth.tar"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# validation transforms
val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

def load_model(model_path, device=DEVICE):
    model=UNET(in_channels=3, out_channels=1).to(device)
    ckpt=torch.load(model_path, map_location=device) # load all tensors onto the device (cpu, cuda, mps)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def segment_images( model,img_path):
    img_pil=Image.open(img_path).convert("RGB")
    #orig_w, orig_h = img_pil.size
    img_np = np.array(img_pil)

    aug=val_transform(image=img_np)
    x=aug["image"].unsqueeze(0).to(DEVICE)  # add batch dimension (1, C, H, W)

    logits=model(x)
    #logits_resized=F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False) # resize the output to the original image size
    prob=torch.sigmoid(logits_resized)  # apply sigmoid to get probabilities between 0 and 1
    pred=(prob>0.5).float()  # threshold the probabilities to get binary mask

    mask = (pred.squeeze().cpu().numpy() * 255).astype("uint8")  # convert to numpy array and scale to [0, 255]
    return Image.fromarray(mask) # convert to PIL image




def main():
    model = load_model(MODEL_PATH, device=DEVICE)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    img_paths = []
    for e in exts:
        img_paths.extend(glob.glob(os.path.join(INPUT_DIR, e)))
    img_paths.sort()

    print(f"Found {len(img_paths)} images in {INPUT_DIR}")
    if not img_paths:
        return

    for img_path in tqdm(img_paths):
        mask = segment_images(model, img_path)
        stem, _ = os.path.splitext(os.path.basename(img_path))
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_mask.png")
        mask.save(out_path)

    print(f"Saved masks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


