import torch
import albumentations as A # for data augmentation
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm # for 100%|█████████████████████| 100/100 [00:00<00:00, 2000it/s]
import  torch.nn as nn
import torch.optim as optim
from Model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,)



# Hyperparameters etc.
LEARINING_RATE=1e-4
# use mps for macOS with M1 chip or cuda for NVIDIA GPU
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
BATCH_SIZE=16
NUM_EPOCHS=3
NUM_WORKERS=2
IMAGE_HEIGHT=160 # 1280 originally, make smaller if you want to train faster
IMAGE_WIDTH=240 # 1918 originally
PIN_MEMORY=True
LOAD_MODEL=True

TRAIN_IMG_DIR="/Users/mhmdgamal/Downloads/Segmentation_from_scratch_usingU-Net/data/train/"
TRAIN_MASK_DIR="/Users/mhmdgamal/Downloads/Segmentation_from_scratch_usingU-Net/data/train_masks/"
VAL_IMG_DIR="/Users/mhmdgamal/Downloads/Segmentation_from_scratch_usingU-Net/data/data_validation/"
VAL_MASK_DIR="/Users/mhmdgamal/Downloads/Segmentation_from_scratch_usingU-Net/data/data_validation_mask/"



def train_fn(loader,model,optimizer,loss_fn,scaler):

    loop=tqdm(loader)

    for batch_idx,(data,target) in enumerate(loop):
        data=data.to(device=DEVICE) #
        target=target.float().unsqueeze(1).to(device=DEVICE) # unsqueeze to add channel dimension (BATCH_SIZE,1,HEIGHT,WIDTH)

        # forward
        with torch.cuda.amp.autocast(): # automatic mixed precision to save memory and speed up training
            predictions=model(data)
            loss=loss_fn(predictions,target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward() # scale the loss to prevent underflow when using mixed precision , Automatic Mixed Precision (AMP)
        scaler.step(optimizer) # optimizer.step() but with scaling
        scaler.update() # update the scaler for next iteration and check for overflow or underflow

        # update tqdm loop
        loop.set_postfix(loss=loss.item()) # display loss in tqdm loop and itmem() to get the value of the tensor as a standard Python number



def main():
    train_transform=A.Compose( # Compose is used to combine multiple augmentations as a p
        [
            A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH ),
            A.Rotate(limit=35,p=1.0), # rotate the image by a random angle between -35 and 35 degrees with probability of 1.0 call  Data Augmentation and p is the probability of applying the transformation(100% of the time)
            A.HorizontalFlip(p=0.5), # flip the image horizontally with probability of
            # Tensor doesn't divide by 255 like pytorch's
            # it's done inside Normalize function
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ), # normalize the image to [0,1] ,new pixel value = (pixel - mean) / std , here we don't change the mean and std because we want the values to be between 0 and 1

            ToTensorV2(), # convert the image to a PyTorch tensor to numPy array
        ]
    )


    Val_transform=A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH ),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(), # convert the image to a PyTorch tensor to numPy array


        ]
    )

    model=UNET(in_channels=3,out_channels=1).to(DEVICE) # in_channels=3 for RGB images and out_channels=1 for binary segmentation if multi-class segmentation change "chanel" it to the number of classes
    loss_fn=nn.BCEWithLogitsLoss() # binary cross entropy loss with logits because the model's output is not between 0 and 1
    optimizer=optim.Adam(model.parameters(),lr=LEARINING_RATE)

    train_loader,val_loader=get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        Val_transform,
        NUM_WORKERS,
        PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"),model)
    check_accuracy(val_loader, model, device=DEVICE)

    scaler=torch.cuda.amp.GradScaler() # for automatic mixed precision

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader,model,optimizer,loss_fn,scaler)
        # save model
        checkpoint={
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader,model,device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader,model,folder="saved_images/",device=DEVICE
        )




if __name__ == "__main__":
    main()