import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self,in_channels,out_channels):
        super(DoubleConv, self).__init__()  # Initialize the parent class nn.Module
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,features=[ 64 , 128 , 256 , 512 ]): # in_channels = 3 for RGB images, out_channels = 1 for binary segmentation , features = number of filters in each layer
        super(UNET,self).__init__() # Initialize the parent class nn.Module

        self.ups=nn.ModuleList() # Upsampling layers  of Decoder

        self.downs=nn.ModuleList() # Downsampling layers of Encoder

        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

        # Down part of UNET (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature)) # DoubleConv = block (Conv → BN → ReLU) × 2
            # DoubleConv(in_channels=3,feature=64)->(in_channels=64,feature=128)->(in_channels=128,feature=256)->(in_channels=256,feature=512)
            in_channels = feature

        # Up part of UNET (Decoder)
        for feature in reversed(features): # reversed(features) = [512,256,128,64]
            self.ups.append( # feature*2 because the bottleneck has double the number of features and skip connections concatenate features  from encoder (e.g., 512 from encoder + 512 from bottleneck = 1024, then upsample to 512)
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
            )
            self.ups.append(DoubleConv(feature*2,feature)) #(1024,512)->(512,256)->(256,128)->(128,64)

        self.bottleneck=DoubleConv(features[-1],features[-1]*2) # features[-1]=512 -> (512,1024)
        self.final_conv=nn.Conv2d(features[0],out_channels,kernel_size=1) # features[0]=64 -> (64,1)




    def forward(self,x):

        skip_connections=[]

        for down in self.downs:

            x=down(x) # x is the input image (channels, height, width), down is the DoubleConv block
            skip_connections.append(x)
            x=self.pool(x)

        x=self.bottleneck(x)

        skip_connections=skip_connections[::-1] # reverse the skip connections to match the order of upsampling

        for idx in range(0,len(self.ups),2):
        # step by 2 because we have two layers for each upsampling (ConvTranspose2d and DoubleConv) , [ConvTranspose2d, DoubleConv, ConvTranspose2d, DoubleConv, ConvTranspose2d, DoubleConv, ...]
            x=self.ups[idx](x) # x is the output of bottleneck, self.ups[idx] is ConvTranspose2d
            skip_connection=skip_connections[idx//2] # get the corresponding skip connection

            if x.shape != skip_connection.shape: # if the shapes are not the same, we need to resize
                x=TF.resize(x,size=skip_connection.shape[2:]) # resize to the size of the skip connection

            concat=torch.cat((skip_connection,x),dim=1) # dim=1 because we want to concatenate along the feature map dimension ,(batch_size, channels, height, width),dim=0=batch_size,dim=1=channels,dim=2=height,dim=3=width
            x=self.ups[idx+1](concat)

        return self.final_conv(x) # if i use sigmoid here the output will be between 0 and 1 but it's better to use BCEWithLogitsLoss which combines a Sigmoid layer and the BCELoss in one single class
        # return torch.sigmoid(self.final_conv(x)) # to get the output between 0 and 1





def test():
    x=torch.randn((3,1,160,160)) # batch size = 3, channels = 1, height = 160, width = 160 ,batch size is the number of images in a batch
    model=UNET(in_channels=1,out_channels=1)
    preds=model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape # check if the output shape is the same as the input shape


if __name__=="__main__":
    test()






