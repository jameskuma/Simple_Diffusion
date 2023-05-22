import os
import tqdm
import torch
import imageio
import numpy as np
import torchvision.transforms.functional as torchvision_F
from torch.utils.data import Dataset, DataLoader

class POKEMON_DATASET(Dataset):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.img_size = hparams.img_size
        self.images = self.preload_images(hparams)

    def preload_images(self, hparams):
        root_path = f"{hparams.basedir}/{hparams.datatype}"
        images = []
        for fni in tqdm.tqdm(sorted(os.listdir(root_path)), desc="loading dataset", leave=False):
            if fni.endswith((".JPG", ".png", ".jpg")):
                pi = f"{root_path}/{fni}"
                image = np.array(imageio.imread(pi)) / 255.
                image = image[..., :3]*image[..., 3:] + (1.-image[..., 3:])
                images.append(image)
        images = np.stack(images, axis=0)
        return images
    
    def op_images_augment(self, image:np.ndarray):
        image = torch.from_numpy(image).float().permute([2,0,1])
        image = torchvision_F.center_crop(torchvision_F.resize(image[None,...], size=[self.img_size], antialias=True), [self.img_size])[0]
        image = torchvision_F.normalize(image, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        return image
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, ind):
        return self.op_images_augment(self.images[ind])
    
    def set_loader(self, bs=None, is_shuffle=True):
        return DataLoader(self, 
                          batch_size=bs,
                          shuffle=is_shuffle)