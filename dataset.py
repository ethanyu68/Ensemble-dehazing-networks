import torch.utils.data as data
import torch
import h5py, cv2
import numpy as np
import random
import torchvision
from PIL import Image

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, patch_size, aug=False):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path, 'r')
        self.input = hf.get('INPUT')
        self.target = hf.get('TARGET')
        self.path_size = patch_size
        self.toPIL = torchvision.transforms.ToPILImage()
        self.color_auger = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3)
        self.aug = aug

    def __getitem__(self, index):
        input_image = self.input[index,:,:,:]
        target_image = self.target[index, :, :, :]
        row_ind = random.randint(0, input_image.shape[0] - self.path_size)
        col_ind = random.randint(0, input_image.shape[1] - self.path_size)

        HZ = self.input[index, row_ind:row_ind+self.path_size, col_ind:col_ind+self.path_size, :]
        if self.aug:
            HZ = Image.fromarray(np.uint8(HZ*255.0))
            HZ = self.color_auger(HZ)
            HZ = np.asarray(HZ, dtype=np.float64)/255.0
        HZ = np.rollaxis(HZ, 2)
        GT = np.rollaxis(self.target[index,
                         row_ind:row_ind + self.path_size,
                         col_ind:col_ind + self.path_size, :], 2)
        if random.random() < .5:
            # print("Flip H")
            np.flip(HZ, 1)
            np.flip(GT, 1)

        if random.random() < .5:
            # print("Flip W")
            np.flip(HZ, 2)
            np.flip(GT, 2)

        # cv2.imwrite('./train_input.png', self.input[index,:,:,:]*255)
        # cv2.imwrite('./train_output.png', self.target[index, :, :, :]*255)
        return torch.from_numpy(HZ).float(), torch.from_numpy(GT).float()
        
    def __len__(self):
        return self.input.shape[0]

