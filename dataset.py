from torch.utils.data import Dataset
import os
import csv
import PIL
import torch
import torchvision
import copy

from torchvision.datasets.folder import default_loader

from settings import PATH_TO_IMAGENET_TRAIN, PATH_VAL_SOLUTIONS

torchvision.datasets.ImageNet

class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_val, transform=None, transform_original=None, return_original_image=False):
        super().__init__()
        self.dataset_path = dataset_path_val
        self.transform = transform
        self.transform_original = transform_original
        self.return_original_image = return_original_image

        dataset_path_train = PATH_TO_IMAGENET_TRAIN#"/localdata/xai_derma/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"

        wnid_to_class = {}
        i = 0
        for folder_name in sorted(os.listdir(dataset_path_train)):
            wnid_to_class[folder_name] = i
            i += 1
        val_solutions_path = PATH_VAL_SOLUTIONS#"/localdata/xai_derma/imagenet-object-localization-challenge/LOC_val_solution.csv"

        self.image_id_to_class = {}
        with open(val_solutions_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                image_id, wnid = line[0].split(",")
                self.image_id_to_class[image_id] = wnid_to_class[wnid]

        self.files = os.listdir(self.dataset_path)


    def __len__(self):
        #print("len dataset: {}".format(len(os.listdir(self.dataset_path))))
        return len(self.files)


    def __getitem__(self, index):
        filepath = self.files[index]
        class_idx = self.image_id_to_class[filepath.split(".")[0]]

        sample = default_loader(os.path.join(self.dataset_path, filepath))
        original_sample = copy.deepcopy(sample)
        if self.transform is not None:
            sample = self.transform(sample)
            #return self.transform(sample), class_idx

        if self.return_original_image:
            #print("return original sample")
            original_sample = self.transform_original(original_sample)
            return sample, class_idx, original_sample

        #print("do not return original sample")
        return sample, class_idx



class ImageNetValDataset_500(Dataset):
    def __init__(self, transform, transform_crop_resize, image_folder_path=""):
        self.transform = transform
        self.transform_crop_resize = transform_crop_resize
        self.ToTensor = torchvision.transforms.ToTensor()
        self.image_folder_path = image_folder_path
        self.files = os.listdir(image_folder_path)


    def __len__(self):
        return int(len(self.files)/100)


    def __getitem__(self, index):
        original_img = PIL.Image.open(self.image_folder_path+self.files[index*100])

        if self.transform != None:
            img = self.ToTensor(original_img)
            if img.shape[0] == 1:
                img = torch.stack([img[0], img[0], img[0]])
            img = self.transform(img)

        original_img = self.ToTensor(original_img)
        if original_img.shape[0] == 1:
            original_img = torch.stack([original_img[0], original_img[0], original_img[0]])
        original_img = self.transform_crop_resize(original_img)

        return img, original_img



class AFHQ_Dataset(Dataset):
    def __init__(self, transform=None, image_folder_path=""):
        self.transform = transform
        self.ToTensor = torchvision.transforms.ToTensor()

        self.image_folder_path = image_folder_path
        self.files = os.listdir(self.image_folder_path)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        img = PIL.Image.open(self.image_folder_path + self.files[index])
        normal_img = self.ToTensor(img)

        if self.transform != None:
            img = self.transform(img)

        return img, normal_img












