
import torch
from monai.networks.nets import TorchVisionFCModel
import monai
from core import SurrogateModelHooks
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random


def load_model():
    model = TorchVisionFCModel(pretrained=False, use_conv=True)

    state_dict = torch.load("pathology_tumor_detection/models/model.pt")#.to("cuda").eval()

    model.load_state_dict(state_dict)
    model = model.to("cuda").eval()
    #print(model)
    #print(model_load)

    return model


def load_dataset(transform=None):
    #val_indice = [1]
    #data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/testing/images/test_" + str(i).zfill(3) + ".tif"} for i in val_indice]
    #dataset = monai.data.MaskedPatchWSIDataset(data, patch_size=[224, 224], mask_level=6, transform=transform)
    indize_normal = [i for i in range(160)]
    indize_normal.remove(85)

    data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/training/tumor/tumor_" + str(i+1).zfill(3) + ".tif"} for i in range(111)] + [
        {"image": "/home/digipath2/data/Camelyon/CAMELYON16/training/normal/normal_" + str(i+1).zfill(3) + ".tif"} for i in indize_normal
    ]
    dataset = monai.data.MaskedPatchWSIDataset(data, patch_size=[224, 224], mask_level=6, transform=transform)
    return dataset



def save_tumor_pred_indices(dataset, model):
    indices = []
    index = 0
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=False, num_workers=32, pin_memory=False)

    with torch.no_grad():
        for data in tqdm(dataloader, ascii=True):
            img = data["image"].to("cuda")
            pred = model(img)
            for i, p in enumerate(pred):
                if p > 0:
                    indices.append(index)
                index += 1

    print("number of tumor samples: {}".format(len(indices)))
    print("total samples: {}".format(index))
    np.save("camelyon16_tumor_indice.npy", indices)


def save_train_indice():
    num_samples = 99410615
    tumor_indice = set(np.load("camelyon16_data_indice/camelyon16_tumor_indice.npy"))

    non_tumor_pred_indice = set()
    while len(non_tumor_pred_indice) < len(tumor_indice):
        idx = random.randint(0, num_samples-1)
        if not idx in tumor_indice:
            if not idx in non_tumor_pred_indice:
                non_tumor_pred_indice.add(idx)
                if len(non_tumor_pred_indice) % 10000 == 0:
                    print(len(non_tumor_pred_indice))

    np.save("camelyon16_data_indice/non_tumor_pred_indice.npy", list(non_tumor_pred_indice))






def get_tumor_pred_imgs(batch_data, n_samples_wanted=8):
    tumor_samples = []
    preds = model(batch_data)

    for i, pred in enumerate(preds):
        if pred > 0.0:
            tumor_samples.append(batch_data[i])
        if len(tumor_samples) >= n_samples_wanted:
            break

    tumor_samples = torch.stack(tumor_samples, dim=0)
    return tumor_samples


def get_conv_layers_step(module, conv_downsample_modules):
    if len(list(module.children())) == 0:
        #print("no children")
        if isinstance(module, torch.nn.Conv2d):
            if module.stride[0] == 2:
                conv_downsample_modules.append(module)
    else:
        #print("has children")
        for child_module in module.children():
            get_conv_layers_step(child_module, conv_downsample_modules)




transform = monai.transforms.Compose([
    monai.transforms.CastToTyped(keys="image", dtype="float32"),
    monai.transforms.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=-1.0, b_max=1.0),
    monai.transforms.ToTensord(keys="image"),
])

tumor_indices_train = np.load("camelyon16_data_indice/camelyon16_tumor_indice.npy")
non_tumor_indices_train = np.load("camelyon16_data_indice/non_tumor_pred_indice.npy")

dataset = load_dataset(transform=transform)
tumor_ds = torch.utils.data.Subset(dataset, tumor_indices_train)
non_tumor_ds = torch.utils.data.Subset(dataset, non_tumor_indices_train)


train_ds = torch.utils.data.ConcatDataset([tumor_ds, non_tumor_ds])

#print("len dataset: {}".format(len(dataset)))

#raise RuntimeError

model = load_model()


#save_train_indice()
#save_tumor_pred_indices(dataset, model)
#raise RuntimeError

train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=512, shuffle=True, num_workers=16, pin_memory=False)


conv_downsample_modules = []
get_conv_layers_step(model, conv_downsample_modules)

network = SurrogateModelHooks(model, hook_points=conv_downsample_modules, sample_input=dataset[0]['image'].unsqueeze(dim=0).to("cuda"))
network = network.to("cuda")
network = network.eval()

optimizer = network.configure_optimizers()[0]

ckpt_path = "ckpts/" + "chamelyon_resnet18" + "/"
Path(ckpt_path).mkdir(parents=True, exist_ok=True)

network.save_checkpoint(ckpt_path, 0)

n_epochs_to_train = 10

i = 0
for epoch in range(n_epochs_to_train):
    for data in tqdm(train_loader, ascii=True):
        img = data["image"].to("cuda")

        optimizer.zero_grad()
        loss = network.training_step(img, i)
        loss.backward()
        optimizer.step()

        i += 1

    network.save_checkpoint(ckpt_path, epoch+1)
