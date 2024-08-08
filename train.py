import torch
import torchvision
from tqdm import tqdm

from torchvision.models import resnet50, resnet34

from settings import PATH_TO_IMAGENET_TRAIN, SEED
from core import SurrogateModelHooks, bilinear_downsample, average_pool_downsample
from pathlib import Path


def load_model(model_str, trained=True):
    if model_str == "resnet50":
        if trained:
            return resnet50(pretrained=True).to("cuda").eval()
        return resnet50(pretrained=False).to("cuda").eval()
    elif model_str == "resnet34":
        if trained:
            return resnet34(pretrained=True).to("cuda").eval()
        return resnet34(pretrained=False).to("cuda").eval()
    elif model_str == "resnet50_robust":
        if not trained:
            return resnet50(pretrained=False).to("cuda").eval()
        classifier_model = resnet50(pretrained=False)

        checkpoint = torch.load("models/ImageNet.pt")
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        new_state_dict = {}
        for key in checkpoint['model'].keys():
            if not "attacker" in key:
                if 'model' in key:
                    new_key = key[13:]
                    #print(new_key)
                    new_state_dict[new_key] = checkpoint['model'][key]

        classifier_model.load_state_dict(new_state_dict)

        return classifier_model.eval().to("cuda")


def get_conv_layers_step(module, conv_downsample_modules):
    """ collect all convolutions with stride 2 (all the downsamplings) """
    if len(list(module.children())) == 0:
        #print("no children")
        if isinstance(module, torch.nn.Conv2d):
            if module.stride[0] == 2:
                conv_downsample_modules.append(module)
    else:
        #print("has children")
        for child_module in module.children():
            get_conv_layers_step(child_module, conv_downsample_modules)


def train(model_str_from, n_epochs_to_train=10, batch_size=64, downsample_function=bilinear_downsample):
    model_from = load_model(model_str_from)

    size = 224
    transforms = [
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    transforms = torchvision.transforms.Compose(transforms)
    dataset = torchvision.datasets.ImageFolder(PATH_TO_IMAGENET_TRAIN, transform=transforms)




    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)

    conv_downsample_modules = []
    get_conv_layers_step(model_from, conv_downsample_modules)

    network = SurrogateModelHooks(model_from, hook_points=conv_downsample_modules, surrogate_downsampling=downsample_function,
                                  sample_input=dataset[0][0].unsqueeze(dim=0).to("cuda"))
    network = network.to("cuda")
    network = network.eval()


    optimizer = network.configure_optimizers()[0]

    ckpt_path = "ckpts/" + model_str_from + "/"
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)

    network.save_checkpoint(ckpt_path, 0)

    i = 0
    for epoch in range(n_epochs_to_train):
        #for data in tqdm(train_loader, ascii=True):
        for data in tqdm(train_loader, ascii=True):
            img = data[0].to("cuda")

            optimizer.zero_grad()
            loss = network.training_step(img, i)
            loss.backward()
            optimizer.step()

            i += 1

        network.save_checkpoint(ckpt_path, epoch+1)


if __name__ == '__main__':
    train("resnet34", batch_size=64, downsample_function=bilinear_downsample)
    train("resnet50", batch_size=64, downsample_function=bilinear_downsample)
    train("resnet50_robust", batch_size=64, downsample_function=bilinear_downsample)









