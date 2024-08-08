import torch
import torchvision
from tqdm import tqdm

from settings import PATH_TO_IMAGENET_TRAIN, SEED, PATH_TO_IMAGENET_VAL
from core import EvalSurrogateModel, bilinear_downsample, average_pool_downsample

from dataset import ImageNetValDataset
import captum

from torchvision.utils import save_image

import matplotlib.pyplot as plt
from train import get_conv_layers_step

from eval_tv import eval_pred_difference_imagenet
import copy
from torchmetrics.functional.image import total_variation
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# necessary for insertion and deletion
try:
    from evaluation import CausalMetric, auc, gkern
    from explanations import RISE
except:
    pass

from pathlib import Path


from train import load_model
import time


DATA_RESIZE_SIZE = 256 #256 #480
DATA_CROP_SIZE = 224 #224 #480



class ImageNetDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, target = self.dataset.__getitem__(index)
        return x


def get_saliency_method(model, method, layer, input_layer=False, use_smooth_grad=False, grad_cam_layer_override=None):
    if method == "GradCAM":
        if grad_cam_layer_override is not None:
            return captum.attr.LayerGradCam(model, layer=grad_cam_layer_override)
        return captum.attr.LayerGradCam(model, layer=model.layer4)
    elif method == "DeepLift_Input":
        return captum.attr.DeepLift(model, multiply_by_inputs=True)
    elif method == "IG_Input":
        return captum.attr.IntegratedGradients(model, multiply_by_inputs=True)
    elif method == "Grad_Input":
        return captum.attr.Saliency(model)


    if input_layer:
        if method == "DeepLift":
            ig = captum.attr.DeepLift(model, multiply_by_inputs=True)
    else:
        if method == "DeepLift":
            ig = captum.attr.LayerDeepLift(model, layer, multiply_by_inputs=True)
        elif method == "IG":
            ig = captum.attr.LayerIntegratedGradients(model, layer, multiply_by_inputs=True)
        elif method == "Grad":
            ig = captum.attr.LayerGradientXActivation(model, layer, multiply_by_inputs=False)
    if use_smooth_grad:
        return captum.attr.NoiseTunnel(ig)
    return ig



def get_saliency_result(model, saliency_module, dataloader, baselines=0.0):
    saliency_images = []

    for x, _ in tqdm(dataloader, ascii=True):
        x = torch.tensor(x).to("cuda")
        with torch.no_grad():
            pred = model(x)
            #print("min pred: {}, max pred: {}".format(torch.min(pred), torch.max(pred)))
            top = np.argmax(pred.cpu().numpy(), -1)
            top = torch.from_numpy(top).to("cuda")

        #target = target.to("cuda")
        #grad = saliency_module.attribute(x, target=top, stdevs=0.2, nt_samples=20, nt_samples_batch_size=1).detach()
        try:
            grad = saliency_module.attribute(x, target=top, baselines=baselines).detach()
        except:
            grad = saliency_module.attribute(x, target=top).detach()
        saliency_images.append(torch.mean(grad, dim=[1]).cpu())

    return torch.cat(saliency_images, dim=0)



#def evaluate_tv(model, dataloader, layer, input_layer=False, method="DeepLift"):
def evaluate_tv(saliency_module, dataloader):
    total_tv = []
    #model = model.to("cuda")

    #GradientAccessor = get_saliency_method(model, method, layer, input_layer=input_layer)

    i = 0
    for x, target in tqdm(dataloader, ascii=True):
        x = x.to("cuda")
        target = target.to("cuda")
        #grad = GradientAccessor.attribute(x, target=target, stdevs=0.2, nt_samples=20, nt_samples_batch_size=1).detach()

        #grad = saliency_module.attribute(x, target=target, stdevs=0.2, nt_samples=20, nt_samples_batch_size=1).detach()
        grad = saliency_module.attribute(x, target=target).detach()

        with torch.no_grad():
            grad_scaled = grad - torch.mean(grad, dim=[2, 3]).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grad_scaled = grad_scaled / torch.max(torch.mean(torch.abs(grad_scaled), dim=[2, 3]).unsqueeze(dim=-1).unsqueeze(dim=-1), torch.tensor(0.000001).to("cuda"))
            tv = total_variation(grad_scaled, reduction='sum')
            tv_pixelwise_mean = tv / (grad.shape[1]*grad.shape[2] * grad.shape[3])
            #print("tv shape: {}".format(tv.shape))
            #raise RuntimeError
            total_tv.append(tv_pixelwise_mean)

        i += 1
        #break

        #if i > 5:
        #    break

    total_tv = torch.stack(total_tv, dim=0)
    mean_tv = torch.mean(total_tv)
    std_tv = torch.std(total_tv)

    print("mean_tv: {}".format(mean_tv))
    print("std_tv: {}".format(std_tv))

    return mean_tv


def get_mask_n_highest_pixels(data_tensor, num_to_get):
    highest_indize = []
    mask = np.ones_like(data_tensor)

    copy_data_tensor = np.copy(data_tensor)
    for i in range(num_to_get):
        index = np.unravel_index(copy_data_tensor.argmax(), copy_data_tensor.shape)
        highest_indize.append(index)
        copy_data_tensor[index] = -np.inf
        mask[index] = 0.0

    return torch.from_numpy(mask).to("cuda")


class MaskForwardHook:
    def __init__(self, hook_point, mask):
        hook_point.register_forward_hook()
        self.mask = mask

    def update_mask(self, new_mask):
        self.mask = new_mask

    def hook_forward(self, module, input_, output):
        return output * self.mask




def hook_model(model, convolutions, surrogate_downsampling=bilinear_downsample):
    hook_points = []
    get_conv_layers_step(model, hook_points)
    #print("hook points: {}".format(hook_points))
    #raise RuntimeError
    network = EvalSurrogateModel(model, hook_points=hook_points, surrogate_convolutions=convolutions, is_stitched_classifier=False,
                                 surrogate_downsampling=surrogate_downsampling)

    return network


def get_imagenet_val_ds_bs_one():
    dataset = get_dataset(return_original_sample=False)
    dataset = torch.utils.data.Subset(dataset, [i*100 for i in range(500)])
    val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    return val_loader



def get_imagenet_val_ds(batch_size=10):
    dataset = get_dataset(return_original_sample=False)
    dataset = torch.utils.data.Subset(dataset, [i*100 for i in range(500)])

    #dataset = torch.utils.data.Subset(dataset, [i*100 for i in range(500)])
    val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    return val_loader


def test_tv(model_from, tv_hook_layer, convolutions=None, input_layer=False, method="DeepLift",
            use_grad_fix=False, use_forward_fix=False, surrogate_downsampling=bilinear_downsample):
    val_loader = get_imagenet_val_ds_bs_one()

    hook_points = []
    get_conv_layers_step(model_from, hook_points)

    saliency_module = get_saliency_method(model_from, method, tv_hook_layer, input_layer=input_layer)

    if use_grad_fix:
        for hook_point in hook_points:
            hook_point.register_backward_hook(conv_backward_hook_grad_fix)
        return evaluate_tv(saliency_module, val_loader)
    elif use_forward_fix:
        for conv_layer in hook_points:
            ForwardGradFixHook(conv_layer, conv_layer.weight, conv_layer.bias, conv_layer.padding)
        return evaluate_tv(saliency_module, val_loader)

    else:
        EvalSurrogateModel(model_from, hook_points=hook_points, surrogate_convolutions=convolutions, is_stitched_classifier=False,
                           surrogate_downsampling=surrogate_downsampling)
        return evaluate_tv(saliency_module, val_loader)


class SoftmaxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)



def test_saliency_difference(model_one, model_two, layer_one, layer_two):
    ig_one = captum.attr.LayerIntegratedGradients(model_one, layer_one, multiply_by_inputs=True)
    ig_two = captum.attr.LayerIntegratedGradients(model_two, layer_two, multiply_by_inputs=True)

    saliency_module_one = captum.attr.NoiseTunnel(ig_one)
    saliency_module_two = captum.attr.NoiseTunnel(ig_two)

    val_loader = get_imagenet_val_ds_bs_one()

    total = []

    for x, target in tqdm(val_loader, ascii=True):
        x = x.to("cuda")
        target = target.to("cuda")

        saliency_map_one = saliency_module_one.attribute(x, target=target, stdevs=0.2, nt_samples=5, nt_samples_batch_size=1).detach()
        saliency_map_two = saliency_module_two.attribute(x, target=target, stdevs=0.2, nt_samples=5, nt_samples_batch_size=1).detach()

        l1_difference = F.l1_loss(saliency_map_one, saliency_map_two, reduce='mean')
        total.append(l1_difference.detach().cpu())


    total = torch.stack(total, dim=0)
    mean_tv = torch.mean(total)
    std_tv = torch.std(total)

    print("mean difference: {}".format(mean_tv))
    print("std difference: {}".format(std_tv))




def test(model_from, convolutions=None):
    hook_points = []
    model_original = copy.deepcopy(model_from)
    get_conv_layers_step(model_from, hook_points)
    network = EvalSurrogateModel(model_from, hook_points=hook_points, surrogate_convolutions=convolutions, is_stitched_classifier=False)
    network = network.to("cuda")
    network = network.eval()

    val_loader, val_loader_x_only = get_dataloader(64)

    #plot_gradients(network, val_loader)
    #get_test_accuracy(network, val_loader)
    #eval_pred_difference_imagenet(model_original, network, val_loader_x_only)



def get_dataset(return_original_sample=False):
    transform = [
        torchvision.transforms.Resize(DATA_RESIZE_SIZE),
        torchvision.transforms.CenterCrop(DATA_CROP_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    transform = torchvision.transforms.Compose(transform)


    transform_original_sample = [
        torchvision.transforms.Resize(DATA_RESIZE_SIZE),
        torchvision.transforms.CenterCrop(DATA_CROP_SIZE),
        torchvision.transforms.ToTensor(),
    ]
    transform_original_sample = torchvision.transforms.Compose(transform_original_sample)


    if return_original_sample:
        dataset = ImageNetValDataset(transform=transform,
                                    dataset_path_val=PATH_TO_IMAGENET_VAL,
                                    return_original_image=True,
                                    transform_original=transform_original_sample)
    else:
        dataset = ImageNetValDataset(transform=transform,
                                    dataset_path_val=PATH_TO_IMAGENET_VAL)

    return dataset


def get_dataloader(batch_size, return_original_sample=False):
    dataset = get_dataset(return_original_sample)
    dataset_x_only = ImageNetDatasetWrapper(dataset)
    val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)
    val_loader_x_only = torch.utils.data.DataLoader(
            dataset_x_only, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)


    return val_loader, val_loader_x_only






class SurrogateModelInputHook(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.identity_layer_for_input_hook = nn.Identity()

    def forward(self, x):
        x = self.identity_layer_for_input_hook(x)
        return self.model(x)


def get_indices_n_highest_pixels(data_tensor, num_to_get=4):
    highest_indices = []

    copy_data_tensor = np.copy(data_tensor)
    for i in range(num_to_get):
        index = np.unravel_index(copy_data_tensor.argmax(), copy_data_tensor.shape)
        highest_indices.append(index)
        copy_data_tensor[index] = -np.inf

    return highest_indices


#torchvision.transforms.GaussianBlur
#F.gaussian_blur

def get_layer_hook_point(model, layer_str):
    layer_hook_point = None
    for module in layer_str.split("."):
        if layer_hook_point == None:
            layer_hook_point = getattr(model, module)
        else:
            layer_hook_point = getattr(layer_hook_point, module)

    return layer_hook_point





def plot_img():
    val_loader = get_dataloader(64)
    img = find_img_of_class(val_loader, 207)
    save_image(img, "imgs/labrador.jpg")



def find_img_of_class(val_loader, target_class=207, return_original_sample=False):
    #print("target class: {}".format(target_class))
    target_class = int(target_class)
    for x, targets, original_sample in val_loader:
        #print("in loop")
        for i, target in enumerate(targets):
            #print(target)
            if int(target) == target_class:
                if return_original_sample:
                    #print("return original sample")
                    #print(x[i].unsqueeze(dim=0).shape)
                    #print(original_sample[i].unsqueeze(dim=0).shape)

                    return x[i].unsqueeze(dim=0), original_sample[i].unsqueeze(dim=0)
                #print("return single sample")
                return x[i].unsqueeze(dim=0)


def get_sample_of_class(target_class, return_original_sample=False):
    val_loader, val_loader_x_only = get_dataloader(batch_size=64, return_original_sample=True)
    return find_img_of_class(val_loader, target_class, return_original_sample)


def conv_backward_hook_grad_fix(module, grad_input, grad_output):
    accumulated_grad = None

    for h in range(2):
        for w in range(2):
            rolled_grad = torch.roll(grad_input[0], shifts=[h, w], dims=[2, 3])
            if accumulated_grad is None:
                accumulated_grad = rolled_grad
            else:
                accumulated_grad += rolled_grad

    return (accumulated_grad/4.0, grad_input[1], grad_input[2])




class ForwardGradFixHook:
    def __init__(self, hook_point, conv_weight, conv_bias, padding):
        hook_point.register_forward_hook(self.forward_hook)
        self.conv_weight = conv_weight
        self.conv_bias = conv_bias
        self.padding = padding

    def forward_hook(self, module, original_input, original_output):
        accumulated_activation = None

        #print("len original input: {}".format(len(original_input)))

        #print("original input 0 shape: {}".format(len(original_input[0].shape)))

        for h in range(2):
            for w in range(2):
                rolled_input = torch.roll(original_input[0], shifts=[h, w], dims=[2, 3])
                output = F.conv2d(rolled_input, self.conv_weight, self.conv_bias, stride=2, padding=self.padding)
                if accumulated_activation is None:
                    accumulated_activation = output
                else:
                    accumulated_activation += output


        #print("len original output: {}".format(len(original_output)))

        return accumulated_activation/4.0


def forward_fix_hook_model(model):
    hook_points = []
    get_conv_layers_step(model, hook_points)
    #for hook_point in hook_points:
    #    hook_point.register_backward_hook(conv_backward_hook_grad_fix)
    j = 0
    for conv_layer in hook_points:
        if j == 0:
            j += 1
            continue
        ForwardGradFixHook(conv_layer, conv_layer.weight, conv_layer.bias, conv_layer.padding)


def backward_fix_hook_model(model):
    hook_points = []
    get_conv_layers_step(model, hook_points)
    j = 0
    for conv_layer in hook_points:
        conv_layer.register_backward_hook(conv_backward_hook_grad_fix)



def plot_gradients(model_from, target_class=270, layer_str="layer2", idx=0, convolutions=None, image_save_path="imgs/saliency/"):
    hook_points = []
    get_conv_layers_step(model_from, hook_points)

    network = EvalSurrogateModel(model_from, hook_points=hook_points, surrogate_convolutions=convolutions, is_stitched_classifier=False)
    #network = model_from
    network = network.to("cuda")
    network = network.eval()

    #model_from = model_from.to("cuda")
    #model_from = model_from.eval()


    val_loader, val_loader_x_only = get_dataloader(batch_size=64, return_original_sample=True)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++print("find image from class...")
    x, x_original = find_img_of_class(val_loader, target_class, return_original_sample=True)
    x = x.to("cuda")

    if layer_str != "Input":
        layer = eval("network.classifier." + layer_str)
    else:
        layer = layer_str

    plot_gradients_sample(network, layer, target_class, x, x_original, idx,
                          folder_path=image_save_path+ layer_str)
    #plot_gradients_sample(network, network.classifier.features[3], target_class, x, x_original, summary_writer, idx)


def plot_gradients_sample(network, layer, target_class, x, x_original, idx=0, folder_path=""):

    #ig = captum.attr.IntegratedGradients(network.forward, multiply_by_inputs=True)
    #ig = captum.attr.InputXGradient(network.forward)
    #ig = captum.attr.Saliency(network.forward)
    #ig = captum.attr.DeepLift(network)
    #ig = captum.attr.LayerGradCam(network.forward, network.classifier.layer4)
    #ig = captum.attr.LayerIntegratedGradients(network.forward, layer, multiply_by_inputs=True)
    if layer == "Input":
        ig = captum.attr.DeepLift(network, multiply_by_inputs=True)
    else:
        ig = captum.attr.LayerDeepLift(network, layer, multiply_by_inputs=True)
    #ig = captum.attr.LayerGradientXActivation(network, layer, multiply_by_inputs=False)
    GradientAccessor = captum.attr.NoiseTunnel(ig)


    #grad = GradientAccessor.attribute(x, target=target_class, stdevs=0.2, nt_samples=20, nt_samples_batch_size=1, internal_batch_size=1)[0].detach().cpu()
    #grad = GradientAccessor.attribute(x, target=target_class)[0].detach().cpu()
    #grad = captum.attr.LayerAttribution.interpolate(grad.unsqueeze(dim=0), (224, 224)).squeeze(dim=0)

    #baseline = torch.tensor([-2.1179, -2.0357, -1.8044]).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).to("cuda") * torch.ones_like(x)
    baseline = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).to("cuda") * torch.ones_like(x)
    #grad = GradientAccessor.attribute(x, target=target_class, stdevs=0.2, nt_samples=20, nt_samples_batch_size=1)[0].detach().cpu()
    grad = ig.attribute(x, target=target_class, baselines=baseline)[0].detach().cpu()

    #raise RuntimeError

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 20))
    plt.imshow(torch.mean(grad, dim=[0]))
    #plt.colorbar()
    plt.axis('off')
    plt.savefig(folder_path + "grad_" + str(idx) + ".png")
    time.sleep(0.1)

    plt.close()


    x = x_original.cpu()[0].permute(1, 2, 0)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(x)
    plt.axis('off')
    plt.savefig(folder_path + "image_" + str(idx) + ".png")
    #plt.colorbar()
    time.sleep(0.1)
    plt.close()

    grad = torch.mean(grad, dim=0)
    overlay_mask = torch.clamp(grad, min=0.0, max=100000000000)
    overlay_mask = overlay_mask / torch.max(overlay_mask)
    #overlay_mask += 0.1
    overlay_mask = torch.clamp(overlay_mask, min=0.1, max=1.0)
    overlay_mask = torch.nn.functional.interpolate(overlay_mask.unsqueeze(dim=0).unsqueeze(dim=0), size=(x_original.shape[2], x_original.shape[3]))
    overlay = x_original * overlay_mask
    print(overlay.shape)

    fig = plt.figure(figsize=(20, 20))
    plt.imshow(overlay[0].permute(1, 2, 0))
    plt.axis('off')
    #plt.colorbar()
    plt.savefig(folder_path + "overlay_" + str(idx) + ".png")
    time.sleep(0.1)

    plt.close()



def get_test_accuracy(network, val_loader):
    sum_correct = 0
    total = 0
    for x, target in tqdm(val_loader, ascii=True):
        x = x.to("cuda")
        target = target.to("cuda")
        with torch.no_grad():
            pred = network(x)
            pred = torch.nn.functional.softmax(pred)
            pred = torch.argmax(pred, dim=1)

            #print("pred shape: {}".format(pred.shape))
            #print("target shape: {}".format(target.shape))

            sum_correct += torch.sum(pred == target)
            total += target.shape[0]

    print("accuracy: {}".format(sum_correct/total))



def test_all_tv_input_layer():
    model_names = ["resnet34"] + ["resnet50"] + ["resnet50_robust"]

    for i, model_name in enumerate(model_names):
        print("--------------- non hooked model ------------------------------")
        print("model: {}".format(model_name))
        model = load_model(model_name).eval()
        test_tv(model, None, None, True)
        print("---------------------------------------------------------------")

        print("--------------- hooked model ------------------------------")
        convolutions = torch.load("ckpts/"+ model_name +"/0010.pt")
        test_tv(model, None, convolutions, True)
        print("---------------------------------------------------------------")



def test_all_saliency_difference():
    model_names = ["resnet34" for i in range(3)] + ["resnet50" for i in range(3)] + ["resnet50_robust" for i in range(3)]
    layer_names = ["layer" + str(i+1) for i in range(3)] + ["layer" + str(i+1) for i in range(3)] + ["layer" + str(i+1) for i in range(3)]

    for i, model_name in enumerate(model_names):
        layer_name = layer_names[i]
        print("model: {}".format(model_name))
        print("layer: {}".format(layer_name))
        model = load_model(model_name).eval()
        convolutions = torch.load("ckpts/"+ model_name +"/0010.pt")
        hook_model(model, convolutions)
        wrapped_model = SoftmaxWrapper(model)
        layer = eval("wrapped_model.model." + layer_name)


        model_hooked = load_model(model_name).eval()
        backward_fix_hook_model(model_hooked)
        wrapped_hooked_model = SoftmaxWrapper(model_hooked)
        layer_hooked = eval("wrapped_hooked_model.model." + layer_name)

        test_saliency_difference(wrapped_model, wrapped_hooked_model, layer, layer_hooked)


def test_all_tv(method="DeepLift", run_original_model=True, use_grad_fix=False, use_forward_fix=False,
                use_bilinear_forward_fix=False, surrogate_downsampling=bilinear_downsample):
    model_names = ["resnet34" for i in range(3)] + ["resnet50" for i in range(3)] + ["resnet50_robust" for i in range(3)]
    layer_names = ["layer1[2]", "layer2[3]", "layer3[5]", "layer1[2]", "layer2[3]", "layer3[5]", "layer1[2]", "layer2[3]", "layer3[5]"]
    #layer_names = ["layer" + str(i+1) for i in range(3)] + ["layer" + str(i+1) for i in range(3)] + ["layer" + str(i+1) for i in range(3)]
    #layer_names = ["layer" + str(i+1) + "[0]" for i in range(3)] + ["layer" + str(i+1) + "[0]" for i in range(3)] + ["layer" + str(i+1) + "[0]" for i in range(3)]

    grad_fix_method = "original"

    model_names = ["resnet34" for i in range(3+4+6)] + ["resnet50" for i in range(3+4+6)] + ["resnet50_robust" for i in range(3+4+6)]
    layer_names = []
    block_number = [3, 4, 6]
    for i in range(3):
        for layer_idx in range(3):
            for block in range(block_number[layer_idx]):
                layer_name = "layer" + str(layer_idx+1) + "["+str(block)+"]"
                layer_names.append(layer_name)

    #layer_names = ["layer" + str(i+1) + "[" + str(i) + "]" for i in range(3)] + ["layer" + str(i+1) + "[0]" for i in range(3)] + ["layer" + str(i+1) + "[0]" for i in range(3)]


    #model_names = ["resnet50" for i in range(3)] + ["resnet50_robust" for i in range(3)]
    #layer_names = ["layer" + str(i+1) for i in range(3)] + ["layer" + str(i+1) for i in range(3)]

    mean_tv_original = []
    mean_tv_hooked = []

    for i, model_name in enumerate(model_names):
        layer_name = layer_names[i]
        print("model: {}".format(model_name))
        print("layer: {}".format(layer_name))
        model = load_model(model_name).eval()
        layer = eval("model." + layer_name)
        if run_original_model:
            print("--------------- non hooked model ------------------------------")
            mean_tv = test_tv(model, layer, None, method=method)
            mean_tv_original.append(mean_tv.cpu())
            print("---------------------------------------------------------------")

        print("--------------- hooked model ------------------------------")
        convolutions = torch.load("ckpts/"+ model_name +"/0010.pt")
        mean_tv = test_tv(model, layer, convolutions, method=method, use_grad_fix=use_grad_fix,
                use_forward_fix=use_forward_fix,
                surrogate_downsampling=surrogate_downsampling)
        mean_tv_hooked.append(mean_tv.cpu())
        print("---------------------------------------------------------------")
        #if i > 12:
        #    break

    if use_forward_fix:
        np.save("saved_np/imagenet_mean_tv_" + method + "_forward_hook.npy", mean_tv_hooked)
    elif use_grad_fix:
        np.save("saved_np/imagenet_mean_tv_" + method + "_backward_hook.npy", mean_tv_hooked)
    else:
        np.save("saved_np/imagenet_mean_tv_" + method + "_surrogate.npy", mean_tv_hooked)
    if run_original_model:
        np.save("saved_np/imagenet_mean_tv_" + method + "_original.npy", mean_tv_original)


    #np.save("saved_np/imagenet_mean_tv_original.npy", mean_tv_original)

    """
    fig = plt.figure(dpi=200)
    plt.plot([i for i in range(13)], mean_tv_original[:13], '--bo', label="Original")
    plt.plot([i for i in range(13)], mean_tv_hooked[:13], '--gv', label="Bilinear Surrogate")
    fig.legend(loc='outside upper center')
    plt.savefig("imgs/tv_plot_resnet34.png")
    plt.close()

    fig = plt.figure(dpi=200)
    plt.plot([i for i in range(13)], mean_tv_original[13:26], '--bo', label="Original")
    plt.plot([i for i in range(13)], mean_tv_hooked[13:26], '--gv', label="Bilinear Surrogate")
    fig.legend(loc='outside upper center')
    plt.savefig("imgs/tv_plot_resnet50.png")
    plt.close()

    fig = plt.figure(dpi=200)
    plt.plot([i for i in range(13)], mean_tv_original[26:], '--bo', label="Original")
    plt.plot([i for i in range(13)], mean_tv_hooked[26:], '--gv', label="Bilinear Surrogate")
    fig.legend(loc='outside upper center')
    plt.savefig("imgs/tv_plot_resnet50_robust.png")
    plt.close()
    """
    print("mean tv original: {}".format(mean_tv_original))
    print("mean tv hooked: {}".format(mean_tv_hooked))




def get_all_val_accuracies(run_original_model=True, use_forward_fix=False, surrogate_downsampling=bilinear_downsample):
    model_names = ["resnet34", "resnet50", "resnet50_robust"]

    for model_name in model_names:
        print("-------------------------------------------------------")
        print("model name: {}".format(model_name))
        print("-------------------------- original model -----------------------")
        model = load_model(model_name).eval()
        dataset = get_dataset(return_original_sample=False)
        if run_original_model:
            val_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=False)
            get_test_accuracy(model, val_loader)

        print("-------------------------- hooked model -----------------------")
        if use_forward_fix:
            forward_fix_hook_model(model)
        else:
            convolutions = torch.load("ckpts/"+ model_name +"/0010.pt")
            hook_model(model, convolutions, surrogate_downsampling=surrogate_downsampling)
        val_loader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=False)
        get_test_accuracy(model, val_loader)

        print("-------------------------------------------------------")


def get_all_val_pred_differences(target_class_only=False, use_forward_hook=False):
    model_names = ["resnet34", "resnet50", "resnet50_robust"]

    for model_name in model_names:
        print("-------------------------------------------------------")
        print("model name: {}".format(model_name))
        model = load_model(model_name).eval()
        model_original = copy.deepcopy(model)
        dataset = get_dataset(return_original_sample=False)
        #dataset = ImageNetDatasetWrapper(dataset)
        val_loader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=False)
        if use_forward_hook:
            forward_fix_hook_model(model)
        else:
            convolutions = torch.load("ckpts/"+ model_name +"/0010.pt")
            hook_model(model, convolutions)
        eval_pred_difference_imagenet(model_original, model, val_loader, target_class_only=target_class_only)


        print("-------------------------------------------------------")




def insertion_deletion_metrics(val_loader, saliency_maps, model, n_classes=1000, substrate_fn=torch.zeros_like):
    #images = np.empty((len(val_loader), batch_size, 3, image_size, image_size))
    images = None
    for j, (img, _) in enumerate(tqdm(val_loader, total=len(val_loader), desc='Loading images', ascii=True)):
        if images is None:
            batch_size, _, image_size, _ = img.shape
            images = np.empty((len(val_loader), batch_size, 3, image_size, image_size))
        images[j] = img
    images = images.reshape((-1, 3, image_size, image_size))

    step_fac = 1
    batch_size_del_ins_metrics = 100
    if image_size > 512:
        batch_size_del_ins_metrics = 2
        step_fac = 7

    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)

    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

    deletion = CausalMetric(model, 'del', image_size * 8*step_fac, substrate_fn=substrate_fn)
    insertion = CausalMetric(model, 'ins', image_size * 8*step_fac, substrate_fn=blur)

    h = deletion.evaluate(torch.from_numpy(images.astype('float32')), saliency_maps, batch_size_del_ins_metrics, n_classes=n_classes)
    del_mean_auc, del_std_auc = get_mean_std_auc(h)
    print("deletion mean auc: {}; std auc: {}".format(del_mean_auc, del_std_auc))

    #scores['del'].append(mean_auc)
    #scores['del_std'].append(std_auc)

    h = insertion.evaluate(torch.from_numpy(images.astype('float32')), saliency_maps, batch_size_del_ins_metrics, n_classes=n_classes)
    ins_mean_auc, ins_std_auc = get_mean_std_auc(h)
    print("insertion mean auc: {}; std auc: {}".format(ins_mean_auc, ins_std_auc))
    #scores['ins'].append(mean_auc)
    #scores['ins_std'].append(std_auc)

    return del_mean_auc, del_std_auc, ins_mean_auc, ins_std_auc




def run_insertion_deletion_metrics(saliency_method="DeepLift", smooth_grad=False, use_surrogate=True, layer_strings=["layer1", "layer2", "layer3"],
                                   backward_fix=False, forward_fix=False, saliency_computation_val_loader_bs=10, surrogate_explain_original=False,
                                   post_softmax=False):

    model_strings = ["resnet34", "resnet50", "resnet50_robust"]
    #layer_strings = ["layer1", "layer2", "layer3"]

    model_names = ["resnet34" for i in range(3)] + ["resnet50" for i in range(3)] + ["resnet50_robust" for i in range(3)]
    layer_names = ["layer1[2]", "layer2[3]", "layer3[5]", "layer1[2]", "layer2[3]", "layer3[5]", "layer1[2]", "layer2[3]", "layer3[5]"]


    for model_str in model_strings:
        scores = {'del': [], 'ins': [], 'del_std': [], 'ins_std': []}
        print("++++++++++++++++++++++++++++++++ {} ++++++++++++++++++++++++++++++++++++++++++++".format(model_str))
        for layer_str in layer_strings:
            print("++++++++++++++++++++++++++++++++ {} ++++++++++++++++++++++++++++++++++++++++++++".format(layer_str))
            val_loader = get_imagenet_val_ds(batch_size=saliency_computation_val_loader_bs)
            model = load_model(model_str).eval().to("cuda")
            model_test = load_model(model_str).eval().to("cuda")

            layer = eval("model." + layer_str)

            if use_surrogate:
                conv_layers = []
                get_conv_layers_step(model, conv_layers)
                convolutions = torch.load("/home/digipath2/projects/xai/papers/bilinear-surrogate/ckpts/" + model_str + "/0010.pt")
                EvalSurrogateModel(model, hook_points=conv_layers, surrogate_convolutions=convolutions, is_stitched_classifier=False)
                if not surrogate_explain_original:
                    EvalSurrogateModel(model_test, hook_points=conv_layers, surrogate_convolutions=convolutions, is_stitched_classifier=False)
            elif backward_fix:
                backward_fix_hook_model(model)
            elif forward_fix:
                forward_fix_hook_model(model)

            if post_softmax:
                model = nn.Sequential(model, nn.Softmax(dim=1))
            method = get_saliency_method(model, saliency_method, layer, use_smooth_grad=smooth_grad)
            saliency_maps = get_saliency_result(model, method, val_loader)

            #saliency_maps = torch.randn_like(saliency_maps)

            saliency_maps = F.interpolate(saliency_maps.unsqueeze(dim=1), size=224)[:, 0]

            val_loader = get_imagenet_val_ds(batch_size=10)
            #scores = {'del': [], 'ins': []}
            model_test = nn.Sequential(model_test, nn.Softmax(dim=1))
            del_mean_auc, del_std_auc, ins_mean_auc, ins_std_auc = insertion_deletion_metrics(val_loader, saliency_maps, model_test)
            scores['del'].append(del_mean_auc)
            scores['del_std'].append(del_std_auc)
            scores['ins'].append(ins_mean_auc)
            scores['ins_std'].append(ins_std_auc)

        #print(scores)
        #raise RuntimeError
        if use_surrogate:
            if surrogate_explain_original:
                np.save("results_post_softmax/auc_values_" + model_str + "_" + saliency_method + "_surrogate_explain_original" + ".npy", scores)
            else:
                np.save("results_post_softmax/auc_values_" + model_str + "_" + saliency_method + "_surrogate" + ".npy", scores)
        elif backward_fix:
            np.save("results_post_softmax/auc_values_" + model_str + "_" + saliency_method + "_backward_hook" + ".npy", scores)
        elif forward_fix:
            np.save("results_post_softmax/auc_values_" + model_str + "_" + saliency_method + "_forward_hook" + ".npy", scores)
        else:
            np.save("results_post_softmax/auc_values_" + model_str + "_" + saliency_method + ".npy", scores)

        torch.cuda.empty_cache()

    #np.save("saved_np/imagenet_mean_tv_forward_hook.npy", mean_tv_hooked)


def randomize_model(original_model, random_weights_model=None, layer_strings=None):
    for name1, param1 in original_model.named_parameters():
        #print(name1)
        replace_layer_params = False
        for layer_str in layer_strings:
            if layer_str in name1:
                replace_layer_params = True
                break

        if replace_layer_params:
            if name1 in random_weights_model.state_dict():  # Check for matching layer names
                param1.data.copy_(random_weights_model.state_dict()[name1])  # Replace weights



def run_insertion_deletion_metrics_randomized_model(model_name, saliency_method="DeepLift",
                                                    saliency_computation_val_loader_bs=10,
                                                    randomized_saliency_map=False,
                                                    layers_to_randomize = [[], ["fc"], ["layer4"], ["layer3"], ["layer2"], ["layer1"], "all"]):

    model_test = load_model(model_name)
    model_test = nn.Sequential(model_test, nn.Softmax(dim=1))

    layers = ["layer1", "layer2", "layer3", "layer4"]
    #layers_to_randomize = [[], ["fc"], ["layer4"], ["layer3"], ["layer2"], ["layer1"], "all"]

    for layer_str in layers:
        model_randomized = load_model(model_name, trained=False)
        model = load_model(model_name)
        layer = eval("model." + layer_str)
        scores = {'del': [], 'ins': [], 'del_std': [], 'ins_std': []}

        for layers_list_to_randomize in layers_to_randomize:
            print("layers to randomize: {}".format(layers_list_to_randomize))
            if layers_list_to_randomize == "all":
                model = model_randomized
                layer = eval("model_randomized." + layer_str)
            else:
                randomize_model(model, model_randomized, layers_list_to_randomize)

            method = get_saliency_method(model, saliency_method, layer, use_smooth_grad=False)
            val_loader = get_imagenet_val_ds(batch_size=saliency_computation_val_loader_bs)

            saliency_maps = get_saliency_result(model, method, val_loader)
            if randomized_saliency_map:
                saliency_maps = torch.randn_like(saliency_maps)
            saliency_maps = F.interpolate(saliency_maps.unsqueeze(dim=1), size=224)[:, 0]

            val_loader = get_imagenet_val_ds(batch_size=10)
            #scores = {'del': [], 'ins': []}
            del_mean_auc, del_std_auc, ins_mean_auc, ins_std_auc = insertion_deletion_metrics(val_loader, saliency_maps, model_test)
            scores['del'].append(del_mean_auc)
            scores['del_std'].append(del_std_auc)
            scores['ins'].append(ins_mean_auc)
            scores['ins_std'].append(ins_std_auc)

        if randomized_saliency_map:
            np.save("results_randomize_saliency_map/" + model_name + "_" + "_" + layer_str + "_" + saliency_method + ".npy", scores)
        else:
            np.save("results_randomize_model/" + model_name + "_" + "_" + layer_str + "_" + saliency_method + ".npy", scores)




def run_insertion_deletion_metrics_RISE(use_surrogate=True):
    model_strings = ["resnet34", "resnet50", "resnet50_robust"]

    scores = {'del': [], 'ins': [], 'del_std': [], 'ins_std': []}

    for model_str in model_strings:
        print("++++++++++++++++++++++++++++++++ {} ++++++++++++++++++++++++++++++++++++++++++++".format(model_str))
        model = load_model(model_str).eval().to("cuda")
        model_softmax = nn.Sequential(model, nn.Softmax(dim=1))
        explainer = RISE(model, (224, 224), gpu_batch=800)
        #explainer.generate_masks(N=5000, s=10, p1=0.1)
        explainer.generate_masks(N=8000, s=7, p1=0.1)
        if use_surrogate:
            conv_layers = []
            get_conv_layers_step(model_softmax, conv_layers)
            convolutions = torch.load("/home/digipath2/projects/xai/papers/bilinear-surrogate/ckpts/" + model_str + "/0010.pt")
            EvalSurrogateModel(model, hook_points=conv_layers, surrogate_convolutions=convolutions, is_stitched_classifier=False)

        saliency_maps = []
        val_loader = get_imagenet_val_ds(batch_size=1)
        with torch.no_grad():
            for img, _ in tqdm(val_loader, desc="RISE explain", ascii=True):
                sal = explainer(img.cuda())[1].cpu()
                saliency_maps.append(sal)
            saliency_maps = torch.stack(saliency_maps, dim=0)

        val_loader = get_imagenet_val_ds(batch_size=10)
        print("saliency maps shape: {}".format(saliency_maps.shape))
        del_mean_auc, del_std_auc, ins_mean_auc, ins_std_auc = insertion_deletion_metrics(val_loader, saliency_maps, model)
        scores['del'].append(del_mean_auc)
        scores['del_std'].append(del_std_auc)
        scores['ins'].append(ins_mean_auc)
        scores['ins_std'].append(ins_std_auc)

    print(scores)
    if use_surrogate:
        np.save("results/auc_values_" + "RISE" + "_surrogate" + ".npy", scores)
    else:
        np.save("results/auc_values_" + "RISE" + ".npy", scores)



def get_mean_std_auc(score_images):
    auc_values = []

    for image_idx in range(score_images.shape[1]):
        curve = score_images[:, image_idx]
        auc_value = auc(curve)

        auc_values.append(auc_value)


    return np.mean(auc_values), np.std(auc_values)



if __name__ == '__main__':
    get_all_val_accuracies(use_forward_fix=False)
    raise RuntimeError
    """
    #get_all_val_accuracies(use_forward_fix=False, surrogate_downsampling=average_pool_downsample, run_original_model=False)

    #print("-------------------------- Total Variation Gradient ------------------------")
    #test_all_tv(method="Grad", run_original_model=False, use_grad_fix=False, use_forward_fix=False, surrogate_downsampling=average_pool_downsample)

    #print("-------------------------- Total Variation DeepLift ------------------------")
    #test_all_tv(method="DeepLift", run_original_model=False, use_grad_fix=False, use_forward_fix=False, surrogate_downsampling=average_pool_downsample)

    #run_insertion_deletion_metrics_RISE(use_surrogate=False)
    #run_insertion_deletion_metrics_RISE(use_surrogate=True)

    #run_insertion_deletion_metrics(saliency_method="DeepLift", use_surrogate=False)


    #run_insertion_deletion_metrics(saliency_method="IG_Input", use_surrogate=False, layer_strings=["layer1"])
    #run_insertion_deletion_metrics(saliency_method="DeepLift_Input", use_surrogate=False, layer_strings=["layer1"])
    #run_insertion_deletion_metrics(saliency_method="Grad_Input", use_surrogate=False, layer_strings=["layer1"])

    run_insertion_deletion_metrics(saliency_method="GradCAM", use_surrogate=False, layer_strings=["layer1"])
    #run_insertion_deletion_metrics(saliency_method="GradCAM", use_surrogate=True, layer_strings=["layer1"])

    layer_strings = ["layer1[" + str(i) + "]" for i in range(3)] + ["layer2[" + str(i) + "]" for i in range(4)] + ["layer3[" + str(i) + "]" for i in range(6)] + ["layer4[" + str(i) + "]" for i in range(3)]

    #run_insertion_deletion_metrics(saliency_method="DeepLift", use_surrogate=False, layer_strings=layer_strings, backward_fix=False)
    #run_insertion_deletion_metrics(saliency_method="DeepLift", use_surrogate=False, layer_strings=layer_strings, forward_fix=True)

    #run_insertion_deletion_metrics(saliency_method="Grad", use_surrogate=False, layer_strings=layer_strings, backward_fix=True)
    #run_insertion_deletion_metrics(saliency_method="Grad", use_surrogate=False, layer_strings=layer_strings, forward_fix=True)

    #run_insertion_deletion_metrics(saliency_method="IG", use_surrogate=False, layer_strings=layer_strings, backward_fix=True,
    #                               saliency_computation_val_loader_bs=4)
    #run_insertion_deletion_metrics(saliency_method="IG", use_surrogate=False, layer_strings=layer_strings, forward_fix=True,
    #                               saliency_computation_val_loader_bs=4)

    #run_insertion_deletion_metrics(saliency_method="Grad", use_surrogate=True, layer_strings=layer_strings, surrogate_explain_original=True)




    #run_insertion_deletion_metrics_randomized_model("resnet34", saliency_method="DeepLift", randomized_saliency_map=True, layers_to_randomize=[[]])
    #run_insertion_deletion_metrics_randomized_model("resnet50", saliency_method="DeepLift", randomized_saliency_map=True, layers_to_randomize=[[]])

    #run_insertion_deletion_metrics_randomized_model("resnet34", saliency_method="IG", saliency_computation_val_loader_bs=2, randomized_saliency_map=True, layers_to_randomize=[[]])
    #run_insertion_deletion_metrics_randomized_model("resnet50", saliency_method="IG", saliency_computation_val_loader_bs=2, randomized_saliency_map=True, layers_to_randomize=[[]])

    #run_insertion_deletion_metrics_randomized_model("resnet34", saliency_method="Grad", randomized_saliency_map=True, layers_to_randomize=[[]])
    #run_insertion_deletion_metrics_randomized_model("resnet50", saliency_method="Grad", randomized_saliency_map=True, layers_to_randomize=[[]])


    #raise RuntimeError
    run_insertion_deletion_metrics_randomized_model("resnet50_robust", saliency_method="DeepLift", randomized_saliency_map=True)
    raise RuntimeError
    run_insertion_deletion_metrics_randomized_model("resnet34", saliency_method="DeepLift")
    run_insertion_deletion_metrics_randomized_model("resnet50", saliency_method="DeepLift")

    run_insertion_deletion_metrics_randomized_model("resnet34", saliency_method="IG", saliency_computation_val_loader_bs=2)
    run_insertion_deletion_metrics_randomized_model("resnet50", saliency_method="IG", saliency_computation_val_loader_bs=2)
    run_insertion_deletion_metrics_randomized_model("resnet50_robust", saliency_method="IG", saliency_computation_val_loader_bs=2)

    run_insertion_deletion_metrics_randomized_model("resnet34", saliency_method="Grad")
    run_insertion_deletion_metrics_randomized_model("resnet50", saliency_method="Grad")
    run_insertion_deletion_metrics_randomized_model("resnet50_robust", saliency_method="Grad")

    raise RuntimeError

    run_insertion_deletion_metrics(saliency_method="GradCAM", use_surrogate=False, layer_strings=["layer1"])
    run_insertion_deletion_metrics(saliency_method="IG_Input", use_surrogate=False, layer_strings=["layer1"])
    run_insertion_deletion_metrics(saliency_method="DeepLift_Input", use_surrogate=False, layer_strings=["layer1"])

    run_insertion_deletion_metrics(saliency_method="DeepLift", use_surrogate=False, layer_strings=layer_strings)
    run_insertion_deletion_metrics(saliency_method="IG", use_surrogate=False, layer_strings=layer_strings,
                                   saliency_computation_val_loader_bs=4)

    run_insertion_deletion_metrics(saliency_method="DeepLift", use_surrogate=True, layer_strings=layer_strings)

    run_insertion_deletion_metrics(saliency_method="IG", use_surrogate=True, layer_strings=layer_strings,
                                   saliency_computation_val_loader_bs=4)


    run_insertion_deletion_metrics(saliency_method="Grad", use_surrogate=False, layer_strings=layer_strings)
    run_insertion_deletion_metrics(saliency_method="Grad", use_surrogate=True, layer_strings=layer_strings)



    raise RuntimeError
    """


    # --------------------- plot samples ------------------------------------
    layers = ["Input", "layer1", "layer2", "layer3"]
    for layer in layers:
        for i, target_class in enumerate([99, 207, 292, 294]):
            model = load_model("resnet34").eval()
            #backward_fix_hook_model(model)
            #forward_fix_hook_model(model)
            plot_gradients(model, target_class=target_class, idx=i, convolutions=None, layer_str=layer, image_save_path="imgs/saliency/original/")

    layers = ["Input", "layer1", "layer2", "layer3"]
    for layer in layers:
        for i, target_class in enumerate([99, 207, 292, 294]):
            model = load_model("resnet34").eval()
            backward_fix_hook_model(model)
            #forward_fix_hook_model(model)
            plot_gradients(model, target_class=target_class, idx=i, convolutions=None, layer_str=layer, image_save_path="imgs/saliency/backward_hooked/")


    layers = ["Input", "layer1", "layer2", "layer3"]
    for layer in layers:
        for i, target_class in enumerate([99, 207, 292, 294]):
            model = load_model("resnet34").eval()
            #backward_fix_hook_model(model)
            forward_fix_hook_model(model)
            plot_gradients(model, target_class=target_class, idx=i, convolutions=None, layer_str=layer, image_save_path="imgs/saliency/forward_hooked/")


    layers = ["Input", "layer1", "layer2", "layer3"]
    for layer in layers:
        for i, target_class in enumerate([99, 207, 292, 294]):
            model = load_model("resnet34").eval()
            convolutions = torch.load("ckpts/resnet34/0010.pt")
            #backward_fix_hook_model(model)
            #forward_fix_hook_model(model)
            plot_gradients(model, target_class=target_class, idx=i, convolutions=convolutions, layer_str=layer, image_save_path="imgs/saliency/bilinear_surrogate/")



    # ------------------ run tv ---------------------------
    methods = ["Grad", "DeepLift", "IG"]

    for method in methods:
        print("------------------- bilinear surrogate ----------------------")
        test_all_tv(method=method, run_original_model=True, use_grad_fix=False, use_forward_fix=False)
        print("------------------- backward hook ----------------------")
        test_all_tv(method=method, run_original_model=False, use_grad_fix=True, use_forward_fix=False)
        print("------------------- forward hook ----------------------")
        test_all_tv(method=method, run_original_model=False, use_grad_fix=False, use_forward_fix=True)


    print("--------------------- prediction difference -----------------------")
    print("-------------------- sum of all classes ---------------------------")
    print("-------------------- bilinear surrogate ---------------------------")
    get_all_val_pred_differences(target_class_only=False, use_forward_hook=False)
    print("-------------------- forward hooked model ---------------------------")
    get_all_val_pred_differences(target_class_only=False, use_forward_hook=False)

    print("-------------------- only target class ---------------------------")
    print("-------------------- bilinear surrogate ---------------------------")
    get_all_val_pred_differences(target_class_only=True, use_forward_hook=False)
    print("-------------------- forward hooked model ---------------------------")
    get_all_val_pred_differences(target_class_only=True, use_forward_hook=False)

    print("------------------ validation accuracy -------------------------")
    print("------------------ bilinear surrogate -------------------------")
    get_all_val_accuracies(use_forward_fix=False)
    print("------------------ forward hooked -------------------------")
    get_all_val_accuracies(use_forward_fix=True)


    methods = ["DeepLift", "Grad", "IG"]

    for method in methods:
        layer_strings = ["layer1[" + str(i) + "]" for i in range(3)] + ["layer2[" + str(i) + "]" for i in range(4)] + ["layer3[" + str(i) + "]" for i in range(6)] + ["layer4[" + str(i) + "]" for i in range(3)]

        # for DeepLift for the original model
        run_insertion_deletion_metrics(saliency_method=method, use_surrogate=False, layer_strings=layer_strings, backward_fix=False, forward_fix=False)
        # for the surrogate model
        run_insertion_deletion_metrics(saliency_method=method, use_surrogate=True, layer_strings=layer_strings, backward_fix=False, forward_fix=False)
        # backward hooked
        run_insertion_deletion_metrics(saliency_method=method, use_surrogate=False, layer_strings=layer_strings, backward_fix=True, forward_fix=False)
        # forward hooked
        run_insertion_deletion_metrics(saliency_method=method, use_surrogate=False, layer_strings=layer_strings, backward_fix=False, forward_fix=True)

    # GradCAM, layer_strings is not used here but should have one element for not crashing the code
    run_insertion_deletion_metrics(saliency_method="GradCAM", use_surrogate=False, layer_strings=["layer1"])
    # run for the input layer, layer_strings is not used here but should have one element for not crashing the code
    run_insertion_deletion_metrics(saliency_method="DeepLift_Input", use_surrogate=False, layer_strings=["layer1"])
    run_insertion_deletion_metrics(saliency_method="Grad_Input", use_surrogate=False, layer_strings=["layer1"])
    run_insertion_deletion_metrics(saliency_method="IG_Input", use_surrogate=False, layer_strings=["layer1"])

