
import torch
from monai.networks.nets import TorchVisionFCModel
import monai
import datetime
from torch.utils.tensorboard import SummaryWriter
from eval_ import plot_gradients_sample, backward_fix_hook_model, forward_fix_hook_model, SurrogateModelInputHook, get_saliency_method, get_saliency_result, insertion_deletion_metrics, evaluate_tv
from core import EvalSurrogateModel
from eval_tv import eval_pred_difference
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import time



def load_model():
    model = TorchVisionFCModel(pretrained=False, use_conv=True)

    state_dict = torch.load("pathology_tumor_detection/models/model.pt")#.to("cuda").eval()

    model.load_state_dict(state_dict)
    model = model.to("cuda").eval()
    #print(model)
    #print(model_load)

    return model

"""
def load_dataset(transform=None):
    #data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/training/tumor/tumor_001.tif"}]
    data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/testing/images/test_011.tif"}]
    dataset = monai.data.MaskedPatchWSIDataset(data, patch_size=[224, 224], mask_level=6, transform=transform)

    return dataset
"""


def load_dataset(transform=None):
    # index at 49 was not present, skip it
    val_indice = [i+1 for i in range(48)] + [i+50 for i in range(80)]
    data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/testing/images/test_" + str(i).zfill(3) + ".tif"} for i in val_indice]
    dataset = monai.data.MaskedPatchWSIDataset(data, patch_size=[224, 224], mask_level=6, transform=transform)

    return dataset


def get_tumor_dataset():
    if not os.path.exists("camelyon_tumor_pred_samples.pt"):
        model = load_model()
        tumor_pred_imgs = []
        transform = monai.transforms.Compose([
            monai.transforms.CastToTyped(keys="image", dtype="float32"),
            monai.transforms.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=-1.0, b_max=1.0),
            monai.transforms.ToTensord(keys="image"),
        ])

        dataset = load_dataset(transform)
        torch.manual_seed(42)
        data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1024, shuffle=True, num_workers=16, pin_memory=False)

        for data in data_loader:
            print("num tumor samples: {}".format(len(tumor_pred_imgs)))
            images = data['image'].to("cuda")
            tumor_pred_imgs_in_batch = get_tumor_pred_imgs(model, images, n_samples_wanted=500)
            if len(tumor_pred_imgs_in_batch) > 0:
                if len(tumor_pred_imgs) == 0:
                    tumor_pred_imgs = tumor_pred_imgs_in_batch
                else:
                    tumor_pred_imgs = torch.cat([tumor_pred_imgs, tumor_pred_imgs_in_batch], dim=0)
                #tumor_pred_imgs.extend(tumor_pred_imgs_in_batch)
                if len(tumor_pred_imgs) >= 500:
                    tumor_pred_imgs = tumor_pred_imgs[:500]
                    torch.save(tumor_pred_imgs, "camelyon_tumor_pred_samples.pt")

                    #raise RuntimeError
                    return torch.utils.data.TensorDataset(tumor_pred_imgs)

    else:
        tumor_pred_imgs = torch.load("camelyon_tumor_pred_samples.pt").cpu()
        return torch.utils.data.TensorDataset(tumor_pred_imgs, torch.ones(len(tumor_pred_imgs)))








def load_dataset_for_tv(transform=None, num_samples=500):
    return get_tumor_dataset()

    val_indice = [i+1 for i in range(48)] + [i+50 for i in range(80)]
    data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/testing/images/test_" + str(i).zfill(3) + ".tif"} for i in val_indice]
    dataset = monai.data.MaskedPatchWSIDataset(data, patch_size=[224, 224], mask_level=6, transform=transform)

    indice = np.load("camelyon_val_indice_from_full_test_data.npy")
    dataset = torch.utils.data.Subset(dataset, indice)


    """
    indice = []
    while len(indice) < num_samples:
        idx = random.randint(0, len(dataset))
        if not idx in indice:
            indice.append(idx)
    np.save("camelyon_val_indice_from_full_test_data.npy", indice)


    #print("len dataset: {}".format(len(dataset)))

    raise RuntimeError
    """


    return dataset




def load_dataset_for_pred_difference(transform=None, num_samples=50000):
    val_indice = [i+1 for i in range(48)] + [i+50 for i in range(80)]
    data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/testing/images/test_" + str(i).zfill(3) + ".tif"} for i in val_indice]
    dataset = monai.data.MaskedPatchWSIDataset(data, patch_size=[224, 224], mask_level=6, transform=transform)

    return dataset



def get_tumor_samples(model, n_samples_wanted=8):
    """
    transform = monai.transforms.Compose([
        monai.transforms.CastToTyped(keys="image", dtype="float32"),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=-1.0, b_max=1.0),
        monai.transforms.ToTensord(keys="image"),
    ])

    data = [{"image": "/home/digipath2/data/Camelyon/CAMELYON16/training/tumor/tumor_" + str(i+1).zfill(3) + ".tif"} for i in range(3)]
    dataset = monai.data.MaskedPatchWSIDataset(data, patch_size=[224, 224], mask_level=6, transform=transform)
    """

    dataset = get_tumor_dataset()

    torch.manual_seed(42)
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=True, num_workers=16, pin_memory=False)

    for data in data_loader:
        images = data[0].to("cuda")
        #images = data['image'].to("cuda")
        return get_tumor_pred_imgs(model, images, n_samples_wanted=n_samples_wanted)




def get_tumor_pred_imgs(model, batch_data, n_samples_wanted=8):
    tumor_samples = []
    with torch.no_grad():
        preds = model(batch_data)

    for i, pred in enumerate(preds):
        # pred is in logits before sigmoid, so 0.0 actually means 50% (0.5 for tumor)
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


class CamelyonDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)[0]#['image']
        return torch.tensor(data), 0


class ModelOutputLogitsWrapper(torch.nn.Module):
    def __init__(self, model, logits_hook_point):
        super().__init__()
        self.model = model
        self.logits_hook_point = logits_hook_point
        self.hook = self.logits_hook_point.register_forward_hook(self.logits_hook_func)

    def logits_hook_func(self, module, input_, output):
        self.logits_output = output


    def forward(self, x):
        self.model(x)
        return self.logits_output





def plot_gradient_samples(tumor_samples, model, folder_path="imgs/saliency/camelyon16/"):#(train_loader, model):
    """
    tumor_imgs = []
    for image in train_loader:
        #image = data['image'].float()
        #image = image/127.5 - 1.0
        with torch.no_grad():
            tumor_samples = get_tumor_pred_imgs(image.to("cuda"))
        break
    """
    """
    model_original = load_model()

    pred = network(tumor_samples)
    pred_original = model_original(tumor_samples)

    print(pred)
    print(pred_original)
    raise RuntimeError
    """
    #torchvision.utils.save_image(tumor_samples.float()*0.5 + 0.5, "imgs/camelyon16_samples.jpg")

    #print(model(tumor_samples))

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    for feature in [4, 5, 6, 7]:


        for i, tumor_sample in enumerate(tumor_samples):
            tumor_sample = torch.tensor(tumor_sample.unsqueeze(dim=0))
            #print(tumor_samples.shape)
            plot_gradients_sample(model, model.model.features[feature], None, tumor_sample, torch.clone(tumor_sample*0.5 + 0.5).cpu(), idx=i, folder_path=folder_path + str(feature) + "_")


    for i, tumor_sample in enumerate(tumor_samples):
        tumor_sample = torch.tensor(tumor_sample.unsqueeze(dim=0))
        #print(tumor_samples.shape)
        plot_gradients_sample(model, model.identity_layer_for_input_hook, None, tumor_sample, torch.clone(tumor_sample*0.5 + 0.5).cpu(), idx=i, folder_path=folder_path + str("Input") + "_")


        #summary_writer.close()





def run_tv():
    model = load_model()
    convolutions = torch.load("ckpts/chamelyon_resnet18/0002.pt")

    conv_downsample_modules = []
    get_conv_layers_step(model, conv_downsample_modules)

    #print("len conv downsamplings: {}".format(len(conv_downsample_modules)))
    #raise RuntimeError

    network = EvalSurrogateModel(model, hook_points=conv_downsample_modules, surrogate_convolutions=convolutions, is_stitched_classifier=False)
    network = network.to("cuda")
    network = network.eval()

    transform = monai.transforms.Compose([
        monai.transforms.CastToTyped(keys="image", dtype="float32"),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=-1.0, b_max=1.0),
        monai.transforms.ToTensord(keys="image"),
    ])


    #return_nodes = {"fc" : 'output'}
    #model = create_feature_extractor(model, return_nodes=return_nodes)

    model = ModelOutputLogitsWrapper(model, model.fc)

    dataset = load_dataset(transform=transform)

    dataset = CamelyonDatasetWrapper(dataset)

    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=True, num_workers=16, pin_memory=False)


    train_loader_tv = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)



    torch.manual_seed(42)

    #plot_gradient_samples(train_loader, model)
    evaluate_tv(model, train_loader_tv, layer=model.model.features[5])



def run_pred_difference(forward_hook_model=False):
    model = load_model()

    if forward_hook_model:
        print("forward hooked model!")
        forward_fix_hook_model(model)
    else:
        print("no hook")
        convolutions = torch.load("ckpts/chamelyon_resnet18/0010.pt")

        conv_downsample_modules = []
        get_conv_layers_step(model, conv_downsample_modules)
        network = EvalSurrogateModel(model, hook_points=conv_downsample_modules, surrogate_convolutions=convolutions, is_stitched_classifier=False)
        network = network.to("cuda")
        network = network.eval()

    #print("len conv downsamplings: {}".format(len(conv_downsample_modules)))
    #raise RuntimeError


    transform = monai.transforms.Compose([
        monai.transforms.CastToTyped(keys="image", dtype="float32"),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=-1.0, b_max=1.0),
        monai.transforms.ToTensord(keys="image"),
    ])

    model_original = load_model()


    #dataset = get_tumor_dataset()
    dataset = load_dataset_for_pred_difference(transform=transform)

    dataset = CamelyonDatasetWrapper(dataset)

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=False)


    torch.manual_seed(42)

    eval_pred_difference(model_original, model, dataloader)

    #plot_gradient_samples(train_loader, model)
    #run_tv(model, train_loader_tv, layer=model.model.features[5])


#run_pred_difference()





def test_all_tv(method="DeepLift", use_grad_fix=False, use_forward_hook=False, use_surrogate=False):
    transform = monai.transforms.Compose([
        monai.transforms.CastToTyped(keys="image", dtype="float32"),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=-1.0, b_max=1.0),
        monai.transforms.ToTensord(keys="image"),
    ])
    dataset = load_dataset_for_tv(transform=transform)
    dataset = CamelyonDatasetWrapper(dataset)

    grad_fix_method = "original"

    #layers = ['features[4]', 'features[5]', 'features[6]']
    layers = ['features[4][0]', 'features[4][1]', 'features[5][0]', 'features[5][1]', 'features[6][0]', 'features[6][1]']

    total_variation_layers = []

    for layer_str in layers:
        dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

        model = load_model()
        torch.manual_seed(42)

        """
        print("------------------------ layer: {} ----------------------------------------------".format(layer))
        print("------------------------- non hooked model --------------------------------------")
        print("---------------------------------------------------------------------------------")
        evaluate_tv(model, dataloader, layer=eval('model.' + layer), method=method)
        print("---------------------------------------------------------------------------------")

        """

        #print("len conv downsamplings: {}".format(len(conv_downsample_modules)))
        #raise RuntimeError

        if use_grad_fix:
            backward_fix_hook_model(model)
            network = model
            grad_fix_method = "backward_hook"
        elif use_forward_hook:
            forward_fix_hook_model(model)
            network = model
            grad_fix_method = "forward_hook"
        elif use_surrogate:
            grad_fix_method = "surrogate"
            convolutions = torch.load("ckpts/chamelyon_resnet18/0010.pt")

            conv_downsample_modules = []
            get_conv_layers_step(model, conv_downsample_modules)
            network = EvalSurrogateModel(model, hook_points=conv_downsample_modules, surrogate_convolutions=convolutions, is_stitched_classifier=False)
        else:
            network = model
        network = network.to("cuda")
        network = network.eval()

        tv_hook_layer = eval("model." + layer_str)
        saliency_module = get_saliency_method(network, method, tv_hook_layer, input_layer=False)
        mean_tv = evaluate_tv(saliency_module, dataloader)

        total_variation_layers.append(mean_tv.cpu())



        """
        torch.manual_seed(42)
        print("------------------------- hooked model --------------------------------------")
        print("---------------------------------------------------------------------------------")
        #evaluate_tv(model, dataloader, layer=eval('model.' + layer), method=method)
        evaluate_tv(model, dataloader, layer=eval('model.' + layer), method=method)
        print("---------------------------------------------------------------------------------")
        """

    np.save("saved_np/mean_tv_camelyon16_" + method + "_" + grad_fix_method + ".npy", total_variation_layers)



def run_insertion_deletion_metrics_directly(model, model_test, layers, dataset, saliency_method, grad_cam_layer_override,
                                            save_path="results_camelyon/resnet18.npy", interp_size=224, batch_size=10,
                                            n_classes=1000, substrate_fn=torch.zeros_like):
    scores = {'del': [], 'ins': [], 'del_std': [], 'ins_std': []}
    val_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=batch_size)
    # = model[0].features[-1]
    for layer in layers:
        method = get_saliency_method(model, saliency_method, layer, use_smooth_grad=False, grad_cam_layer_override=grad_cam_layer_override)
        saliency_maps = get_saliency_result(model, method, val_loader)
        saliency_maps = F.interpolate(saliency_maps.unsqueeze(dim=1), size=interp_size)[:, 0]

        del_mean_auc, del_std_auc, ins_mean_auc, ins_std_auc = insertion_deletion_metrics(val_loader, saliency_maps, model_test, n_classes=n_classes,
                                                                                          substrate_fn=substrate_fn)
        scores['del'].append(del_mean_auc)
        scores['del_std'].append(del_std_auc)
        scores['ins'].append(ins_mean_auc)
        scores['ins_std'].append(ins_std_auc)
        #break

    print("save path: {}".format(save_path))
    np.save(save_path, scores)




#get_tumor_dataset()

def eval_insertion_deletion_metrics():

    model = load_model()
    model_for_saliency = load_model()
    layers = []
    for i in range(4):
        layers.append(model_for_saliency.features[i+4][0])
        layers.append(model_for_saliency.features[i+4][1])

    model_for_saliency = nn.Sequential(model_for_saliency, nn.Flatten(1))
    model = nn.Sequential(model, nn.Flatten(1), nn.Sigmoid())
    tumor_dataset = get_tumor_dataset()

    methods = ["IG_Input", "DeepLift_Input", "Grad_Input"]#["GradCAM", "DeepLift_Input", "IG_Input"]
    # use dummy layer [model.features[4][0]], is overriden in the saliency method
    save_folder = "results_camelyon/"
    for saliency_method in methods:
        save_path = save_folder + "resnet18_" + saliency_method + ".npy"
        run_insertion_deletion_metrics_directly(model_for_saliency, model, [model[0].features[4][0]], tumor_dataset, saliency_method, grad_cam_layer_override=model[0].features[-1], save_path=save_path)

    time.sleep(3.0)


    #raise RuntimeError
    methods = ["DeepLift", "Grad", "IG"]
    save_folder = "results_camelyon/"

    layers = []
    for i in range(4):
        layers.append(model_for_saliency.features[i+4][0])
        layers.append(model_for_saliency.features[i+4][1])

    conv_layers = []
    get_conv_layers_step(model_for_saliency, conv_layers)
    convolutions = torch.load("ckpts/" + "chamelyon_resnet18" + "/0010.pt") #torch.load("/home/digipath2/projects/xai/papers/bilinear-surrogate/ckpts/" + "chamelyon_resnet18" + "/0002.pt")
    EvalSurrogateModel(model_for_saliency, hook_points=conv_layers, surrogate_convolutions=convolutions, is_stitched_classifier=False)
    #model = nn.Sequential(model_for_saliency, nn.Sigmoid())
    for saliency_method in methods:
        save_path = save_folder + "resnet18_" + saliency_method + "_surrogate.npy"
        run_insertion_deletion_metrics_directly(model_for_saliency, model_for_saliency, layers, tumor_dataset, saliency_method, save_path=save_path, grad_cam_layer_override=model[0].features[-1])


    model = load_model()
    model_for_saliency = load_model()
    layers = []
    for i in range(4):
        layers.append(model_for_saliency.features[i+4][0])
        layers.append(model_for_saliency.features[i+4][1])

    model_for_saliency = nn.Sequential(model_for_saliency, nn.Flatten(1))
    model = nn.Sequential(model, nn.Flatten(1), nn.Sigmoid())
    backward_fix_hook_model(model_for_saliency)
    for saliency_method in methods:
        save_path = save_folder + "resnet18_" + saliency_method + "_backward_hook.npy"
        run_insertion_deletion_metrics_directly(model_for_saliency, model, layers, tumor_dataset, saliency_method, save_path=save_path, grad_cam_layer_override=model[0].features[-1])


    model = load_model()
    model_for_saliency = load_model()
    layers = []
    for i in range(4):
        layers.append(model_for_saliency.features[i+4][0])
        layers.append(model_for_saliency.features[i+4][1])

    model_for_saliency = nn.Sequential(model_for_saliency, nn.Flatten(1))
    model = nn.Sequential(model, nn.Flatten(1), nn.Sigmoid())
    forward_fix_hook_model(model_for_saliency)
    for saliency_method in methods:
        save_path = save_folder + "resnet18_" + saliency_method + "_forward_hook.npy"
        run_insertion_deletion_metrics_directly(model_for_saliency, model, layers, tumor_dataset, saliency_method, save_path=save_path, grad_cam_layer_override=model[0].features[-1])


if __name__ == "__main__":



    """


    #eval_insertion_deletion_metrics()
    #raise RuntimeError
    """
    model = load_model()
    convolutions = torch.load("ckpts/chamelyon_resnet18/0010.pt")

    conv_downsample_modules = []
    get_conv_layers_step(model, conv_downsample_modules)

    network = EvalSurrogateModel(model, hook_points=conv_downsample_modules, surrogate_convolutions=convolutions, is_stitched_classifier=False)
    network = network.to("cuda")
    network = network.eval()

    input_surrogate_model = SurrogateModelInputHook(model)


    tumor_samples = get_tumor_samples(model, n_samples_wanted=4)

    plot_gradient_samples(tumor_samples, input_surrogate_model, folder_path="imgs/saliency/camelyon16/bilinear_")

    model = load_model()
    input_surrogate_model = SurrogateModelInputHook(model)
    plot_gradient_samples(tumor_samples, input_surrogate_model, folder_path="imgs/saliency/camelyon16/original_")




    methods = ["Grad", "DeepLift", "IG"]

    for method in methods:

        print("-------------------- Grad --------------------------")
        print("-------------------- original --------------------------")
        test_all_tv(method=method, use_grad_fix=False, use_forward_hook=False)
        print("-------------------- bilinear surrogate --------------------------")
        test_all_tv(method=method, use_grad_fix=False, use_forward_hook=False, use_surrogate=True)
        print("-------------------- backward hook --------------------------")
        test_all_tv(method=method, use_grad_fix=True, use_forward_hook=False)
        print("-------------------- forward hook --------------------------")
        test_all_tv(method=method, use_grad_fix=False, use_forward_hook=True)


    print("-------------------- prediction difference -----------------------")
    print("------------------- bilinear surrogate -------------------------")
    run_pred_difference(forward_hook_model=False)
    print("------------------- forward hooked -------------------------")
    run_pred_difference(forward_hook_model=True)

    eval_insertion_deletion_metrics()
