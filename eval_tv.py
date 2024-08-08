
from torchmetrics.functional.image import total_variation
from tqdm import tqdm
import captum
import torch
import torch.nn.functional as F

def evaluate_tv(model, dataloader, layer, target_class=None, method="DeepLift"):
    total_tv = []
    i = 0
    for data in tqdm(dataloader, ascii=True):
        data = data.to("cuda")
        if method == "DeepLift":
            ig = captum.attr.LayerDeepLift(model, layer, multiply_by_inputs=True)
        elif method == "Grad":
            ig = captum.attr.LayerGradientXActivation(model, layer, multiply_by_inputs=False)
        GradientAccessor = captum.attr.NoiseTunnel(ig)
        grad = GradientAccessor.attribute(data, target=target_class, stdevs=0.2, nt_samples=20, nt_samples_batch_size=1).detach()

        with torch.no_grad():
            grad_scaled = grad - torch.mean(grad, dim=[2, 3]).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grad_scaled = grad_scaled / torch.max(torch.mean(torch.abs(grad_scaled), dim=[2, 3]).unsqueeze(dim=-1).unsqueeze(dim=-1), torch.tensor(0.000001).to("cuda"))
            tv = total_variation(grad_scaled, reduction='sum')
            tv_pixelwise_mean = tv / (grad.shape[1]*grad.shape[2] * grad.shape[3])
            #print("tv shape: {}".format(tv.shape))
            #raise RuntimeError
            total_tv.append(tv_pixelwise_mean)

            #total_tv.append(total_variation(grad))
        i += 1

        #if i > 10:
        #    break

    total_tv = torch.stack(total_tv, dim=0)
    mean_tv = torch.mean(total_tv)
    std_tv = torch.std(total_tv)

    print("mean_tv: {}".format(mean_tv))
    print("std_tv: {}".format(std_tv))




def eval_pred_difference(model_original, model_hooked, dataloader, post_process_function=torch.nn.functional.sigmoid):

    pred_differences = []

    i = 0
    for data in tqdm(dataloader, ascii=True):
        data = data.to("cuda")

        with torch.no_grad():
            preds_original = post_process_function(model_original(data))
            preds_hooked = post_process_function(model_hooked(data))

            #print(torch.min(preds_original))
            #print(torch.max(preds_original))

            l1_loss = torch.abs(preds_hooked - preds_original)
            l1_loss = torch.mean(l1_loss, dim=[2, 3])
            l1_loss = torch.sum(l1_loss, dim=[1])

            for loss in l1_loss:
                pred_differences.append(loss)

        i += 1

        #if i > 100:
        #    break

    total = torch.stack(pred_differences, dim=0)
    mean_ = torch.mean(total)
    std_ = torch.std(total)

    print("mean_ difference: {}".format(mean_))
    print("std_ difference: {}".format(std_))


def eval_pred_difference_highest_class(model_original, model_hooked, dataloader, post_process_function=torch.nn.functional.sigmoid):

    pred_differences = []
    sum_masks = []

    i = 0
    for data in tqdm(dataloader, ascii=True):
        data = data.to("cuda")

        with torch.no_grad():
            preds_original = post_process_function(model_original(data))
            preds_hooked = post_process_function(model_hooked(data))

            preds_original_diagnoses = torch.mean(preds_original, dim=[0, 2, 3])
            preds_original_diagnoses[0] = -1.0                               # ignore normal class
            max_predicted_diagnose = torch.argmax(preds_original_diagnoses)

            print(torch.min(preds_original))
            print(torch.max(preds_original))

            l1_loss = torch.abs(preds_hooked[:, max_predicted_diagnose] - preds_original[:, max_predicted_diagnose])
            l1_loss_mask = torch.where(preds_original[:, max_predicted_diagnose] > 0.1, torch.ones_like(l1_loss), torch.zeros_like(l1_loss))
            l1_loss = torch.sum(l1_loss*l1_loss_mask, dim=[1, 2])
            l1_loss_mask = torch.sum(l1_loss_mask, dim=[1, 2])
            l1_loss_mask = torch.where(l1_loss_mask==0.0, 1.0, l1_loss_mask)
            l1_loss /= l1_loss_mask

            for i, loss in enumerate(l1_loss):
                pred_differences.append(loss)
                #sum_masks.append(l1_loss_mask[i])

        i += 1

        #if i > 10:
        #    break

    total = torch.stack(pred_differences, dim=0)
    mean_ = torch.mean(total)
    std_ = torch.std(total)

    print("mean_ difference: {}".format(mean_))
    print("std_ difference: {}".format(std_))




def eval_pred_difference_highest_class_using_gt(model_original, model_hooked, dataloader, post_process_function=torch.nn.functional.sigmoid):

    pred_differences = []
    sum_masks = []

    k = 0
    for data, target in tqdm(dataloader, ascii=True):
        data = data.to("cuda")
        target = target.to("cuda")

        target = torch.nn.functional.one_hot(target.long())
        target = target.permute(dims=[0, 3, 1, 2])

        with torch.no_grad():
            preds_original = post_process_function(model_original(data))
            preds_hooked = post_process_function(model_hooked(data))

            max_target_diagnose = torch.mean(target.float(), dim=[0, 2, 3])
            max_target_diagnose[0] = -1.0                               # ignore normal class
            max_target_diagnose = torch.argmax(max_target_diagnose)

            #print(torch.min(preds_original))
            #print(torch.max(preds_original))

            l1_loss = torch.abs(preds_hooked[:, max_target_diagnose] - preds_original[:, max_target_diagnose])
            l1_loss_mask = torch.where(preds_original[:, max_target_diagnose] > 0.1, torch.ones_like(l1_loss), torch.zeros_like(l1_loss))
            l1_loss = torch.sum(l1_loss*l1_loss_mask, dim=[1, 2])
            l1_loss_mask = torch.sum(l1_loss_mask, dim=[1, 2])
            l1_loss_mask_non_zero = torch.where(l1_loss_mask==0.0, 1.0, l1_loss_mask)
            l1_loss /= l1_loss_mask_non_zero

            for i, loss in enumerate(l1_loss):
                if torch.sum(l1_loss_mask[i]) > 1.0:
                    pred_differences.append(loss)
                #sum_masks.append(l1_loss_mask[i])

        k += 1

        if k > 100:
            break

    total = torch.stack(pred_differences, dim=0)
    mean_ = torch.mean(total)
    std_ = torch.std(total)

    print("mean_ difference: {}".format(mean_))
    print("std_ difference: {}".format(std_))




def eval_pred_difference_imagenet(model_original, model_hooked, dataloader, target_class_only=False):

    pred_differences = []

    i = 0
    for data, targets in tqdm(dataloader, ascii=True):
        data = data.to("cuda")

        with torch.no_grad():
            preds_original = F.softmax(model_original(data), dim=1)
            preds_hooked = F.softmax(model_hooked(data), dim=1)

            if target_class_only:
                l1_loss = []
                for i, pred in enumerate(preds_original):
                    target = targets[i]
                    pred_hooked = preds_hooked[i]
                    loss = torch.sum(torch.abs(pred_hooked[target] - pred[target]))
                    l1_loss.append(loss)
            else:
                l1_loss = torch.abs(preds_hooked - preds_original)
                l1_loss = torch.sum(l1_loss, dim=[1])

            for loss in l1_loss:
                pred_differences.append(loss)

        i += 1

        #if i > 10:
        #    break

    total = torch.stack(pred_differences, dim=0)
    mean_ = torch.mean(total)
    std_ = torch.std(total)

    print("mean_ difference: {}".format(mean_))
    print("std_ difference: {}".format(std_))



