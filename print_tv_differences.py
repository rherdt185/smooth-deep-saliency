import numpy as np
import os


folder_path = "saved_np/"

saliency_methods = ["Grad", "DeepLift", "IG"]

grad_fix_methods = ["original", "surrogate", "backward_hook", "forward_hook"]

models = ["resnet34", "resnet50", "resnet50_robust"]

prefix = "imagenet_mean_tv_"

for saliency_method in saliency_methods:
    original_tv = np.load(folder_path + prefix + saliency_method + "_original.npy", allow_pickle=True)
    original_tv = np.array(original_tv)
    for grad_fix_method in grad_fix_methods:
        tv_values = np.load(folder_path + prefix + saliency_method + "_" + grad_fix_method + ".npy", allow_pickle=True)
        tv_values = np.array(tv_values)

        if len(models) > 0:
            n_layers = 13

            for i, model in enumerate(models):

                print(model + " " + saliency_method + "_" + grad_fix_method + ": {}".format(1.0 - np.mean(tv_values[i*n_layers:(i+1)*n_layers] / original_tv[i*n_layers:(i+1)*n_layers])))

            # camelyon16
            original_tv_values_camelyon16 = np.load(folder_path + "mean_tv_camelyon16_" + saliency_method + "_original.npy", allow_pickle=True)
            tv_values_camelyon16 = np.load(folder_path + "mean_tv_camelyon16_" + saliency_method + "_" + grad_fix_method + ".npy", allow_pickle=True)
            print("camelyon16" + " " + saliency_method + "_" + grad_fix_method + ": {}".format(1.0 - np.mean(tv_values_camelyon16 / original_tv_values_camelyon16)))





        #print("total " + saliency_method + "_" + grad_fix_method + ": {}".format(1.0 - np.mean(tv_values / original_tv)))

        # include camelyon16
        print("total " + saliency_method + "_" + grad_fix_method + ": {}".format(1.0 - (3.0*np.mean(tv_values / original_tv) + np.mean(tv_values_camelyon16 / original_tv_values_camelyon16))/4.0  ))






