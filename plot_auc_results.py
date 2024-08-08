
import numpy as np
import os

import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 18})


layers = ["1_" + str(i) for i in range(3)] + ["2_" + str(i) for i in range(4)] + ["3_" + str(i) for i in range(6)] + ["4_" + str(i) for i in range(3)]

method = "IG"

results_folder = "results_500"
#results_folder = "results_camelyon"

data_at_input_layer = np.load(results_folder+"/auc_values_" + method + "_Input.npy", allow_pickle=True).item()
data_grad_cam = np.load(results_folder+"/auc_values_GradCAM.npy", allow_pickle=True).item()

models = ["resnet34_" + method, "resnet50_" + method, "resnet50_robust_" + method]
#models = ["resnet18_" + method]

markers = ['--o', '--v', '--^', '--s', '--p', '--h']

def plot_metrics(score_type="ins"):
    for j, model in enumerate(models):
        all_data = []
        all_labels = []
        for file in sorted(os.listdir(results_folder)):
            if model in file:
                #if "surrogate" in file:
                #    if not "explain_original" in file:
                #        continue

                if "explain_original" in file:
                    continue

                if model == "resnet50":
                    if "resnet50_robust" in file:
                        continue

                data = np.load(results_folder+"/" + file, allow_pickle=True).item()
                all_data.append(data[score_type])
                all_labels.append(file[11:])


        plt.figure()
        for i, data in enumerate(all_data):
            label = all_labels[i]

            plt.plot(layers, data, markers[i])

        #print(data_at_input_layer)

        metric_at_input_layer = data_at_input_layer[score_type][j]
        plt.plot(layers, [metric_at_input_layer for _ in range(len(layers))])

        metric_grad_cam = data_grad_cam[score_type][j]
        plt.plot(layers, [metric_grad_cam for _ in range(len(layers))], "--")

        #if model == "resnet50_DeepLift":
        #    if score_type == "ins":
        #        plt.plot(layers, [0.7267 for _ in range(len(layers))], "--")
        #    else:
        #        plt.plot(layers, [0.1076 for _ in range(len(layers))], "--")
        #    plt.legend(all_labels + ["Input", "GradCAM", "RISE"], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        #else:
        all_labels = ["Original", "Backward Hook", "Forward Hook", "Surrogate"]
        plt.legend(all_labels + ["Input", "GradCAM"], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        plt.savefig("figures/" + model + "_" + score_type + ".png", bbox_inches='tight')



def plot_metrics_randomized_layers(score_type="ins"):
    print("plot metrics randomized layers")

    #data = np.load("results_randomize_model/resnet50_robust__layer1_DeepLift.npy", allow_pickle=True)

    #print(data)

    #raise RuntimeError
    results_folder = "results_randomize_model"

    models = ["resnet34", "resnet50", "resnet50_robust"]

    layers = ["layer1", "layer2", "layer3", "layer4"]

    layers_randomized = ["original", "fc", "4_2", "3_5", "2_3", "1_2", "all"]

    for j, model in enumerate(models):
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ model: {}".format(model))

        layer_names_legend = ["Layer 1_2", "Layer 2_3", "Layer 3_5", "Layer 4_2"]
        for file in sorted(os.listdir(results_folder)):
            if "DeepLift" in file:
                #print(file)
                if model in file:
                    #print("model in file")
                    if model == "resnet50":
                        if "resnet50_robust" in file:
                            continue
                    #all_data = []
                    #all_labels = []
                    for layer_idx, layer in enumerate(layers):
                        if layer in file:
                            data_randomized_saliency_map = np.load("results_randomize_saliency_map/" + model + "__" + layer + "_" + "DeepLift.npy", allow_pickle=True).item()[score_type]
                            #print("layer in file")
                            data = np.load(results_folder+"/" + file, allow_pickle=True).item()
                            #all_data.append(data[score_type])
                            #all_labels.append(file[11:])


                            plt.figure()
                            #for i, data in enumerate(all_data):
                            #    label = all_labels[i]

                            plt.plot(layers_randomized, data[score_type], markers[0], markersize=16)
                            plt.plot(layers_randomized, [data_randomized_saliency_map[0] for _ in range(len(layers_randomized))], '--')

                            plt.legend([layer_names_legend[layer_idx]] + ["Random Saliency Map"], loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

                            plt.savefig("figures/randomized_" + model + "_" + layer + "_" + score_type + ".png", bbox_inches='tight')



plot_metrics("ins")
plot_metrics("del")

#plot_metrics_randomized_layers("ins")
#plot_metrics_randomized_layers("del")









