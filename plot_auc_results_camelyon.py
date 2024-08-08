
import numpy as np
import os

import matplotlib.pyplot as plt

layers = ["1_" + str(i) for i in range(2)] + ["2_" + str(i) for i in range(2)] + ["3_" + str(i) for i in range(2)] + ["4_" + str(i) for i in range(2)]
#layers = ["1_" + str(i) for i in range(3)] + ["2_" + str(i) for i in range(4)] + ["3_" + str(i) for i in range(6)] + ["4_" + str(i) for i in range(3)]

method = "Grad"

#results_folder = "results_digipath"
results_folder = "results_camelyon"

for file in os.listdir(results_folder):
    if method in file and "Input" in file:
        data_at_input_layer = np.load(results_folder + "/" + file, allow_pickle=True).item()
    if "GradCAM" in file:
        data_grad_cam = np.load(results_folder + "/" + file, allow_pickle=True).item()

#models = ["resnet34_" + method, "resnet50_" + method, "resnet50_robust_" + method]
models = ["resnet18_" + method]

markers = ['--o', '--v', '--^', '--s', '--p', '--h']

def plot_metrics(score_type="ins"):
    j = 0
    model = "resnet18"
    #for j, model in enumerate(models):
    all_data = []
    all_labels = []
    for file in sorted(os.listdir(results_folder)):
        if model in file:
            if method in file:
                if "explain_original" in file:
                    continue
                if "Input" in file:
                    continue
                if "GradCAM" in file:
                    continue
                if "surrogate_2" in file:
                    continue


                #if model == "resnet50":
                #    if "resnet50_robust" in file:
                #        continue

                data = np.load(results_folder+"/" + file, allow_pickle=True).item()
                all_data.append(data[score_type])
                all_labels.append(file[9:])


    plt.figure()
    for i, data in enumerate(all_data):
        print(data)
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

    plt.savefig("figures/" + model + "_" + method + "_" + score_type + ".png", bbox_inches='tight')

plot_metrics("ins")
plot_metrics("del")











