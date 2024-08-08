
import numpy as np
import os

import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 18})


layers = ["1_" + str(i) for i in range(3)] + ["2_" + str(i) for i in range(4)] + ["3_" + str(i) for i in range(6)] + ["4_" + str(i) for i in range(3)]

method = "DeepLift"

results_folder = "results_500"
#results_folder = "results_camelyon"

data_grad_cam = np.load(results_folder+"/auc_values_GradCAM.npy", allow_pickle=True).item()
#data_grad_cam = np.load(results_folder+"/resnet18_GradCAM.npy", allow_pickle=True).item()

print(data_grad_cam)

methods = ["DeepLift", "IG", "Grad"]
models = ["resnet34", "resnet50", "resnet50_robust"]

markers = ['--o', '--v', '--^', '--s', '--p', '--h']

def plot_metrics(score_type="ins"):
    for j, model in enumerate(models):

        for method in methods:

            min_value = 100000
            min_label = None
            min_layer = None

            max_value = -100000
            max_label = None
            max_layer = None

            all_data = []
            all_labels = []

            metrics_at_input_layer = []
            labels_at_input_layer = []
            for file in sorted(os.listdir(results_folder)):
                if method in file:
                    if model in file:
                        if "explain_original" in file:
                            continue

                        if model == "resnet50":
                            if "resnet50_robust" in file:
                                continue

                        data = np.load(results_folder+"/" + file, allow_pickle=True).item()
                        all_data.append(data[score_type])
                        all_labels.append(file[11:])

                    if "Input" in file:
                        #for method in methods:
                        #    if method in file:
                        data_at_input_layer = np.load(results_folder+"/auc_values_" + method + "_Input.npy", allow_pickle=True).item()

                        metric_at_input_layer = data_at_input_layer[score_type][j]
                        metrics_at_input_layer.append(metric_at_input_layer)
                        labels_at_input_layer.append("Input_" + method)
                        #print("metric at input layer: {}".format(metric_at_input_layer))


            #print("metrics_at_input_layer: {}".format(metrics_at_input_layer))
            metric_grad_cam = data_grad_cam[score_type][j]


            #plt.figure()



            for i, data in enumerate(all_data):
                label = all_labels[i]
                #print(data)
                idx_min_value = np.argmin(data)
                idx_max_value = np.argmax(data)

                min_data_value = min(data)
                max_data_value = max(data)
                #print(max_data_value)

                if min_value > min_data_value:
                    min_value = min_data_value
                    min_label = label
                    min_layer = layers[idx_min_value]

                if max_value < max_data_value:
                    max_value = max_data_value
                    max_label = label
                    max_layer = layers[idx_max_value]



            idx_min_value = np.argmin(metrics_at_input_layer)
            idx_max_value = np.argmax(metrics_at_input_layer)

            min_data_value = min(metrics_at_input_layer)
            max_data_value = max(metrics_at_input_layer)
            #print(max_data_value)

            if min_value > min_data_value:
                min_value = min_data_value
                min_label = labels_at_input_layer[idx_min_value]
                min_layer = "Input"

            if max_value < max_data_value:
                max_value = max_data_value
                max_label = labels_at_input_layer[idx_max_value]
                max_layer = "Input"


            #if "resnet50_robust" in model:
            #    print(min_data_value)
            #    print(min_value)
            #    print(metrics_at_input_layer)




                #plt.plot(layers, data, markers[i])

            #print(data_at_input_layer)

            print("min value: {}; label: {}; layer: {}".format(min_value, min_label, min_layer))
            print("max value: {}; label: {}; layer: {}".format(max_value, max_label, max_layer))


        #plt.plot(layers, [metric_grad_cam for _ in range(len(layers))], "--")

        #if model == "resnet50_DeepLift":
        #    if score_type == "ins":
        #        plt.plot(layers, [0.7267 for _ in range(len(layers))], "--")
        #    else:
        #        plt.plot(layers, [0.1076 for _ in range(len(layers))], "--")
        #    plt.legend(all_labels + ["Input", "GradCAM", "RISE"], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        #else:
        all_labels = ["Original", "Backward Hook", "Forward Hook", "Surrogate"]
        #plt.legend(all_labels + ["Input", "GradCAM"], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        #plt.savefig("figures/" + model + "_" + score_type + ".png", bbox_inches='tight')


print("-------- insertion -------------")
plot_metrics("ins")

print("--------- deletion -------------")
plot_metrics("del")






