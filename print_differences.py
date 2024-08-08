
import numpy as np
import os

import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 18})


layers = ["1_" + str(i) for i in range(3)] + ["2_" + str(i) for i in range(4)] + ["3_" + str(i) for i in range(6)] + ["4_" + str(i) for i in range(3)]

#method = "DeepLift"

results_folder = "results_500"
#results_folder = "results_camelyon"

data_grad_cam = np.load(results_folder+"/auc_values_GradCAM.npy", allow_pickle=True).item()
#data_grad_cam = np.load(results_folder+"/resnet18_GradCAM.npy", allow_pickle=True).item()

print(data_grad_cam)

methods = ["DeepLift"]#, "IG", "Grad"]
models = ["resnet34", "resnet50", "resnet50_robust"]

markers = ['--o', '--v', '--^', '--s', '--p', '--h']

def plot_metrics(score_type="ins"):
    differences = []
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


            # original, backward hook, forward hook, surrogate
            #print(all_labels)
            differences.append(np.mean(np.array(all_data[2])/np.array(all_data[0])))
            #print(np.array(all_data[-1])/np.array(all_data[0]))

            #plt.figure()



            #for i, data in enumerate(all_data):



            #print("min value: {}; label: {}; layer: {}".format(min_value, min_label, min_layer))
            #print("max value: {}; label: {}; layer: {}".format(max_value, max_label, max_layer))

    print(np.mean(np.array(differences)))



print("-------- insertion -------------")
plot_metrics("ins")

print("--------- deletion -------------")
plot_metrics("del")






