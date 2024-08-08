import numpy as np
import matplotlib.pyplot as plt
import matplotlib


font = {
        'size'   : 12}

matplotlib.rc('font', **font)

method = "IG"

mean_tv_original = list(np.load("saved_np/imagenet_mean_tv_" + method + "_original.npy"))
mean_tv_surrogate = list(np.load("saved_np/imagenet_mean_tv_" + method + "_surrogate.npy"))

print(mean_tv_original)
print(mean_tv_surrogate)

mean_tv_hooked = list(np.load("saved_np/imagenet_mean_tv_" + method + "_backward_hook.npy"))
mean_tv_forward_hooked = list(np.load("saved_np/imagenet_mean_tv_" + method + "_forward_hook.npy"))

#print(mean_tv_original)

# forgot layer3[5], append manually from table

fig = plt.figure(dpi=200)
layers = ["1_" + str(i) for i in range(3)] + ["2_" + str(i) for i in range(4)] + ["3_" + str(i) for i in range(6)]
x = [layers[i] for i in range(13)]
plt.plot(x, mean_tv_original[:13], '--bo', label="Original")
plt.plot(x, mean_tv_surrogate[:13], '--rs', label="Bilinear Surrogate")
plt.plot(x, mean_tv_hooked[:13], '--yv', label="Backward Hook")
plt.plot(x, mean_tv_forward_hooked[:13], '--g^', label="Forward Hook")
plt.xlabel("Layer")
plt.ylabel("Total Variation")
fig.legend(loc='outside upper right')
fig.suptitle("ResNet34")
plt.savefig("imgs/tv_plot_" + method + "_resnet34.png")
plt.close()

fig = plt.figure(dpi=200)
plt.plot(x, mean_tv_original[13:26], '--bo', label="Original")
plt.plot(x, mean_tv_surrogate[13:26], '--rs', label="Bilinear Surrogate")
plt.plot(x, mean_tv_hooked[13:26], '--yv', label="Backward Hook")
plt.plot(x, mean_tv_forward_hooked[13:26], '--g^', label="Forward Hook")
plt.xlabel("Layer")
plt.ylabel("Total Variation")
fig.legend(loc='outside upper right')
fig.suptitle("ResNet50")
plt.savefig("imgs/tv_plot_" + method + "_resnet50.png")
plt.close()

fig = plt.figure(dpi=200)
plt.plot(x, mean_tv_original[26:], '--bo', label="Original")
plt.plot(x, mean_tv_surrogate[26:], '--rs', label="Bilinear Surrogate")
plt.plot(x, mean_tv_hooked[26:], '--yv', label="Backward Hook")
plt.plot(x, mean_tv_forward_hooked[26:], '--g^', label="Forward Hook")
plt.xlabel("Layer")
plt.ylabel("Total Variation")
fig.legend(loc='outside upper right')
fig.suptitle("ResNet50 Robust")
plt.savefig("imgs/tv_plot_" + method + "_resnet50_robust.png")
plt.close()