
code the reproduce the results for the experiments on the ImageNet1K and Camelyon16 datasets (the Digipath data and models are not here)

example.ipynb illustrates on an example image (sample.jpg) how to use the bilinear surrogate and backward hook, for the ResNet34
everything to run that example should be included
the checkpoint of the bilinear surrogate for the ResNet34 model is included



for training the models and reproducing all the results in the paper, follow those steps:

set the path to the train and validation data of the ImageNet1K dataset inside settings.py

for the robust resnet50 model checkpoint, download the ImageNet model from here https://github.com/MadryLab/robustness_applications? into the models folder

for the resnet18 camelyon16 checkpoint, download it (the 'Pathology tumor detection' model) from the monai model zoo https://monai.io/model-zoo.html unzip it and copy the contents of the 'pathology_tumor_detection' folder into the 'pathology_tumor_detection' folder here

for the insertion and deletion metrics, we used the implementation of the authors of the RISE paper, which can be found here: https://github.com/eclique/RISE


-first run train.py to train the bilinear surrogate models for the ImageNet1K data
-then eval_.py plots the saliency maps for DeepLift, computes the total variation and insertion and deletion metrics over the layers, and the prediction and accuracy difference


-train_camelyon.py trains the bilinear surrogate model for the Camelyon16 data (takes some time till it starts to run)
-eval_camelyon.py plots examples for the DeepLift saliency maps and runs the total variation and prediction difference and the insertion and deletion metrics

