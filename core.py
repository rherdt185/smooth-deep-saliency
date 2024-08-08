#import pytorch_lightning as pl
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import datetime

from pathlib import Path
import os


hook_output = None

def layer_hook(module, input_, output):
    global hook_output
    hook_output = output


def access_activations_forward_hook(x, forward_function, forward_hook_point):
    handle = forward_hook_point.register_forward_hook(layer_hook)
    with torch.no_grad():
        forward_function(*x)
    handle.remove()

    #if isinstance(hook_output, list) or isinstance(hook_output, tuple):
    #    return hook_output
    return hook_output.detach().cpu()



def bilinear_downsample(x, wanted_output_shape):
    return F.interpolate(x, size=wanted_output_shape[2:], mode='bilinear')


def average_pool_downsample(x, wanted_output_shape):
    return F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)




class ForwardBilinearOverride:
    def __init__(self, hook_point, convolutions, downsample_method=bilinear_downsample):
        hook_point.register_forward_hook(self.layer_hook)
        self.convolutions = convolutions
        self.downsample_method = downsample_method


    def layer_hook(self, module, input_, output):
        #print("len input: {}".format(len(input_)))
        before_downsample = input_[0]
        after_downsample = output

        conv_before, conv_after = self.convolutions

        x = conv_before(before_downsample)
        x = self.downsample_method(x, wanted_output_shape=after_downsample.shape)
        #x = F.interpolate(x, size=after_downsample.shape[2:], mode='bilinear')
        x = conv_after(x)

        return x#torch.zeros_like(output)



class ForwardHookAccess:
    def __init__(self, hook_point):
        hook_point.register_forward_hook(self.layer_hook)

    def layer_hook(self, module, input_, output):
        self.input_ = input_[0]
        self.output = output



class EvalSurrogateModel(nn.Module):
    def __init__(self, classifier, hook_points=None, surrogate_convolutions=None, is_stitched_classifier=False,
                    gradient_plot_hook_layer=None, surrogate_downsampling=bilinear_downsample, **kwargs):
        super().__init__(**kwargs)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = SummaryWriter("logs/bilinear_surrogate_model_" + current_time)

        self.classifier = classifier
        self.classifier = self.classifier.eval()
        classifier_for_hooks = self.classifier
        if is_stitched_classifier:
            classifier_for_hooks = self.classifier.model_to

        if surrogate_convolutions is not None:
            for i, hook_point in enumerate(hook_points):
                if i == 0:
                    continue
                ForwardBilinearOverride(hook_point, surrogate_convolutions[str(i)], downsample_method=surrogate_downsampling)

        if gradient_plot_hook_layer is not None:
            gradient_plot_hook_layer.register_backward_hook(self.gradient_accessor)
            #self.classifier.layer1[2].register_backward_hook(self.gradient_accessor)

        #print(self.classifier.layer4[0].downsample[0])
        #raise RuntimeError



    def gradient_accessor(self, module, grad_input, grad_output):
        print("in backward hook")
        grad_detached = torch.clone(grad_output[0]).detach().cpu()#.numpy()

        for i, grads in enumerate(grad_detached[:5]):
            print(i)
            grads = grad_detached[i]

            fig = plt.figure()
            plt.imshow(torch.mean(grads, dim=[0]))
            self.summary_writer.add_figure("grad", fig, i)
            plt.close()

        self.summary_writer.close()


    def forward(self, x):
        return self.classifier(x)





class SurrogateModelHooks(nn.Module):
    def __init__(self, classifier, hook_points, sample_input, surrogate_downsampling=bilinear_downsample, **kwargs):
        super().__init__(**kwargs)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = SummaryWriter("logs/bilinear_surrogate_model_" + current_time)

        self.forward_hook_accessors = []

        for hook_point in hook_points:
            self.forward_hook_accessors.append(ForwardHookAccess(hook_point))

        self.surrogate_downsampling = surrogate_downsampling

        self.convolutions = nn.ModuleDict()
        with torch.no_grad():
            classifier(sample_input)

        for i, forward_hook_accessor in enumerate(self.forward_hook_accessors):
            num_channels_in = forward_hook_accessor.input_.shape[1]
            num_channels_out = forward_hook_accessor.output.shape[1]
            conv_one = nn.Conv2d(in_channels=num_channels_in, out_channels=num_channels_out,
                                 kernel_size=3, padding='same')

            conv_two = nn.Conv2d(in_channels=num_channels_out, out_channels=num_channels_out,
                                 kernel_size=3, padding='same')

            self.convolutions[str(i)] = nn.ModuleList([conv_one, conv_two])

        self.classifier = classifier
        sample_input = None


    def configure_optimizers(self):
        parameters = []
        for key in self.convolutions.keys():
            convs = self.convolutions[key]
            for conv in convs:

                parameters += list(conv.parameters())
        optimizer = torch.optim.Adam(parameters)
        return [optimizer]


    def training_step(self, x, idx):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            self.classifier(x)

        for i, forward_hook_accessor in enumerate(self.forward_hook_accessors):
            before_downsample = forward_hook_accessor.input_
            after_downsample = forward_hook_accessor.output

            conv_before, conv_after = self.convolutions[str(i)]

            x = conv_before(before_downsample)
            x = self.surrogate_downsampling(x, after_downsample.shape)
            #x = F.interpolate(x, size=after_downsample.shape[2:], mode='bilinear')
            x = conv_after(x)

            loss_layer = F.l1_loss(x, after_downsample)
            self.summary_writer.add_scalar("loss layer " + str(i), loss_layer, global_step=idx)
            total_loss = total_loss + loss_layer

        self.summary_writer.add_scalar("total loss", total_loss, global_step=idx)
        return total_loss


    def save_checkpoint(self, checkpoint_folder_path, epoch):
        Path(checkpoint_folder_path).mkdir(parents=True, exist_ok=True)

        torch.save(self.convolutions, checkpoint_folder_path + str(epoch).zfill(4) + ".pt")





class SurrogateModelNew(nn.Module):
    def __init__(self, classifier, sample_input, **kwargs):
        super().__init__(**kwargs)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = SummaryWriter("logs/bilinear_surrogate_model_" + current_time)

        print("names: {}".format(get_graph_node_names(classifier)))
        #raise RuntimeError

        if 'bn3' in get_graph_node_names(classifier):
            # uses bottleneck block which has bn3 layers, e.g. resnet50
            return_nodes = {
                # node_name: user-specified key for output dict
                # here the downsample happens in the second conv
                'x' : 'input_0',
                'relu' : 'input_1',
                'layer2.0.relu': 'input_2',
                'layer3.0.relu': 'input_3',
                'layer4.0.relu': 'input_4',

                'layer1.2': 'input_5',
                'layer2.3': 'input_6',
                'layer3.5': 'input_7',



                'conv1': 'output_0',
                'maxpool' : 'output_1',
                'layer2.0.conv2': 'output_2',
                'layer3.0.conv2': 'output_3',
                'layer4.0.conv2': 'output_4',

                'layer2.0.downsample.0': 'output_5',
                'layer3.0.downsample.0': 'output_6',
                'layer4.0.downsample.0': 'output_7',
            }
        else:
            # uses basic block which has only up to bn2 layers, e.g. resnet34
            # here the downsample happens in the first conv layer
            return_nodes = {
                # node_name: user-specified key for output dict
                'x' : 'input_0',
                'relu' : 'input_1',
                'layer1.2': 'input_2',
                'layer2.3': 'input_3',
                'layer3.5': 'input_4',

                'layer1': 'input_5',
                'layer2': 'input_6',
                'layer3': 'input_7',



                'conv1': 'output_0',
                'maxpool' : 'output_1',
                'layer2.0.conv1': 'output_2',
                'layer3.0.conv1': 'output_3',
                'layer4.0.conv1': 'output_4',

                'layer2.0.downsample.0': 'output_5',
                'layer3.0.downsample.0': 'output_6',
                'layer4.0.downsample.0': 'output_7',
            }
        self.feature_extractor = create_feature_extractor(classifier, return_nodes)


        self.convolutions = nn.ModuleDict()
        sample_output = self.feature_extractor(sample_input)

        for i in range(8):
            num_channels_one = sample_output['input_' + str(i)].shape[1]
            num_channels_two = sample_output['output_' + str(i)].shape[1]

            conv_one = nn.Conv2d(in_channels=num_channels_one, out_channels=num_channels_two,
                                 kernel_size=3, padding='same')

            conv_two = nn.Conv2d(in_channels=num_channels_two, out_channels=num_channels_two,
                                 kernel_size=3, padding='same')

            self.convolutions[str(i)] = nn.ModuleList([conv_one, conv_two])



    def configure_optimizers(self):
        parameters = []
        for key in self.convolutions.keys():
            convs = self.convolutions[key]
            for conv in convs:

                parameters += list(conv.parameters())
        optimizer = torch.optim.Adam(parameters)
        return [optimizer]


    def training_step(self, x, idx):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            layer_outputs = self.feature_extractor(x)

        for i in range(8):
            before_downsample = layer_outputs['input_' + str(i)]
            after_downsample = layer_outputs['output_' + str(i)]

            conv_before, conv_after = self.convolutions[str(i)]

            x = conv_before(before_downsample)
            x = F.interpolate(x, size=after_downsample.shape[2:], mode='bilinear')
            x = conv_after(x)

            loss_layer = F.l1_loss(x, after_downsample)
            self.summary_writer.add_scalar("loss layer " + str(i), loss_layer, global_step=idx)
            total_loss = total_loss + loss_layer

        self.summary_writer.add_scalar("total loss", total_loss, global_step=idx)
        return total_loss


    def save_checkpoint(self, checkpoint_folder_path, epoch):
        Path(checkpoint_folder_path).mkdir(parents=True, exist_ok=True)

        torch.save(self.convolutions, checkpoint_folder_path + str(epoch).zfill(4) + ".pt")






class SurrogateModel(nn.Module):
    def __init__(self, classifier, sample_input, **kwargs):
        super().__init__(**kwargs)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = SummaryWriter("logs/bilinear_surrogate_model_" + current_time)

        #print("names: {}".format(get_graph_node_names(classifier)))
        #raise RuntimeError

        return_nodes = {
            # node_name: user-specified key for output dict
            'x' : 'layer0',
            'relu' : 'layer1',
            'layer2.0.relu': 'layer2',
            'layer3.0.relu': 'layer3',
            'layer4.0.relu': 'layer4',


            'conv1': 'layer0_down',
            'maxpool' : 'layer1_down',
            'layer2.0.conv2': 'layer2_down',
            'layer3.0.conv2': 'layer3_down',
            'layer4.0.conv2': 'layer4_down',
        }


        self.feature_extractor = create_feature_extractor(classifier, return_nodes)


        self.convolutions = nn.ModuleDict()
        sample_output = self.feature_extractor(sample_input)

        for i in range(5):
            num_channels_one = sample_output['layer' + str(i)].shape[1]
            num_channels_two = sample_output['layer' + str(i) + '_down'].shape[1]

            conv_one = nn.Conv2d(in_channels=num_channels_one, out_channels=num_channels_two,
                                 kernel_size=3, padding='same')

            conv_two = nn.Conv2d(in_channels=num_channels_two, out_channels=num_channels_two,
                                 kernel_size=3, padding='same')

            self.convolutions['layer' + str(i)] = nn.ModuleList([conv_one, conv_two])



    def configure_optimizers(self):
        parameters = []
        for key in self.convolutions.keys():
            convs = self.convolutions[key]
            for conv in convs:

                parameters += list(conv.parameters())
        optimizer = torch.optim.Adam(parameters)
        return [optimizer]


    def training_step(self, x, idx):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            layer_outputs = self.feature_extractor(x)

        for i in range(5):
            before_downsample = layer_outputs['layer' + str(i)]
            after_downsample = layer_outputs['layer' + str(i) + '_down']

            conv_before, conv_after = self.convolutions['layer' + str(i)]

            x = conv_before(before_downsample)
            x = F.interpolate(x, size=after_downsample.shape[2:], mode='bilinear')
            x = conv_after(x)

            loss_layer = F.l1_loss(x, after_downsample)
            self.summary_writer.add_scalar("loss layer " + str(i), loss_layer, global_step=idx)
            total_loss = total_loss + loss_layer

        self.summary_writer.add_scalar("total loss", total_loss, global_step=idx)
        return total_loss


    def save_checkpoint(self, checkpoint_folder_path, epoch):
        Path(checkpoint_folder_path).mkdir(parents=True, exist_ok=True)

        torch.save(self.convolutions, checkpoint_folder_path + str(epoch).zfill(4) + ".pt")




