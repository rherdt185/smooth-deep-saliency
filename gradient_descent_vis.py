import torchvision
import torch
import torch.nn.functional as F
#from lucent.optvis import param, transform
from tqdm import tqdm
import random




class LossObject:
    def __init__(self, loss_function, at_channels=None, at_pixels=None):
        self.loss_function = loss_function
        self.at_channels = at_channels
        self.at_pixels = at_pixels

    def get_loss(self, net_output):
        #print("mean net output: {}".format(torch.mean(net_output)))
        return self.loss_function(net_output)

    def __call__(self, net_output):
        #print("self.at_channels: {}".format(self.at_channels))
        loss = self.get_loss(net_output)
        if self.at_channels is not None:
            loss_out = 0
            if isinstance(self.at_channels, int):
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                #return torch.mean(loss_out)
            elif len(self.at_channels.shape) == 0:
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                #return torch.mean(loss_out)
            #print("at channels: {}".format(self.at_channels))
            else:
                for channel in self.at_channels:
                    loss_out = loss_out + loss[:, channel, :, :]
            if self.at_pixels is not None:
                loss_out_ = torch.clone(loss_out)
                loss_out = 0
                for pixel in self.at_pixels:
                    if len(pixel) == 3:
                        y = pixel[1]
                        x = pixel[2]
                    elif len(pixel) == 2:
                        y = pixel[0]
                        x = pixel[1]
                    else:
                        raise RuntimeError("invalid pixel data")
                    loss_out = loss_out + loss_out_[:, y, x]
            return torch.mean(loss_out)

            #return torch.mean(loss_out)
        elif self.at_pixels is not None:
            loss_out = 0
            for pixel in self.at_pixels:
                if len(pixel) == 3:
                    y = pixel[1]
                    x = pixel[2]
                elif len(pixel) == 2:
                    y = pixel[0]
                    x = pixel[1]
                else:
                    raise RuntimeError("invalid pixel data")
                loss_out = loss_out + loss[:, :, y, x]
            return torch.mean(loss_out)

        return torch.mean(loss)



class LossObject_Target:
    def __init__(self, loss_function, target, at_channels=None, at_pixels=None):
        self.loss_function = loss_function
        self.target = target
        self.target_internally = target
        self.at_channels = at_channels
        self.at_pixels = at_pixels

    def get_loss(self, net_output):
        #print(torch.mean(self.target_internally))
        return self.loss_function(net_output, self.target_internally.to("cuda"))

    def __call__(self, net_output):
        loss = self.get_loss(net_output)#self.loss_function(net_output, self.target.to("cuda"))
        if self.at_channels is not None:
            loss_out = 0
            if isinstance(self.at_channels, int):
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                return torch.mean(loss_out)
            elif len(self.at_channels.shape) == 0:
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                return torch.mean(loss_out)
            for channel in self.at_channels:
                loss_out = loss_out + torch.mean(loss[:, channel, :, :])
            return torch.mean(loss_out)
        elif self.at_pixels is not None:
            loss_out = 0
            for pixel in self.at_pixels:
                if len(pixel) == 3:
                    y = pixel[1]
                    x = pixel[2]
                elif len(pixel) == 2:
                    y = pixel[0]
                    x = pixel[1]
                else:
                    raise RuntimeError("invalid pixel data")
                loss_out = loss_out + loss[:, :, y, x]
            return torch.mean(loss_out)

        return torch.mean(loss)



class OptimGoal_ForwardHook:
    def __init__(self, hook_point, loss_object, weight=-1.0):
        #self.handle = hook_point.register_forward_hook(self.hook_function)
        self.hook_point = hook_point
        self.loss_object = loss_object
        self.hook_output = None
        self.weight = weight

    def set_handle(self):
        self.handle = self.hook_point.register_forward_hook(self.hook_function)

    def remove_handle(self):
        self.hook_output = None
        self.handle.remove()

    def hook_function(self, module, input_, output):
        self.hook_output = output

    def compute_loss(self):
        output = self.hook_output
        #print("output.shape: {}".format(output.shape))
        loss = self.loss_object(output)
        #print("loss: {}".format(loss))
        #self.hook_output.detach()

        return loss*self.weight






def reconstruct_input_gradient_descent(x, traced_network, num_steps=512, regularization="none",
                                       disable_tqdm=False):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        original_activations = traced_network(x)['layer_output']

    fft = True
    color_decorrelate = True
    if regularization == 'none':
        fft = False
        color_decorrelate = False

    param_f = lambda: param.image(x.shape[-1], fft=fft, decorrelate=color_decorrelate, batch=x.shape[0])
    params, image_f = param_f()

    #print("len params: {}".format(len(params)))
    #params[0].requires_grad = True
    optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    upsample = torch.nn.Upsample(size=x.shape[-1], mode="bilinear", align_corners=True)

    for i in tqdm(range(num_steps), ascii=True, disable=disable_tqdm):
        new_val = image_f()
        if regularization == 'jitter_only':
            new_val = normalize(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))

            new_val = upsample(new_val)
        elif regularization == 'none':
            new_val = normalize(new_val)

        reconstructed_activations = traced_network(new_val)["layer_output"]
        loss = F.l1_loss(reconstructed_activations, original_activations)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return image_f().cpu().detach()



hook_output = None
def layer_hook(module, input_, output):
    global hook_output
    #print(output.shape)
    hook_output = output



def reconstruct_input_gradient_descent_from_hook(x, network, layer_hook_point, num_steps=512, regularization="none",
                                       disable_tqdm=False):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    global hook_output

    hook = layer_hook_point.register_forward_hook(layer_hook)
    with torch.no_grad():
        network(x)
        original_activations = torch.clone(hook_output)

    fft = True
    color_decorrelate = True
    if regularization == 'none':
        fft = False
        color_decorrelate = False

    param_f = lambda: param.image(x.shape[-1], fft=fft, decorrelate=color_decorrelate, batch=x.shape[0])
    params, image_f = param_f()

    #print("len params: {}".format(len(params)))
    #params[0].requires_grad = True
    optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    upsample = torch.nn.Upsample(size=x.shape[-1], mode="bilinear", align_corners=True)

    for i in tqdm(range(num_steps), ascii=True, disable=disable_tqdm):
        new_val = image_f()
        if regularization == 'jitter_only':
            new_val = normalize(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))

            new_val = upsample(new_val)
        elif regularization == 'none':
            new_val = normalize(new_val)

        network(new_val)
        reconstructed_activations = hook_output
        loss = F.l1_loss(reconstructed_activations, original_activations)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    img_out = image_f().cpu().detach()
    hook.remove()

    return img_out




hook_input_gan = None
def hook_gan(module, input_, output):
    print("in hook gan function")
    return hook_input_gan


def hook_gan_single_vector(module, input_, output):
    out = torch.ones_like(output)*hook_input_gan.unsqueeze(dim=-1).unsqueeze(dim=-1)
    return out


def optim_gan(initial_gan_layer_activation, forward_function_gan_no_inputs, forward_hook_gan, forward_function_seg_net, optim_goals, from_single_vector=False, num_steps=512,
              regularization='full', normalization_preprocessor=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])):
    global hook_input_gan

    #return forward_function_gan_no_inputs().cpu().detach(), forward_function_gan_no_inputs().cpu().detach()

    for optim_goal in optim_goals:
        optim_goal.set_handle()

    #start_activations =

    #input_seg_net = input_seg_net.to("cuda")
    initial_gan_layer_activation = initial_gan_layer_activation.to("cuda")
    shape = initial_gan_layer_activation.shape


    if from_single_vector:
        hook_input_gan = initial_gan_layer_activation#[0, :, 5, 5]#torch.randn((input_.shape[1])).to("cuda")
        gan_handle = forward_hook_gan.register_forward_hook(hook_gan_single_vector)
    else:
        hook_input_gan = initial_gan_layer_activation#+torch.randn_like(input_)#*0.05
        gan_handle = forward_hook_gan.register_forward_hook(hook_gan)

    #input_ = torch.randn_like(input_)*3.0
    #image_f = lambda : F.relu(input_)
    hook_input_gan.requires_grad=True
    optimizer = torch.optim.Adam([hook_input_gan], lr=5e-2)
    #optimizer = torch.optim.SGD([hook_input_gan], lr=0.1, momentum=0.9)

    #image_f = WorkaroundPointerToInput(input_)

    upsample = torch.nn.Identity()#torch.nn.Upsample(size=input_seg_net.shape[-1], mode="bilinear", align_corners=True)



    with torch.autocast("cuda"):
        use_regularization = True
        for i in tqdm(range(1, max((num_steps,)) + 1), disable=False, ascii=True):
            optimizer.zero_grad()
            #print(torch.mean(hook_input_gan))

            jitter = 1
            SCALE = 1.1

            #new_val = torchvision.transforms.functional.normalize(image_f(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            new_val = forward_function_gan_no_inputs()
            new_val = normalization_preprocessor(new_val)
            #print("new val shape: {}".format(new_val.shape))

            if regularization == 'full':
                #new_val = image_f()
                #new_val = normalize(new_val)
                new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
                #new_val = F.pad(x, (2, 2, 2, 2), mode='reflect')#value=0.5, mode='reflect')

                new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))
                #rotation_values = list(range(-10, 11)) + 5 * [0]

                rotation_values = list(range(-5, 5+1))
                rotation_idx = random.randint(0, len(rotation_values)-1)
                rotate_by = rotation_values[rotation_idx]
                #min_value = -5
                #max_value = 5
                #rotate_by = random.randint(min_value, max_value)# + current_rot
                #scale_values = [1 + (i - 5) / 50. for i in range(11)]
                #scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
                scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
                scale_value_idx = random.randint(0, len(scale_values)-1)
                new_size = int(new_val.shape[-1]*scale_values[scale_value_idx])
                new_val = torchvision.transforms.functional.resize(new_val, new_size)
                new_val = torchvision.transforms.functional.rotate(new_val, angle=rotate_by, interpolation=2)
                #new_val = (new_val - model.normalization_mean) / model.normalization_std
                #new_val = transform_inception(new_val)
                #new_val = transform_classification(new_val)

                new_val = upsample(new_val)
                #new_val = torch.clamp(new_val, min=0.0, max=1.0)
            elif regularization == 'jitter_only':
                #new_val = image_f()
                #new_val = normalize(new_val)
                new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
                new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))

                new_val = upsample(new_val)
            elif regularization == 'none':
                pass
                #new_val = image_f()
                #new_val = normalize(new_val)
                #new_val = input_
            else:
                raise RuntimeError("regularization input is not known")

            forward_function_seg_net(new_val)

            loss = torch.tensor(0.0)

            for optim_goal in optim_goals:
                loss = loss + optim_goal.compute_loss()#*0.0
            #print(loss)

            loss.backward()
            optimizer.step()

        for optim_goal in optim_goals:
            optim_goal.remove_handle()

        gan_out = forward_function_gan_no_inputs().cpu().detach()
        gan_handle.remove()
        return gan_out.float()
        #return hook_input_gan.cpu().detach()





def general_optim(input_shape, forward_functions, optim_goals, num_steps=512, start_from_input=False, regularization='full',
        use_fft=True, use_decorrelate=True, normalization_preprocessor=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])):
    #raise RuntimeError

    from lucent.optvis import param
    #transform_inception = transform.preprocess_inceptionv1()
    #transform_classification = transform.normalize()

    for optim_goal in optim_goals:
        optim_goal.set_handle()


    param_f = lambda: param.image(input_shape[-1], fft=use_fft, decorrelate=use_decorrelate, batch=input_shape[0])
    params, image_f = param_f()

    optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)


    upsample = torch.nn.Upsample(size=input_shape[-1], mode="bilinear", align_corners=True)



    use_regularization = True
    for i in tqdm(range(1, max((num_steps,)) + 1), disable=False, ascii=True):
        optimizer.zero_grad()

        jitter = 1
        SCALE = 1.1

        if start_from_input:
            new_val = image_f()
        else:
            new_val = image_f()
        #new_val = torchvision.transforms.functional.normalize(image_f(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if regularization == 'full':
            #new_val = image_f()
            new_val = normalization_preprocessor(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            #new_val = F.pad(x, (2, 2, 2, 2), mode='reflect')#value=0.5, mode='reflect')

            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))
            #rotation_values = list(range(-10, 11)) + 5 * [0]

            rotation_values = list(range(-5, 5+1))
            rotation_idx = random.randint(0, len(rotation_values)-1)
            rotate_by = rotation_values[rotation_idx]
            #min_value = -5
            #max_value = 5
            #rotate_by = random.randint(min_value, max_value)# + current_rot
            #scale_values = [1 + (i - 5) / 50. for i in range(11)]
            #scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
            scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
            scale_value_idx = random.randint(0, len(scale_values)-1)
            new_size = int(new_val.shape[-1]*scale_values[scale_value_idx])
            new_val = torchvision.transforms.functional.resize(new_val, new_size)
            new_val = torchvision.transforms.functional.rotate(new_val, angle=rotate_by, interpolation=2)
            #new_val = (new_val - model.normalization_mean) / model.normalization_std
            #new_val = transform_inception(new_val)
            #new_val = transform_classification(new_val)

            new_val = upsample(new_val)
        elif regularization == 'jitter_only':
            #new_val = image_f()
            #new_val = normalize(new_val)
            new_val = normalization_preprocessor(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))

            new_val = upsample(new_val)
        elif regularization == 'none':
            new_val = normalization_preprocessor(new_val)
            #pass
            #new_val = image_f()
            #new_val = normalize(new_val)
            #new_val = input_
        else:
            raise RuntimeError("regularization input is not known")

        for forward_function in forward_functions:
            forward_function(new_val)

        loss = 0.0
        for optim_goal in optim_goals:
            loss = loss + optim_goal.compute_loss()
        #print(loss)

        loss.backward()
        optimizer.step()

    for optim_goal in optim_goals:
        optim_goal.remove_handle()
    if start_from_input:
        return image_f().cpu().detach()
    else:
        return image_f().cpu().detach()#image_f().cpu().detach()#x.cpu().detach()
    #return input_.cpu().detach()


