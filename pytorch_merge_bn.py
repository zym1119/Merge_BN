import torch
import os
from collections import OrderedDict
import cv2
import numpy as np
import torchvision.transforms as transforms


"""  Parameters and variables  """
IMAGENET = '/home/zym/ImageNet/ILSVRC2012_img_val_256xN_list.txt'
LABEL = '/home/zym/ImageNet/synset.txt'
TEST_ITER = 10
SAVE = False
TEST_AFTER_MERGE = True


"""  Functions  """
def merge(params, name, layer):
    # global variables
    global weights, bias
    global bn_param

    if layer == 'Convolution':
        # save weights and bias when meet conv layer
        if 'weight' in name:
            weights = params.data
            bias = torch.zeros(weights.size()[0])
        elif 'bias' in name:
            bias = params.data
        bn_param = {}

    elif layer == 'BatchNorm':
        # save bn params
        bn_param[name.split('.')[-1]] = params.data

        # running_var is the last bn param in pytorch
        if 'running_var' in name:
            # let us merge bn ~
            tmp = bn_param['weight'] / torch.sqrt(bn_param['running_var'] + 1e-5)
            weights = tmp.view(tmp.size()[0], 1, 1, 1) * weights
            bias = tmp*(bias - bn_param['running_mean']) + bn_param['bias']

            return weights, bias

    return None, None


"""  Main functions  """
# import pytorch model
import models.shufflenetv2.shufflenetv2_merge as shufflenetv2
pytorch_net = shufflenetv2.ShuffleNetV2().eval()
model_path = shufflenetv2.weight_file

# load weights
print('Finding trained model weights...')
try:
    for file in os.listdir(model_path):
        if 'pth' in file:
            print('Loading weights from %s ...' % file)
            trained_weights = torch.load(os.path.join(model_path, file))
            # pytorch_net.load_state_dict(trained_weights)
            print('Weights load success')
            break
except:
    raise ValueError('No trained model found or loading error occurs')

# go through pytorch net
print('Going through pytorch net weights...')
new_weights = OrderedDict()
inner_product_flag = False
for name, params in trained_weights.items():
    if len(params.size()) == 4:
        _, _ = merge(params, name, 'Convolution')
        prev_layer = name
    elif len(params.size()) == 1 and not inner_product_flag:
        w, b = merge(params, name, 'BatchNorm')
        if w is not None:
            new_weights[prev_layer] = w
            new_weights[prev_layer.replace('weight', 'bias')] = b
    else:
        # inner product layer
        # if meet inner product layer,
        # the next bias weight can be misclassified as 'BatchNorm' layer as len(params.size()) == 1
        new_weights[name] = params
        inner_product_flag = True

# align names in new_weights with pytorch model
# after move BatchNorm layer in pytorch model,
# the layer names between old model and new model will mis-align
print('Aligning weight names...')
pytorch_net_key_list = list(pytorch_net.state_dict().keys())
new_weights_key_list = list(new_weights.keys())
assert len(pytorch_net_key_list) == len(new_weights_key_list)
for index in range(len(pytorch_net_key_list)):
    new_weights[pytorch_net_key_list[index]] = new_weights.pop(new_weights_key_list[index])

# save new weights
if SAVE:
    torch.save(new_weights, model_path + '/' + file.replace('.pth', '_merged.pth'))

# test merged pytorch model
if TEST_AFTER_MERGE:
    try:
        pytorch_net.load_state_dict(new_weights)
        print('Pytorch net load weights success~')
    except:
        raise ValueError('Load new weights error')

    print('-' * 50)
    with open(LABEL) as f:
        labels = f.read().splitlines()
    with open(IMAGENET) as f:
        images = f.read().splitlines()
        for _ in range(TEST_ITER):
            # cv2 default chann el is BGR
            image_path, label = images[np.random.randint(0, len(images))].split(' ')
            # image_path, label = images[0].split(' ')
            input_image = cv2.imread(image_path)
            input_image = cv2.resize(input_image, (224, 224))
            input_image = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])(input_image)
            input_image = input_image.view(1, 3, 224, 224)
            output_logits = pytorch_net(input_image)
            _, index = output_logits.max(dim=1)
            print('true label: \t%s' % labels[int(label)])
            print('predict label:\t%s' % labels[int(index)])
            print('-' * 50)
