import caffe
import os
import numpy as np
import google.protobuf as pb
import google.protobuf.text_format


# project root
ROOT = '/home/zym/tensorrt/mobilenet'

# choose your source model and destination model
WEIGHT = os.path.join(ROOT, 'mobilenet.caffemodel')
MODEL = os.path.join(ROOT, 'mobilenet.prototxt')
DEPLOY_MODEL = os.path.join(ROOT, 'mobilenet_deploy.prototxt')

# set network using caffe api
caffe.set_mode_gpu()
net = caffe.Net(MODEL, WEIGHT, caffe.TRAIN)
dst_net = caffe.Net(DEPLOY_MODEL, caffe.TEST)
with open(MODEL) as f:
    model = caffe.proto.caffe_pb2.NetParameter()
    pb.text_format.Parse(f.read(), model)

# go through source model 
for i, layer in enumerate(model.layer):
    if layer.type == 'Convolution':
        # extract weight and bias in Convolution layer
        name = layer.name
        if 'fc' in name:
            dst_net.params[name][0].data[...] = net.params[name][0].data
            dst_net.params[name][1].data[...] = net.params[name][1].data
            break
        w = net.params[name][0].data
        batch_size = w.shape[0]
        try:
            b = net.params[name][1].data
        except:
            b = np.zeros(batch_size)

        # extract mean and var in BN layer
        bn = name+'/bn'
        mean = net.params[bn][0].data
        var = net.params[bn][1].data
        scalef = net.params[bn][2].data
        if scalef != 0:
            scalef = 1. / scalef
        mean = mean * scalef
        var = var * scalef

        # extract gamma and beta in Scale layer
        scale = name+'/scale'
        gamma = net.params[scale][0].data
        beta = net.params[scale][1].data

        # merge bn
        tmp = gamma/np.sqrt(var+1e-5)
        w = np.reshape(tmp, (batch_size, 1, 1, 1))*w
        b = tmp*(b-mean)+beta

        # store weight and bias in destination net
        dst_net.params[name][0].data[...] = w
        dst_net.params[name][1].data[...] = b

dst_net.save('mobilenet_deploy.caffemodel')

# test merged network
img = caffe.io.load_image('/home/zym/imagenet_test.JPEG')

transformer = caffe.io.Transformer({'data': dst_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255)
transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
transformer.set_input_scale('data', 0.017)

# get merged network output
img = transformer.preprocess('data', img)
dst_net.blobs['data'].data[...] = img
out = dst_net.forward()['prob']
print(np.argmax(out.flatten()))
