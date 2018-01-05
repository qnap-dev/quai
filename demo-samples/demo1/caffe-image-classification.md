# Image Classification using Caffe and QuAI

## Login the NAS
```
ssh admin@nas_ip
```

## (Option) Confirm the GPU card is mounted
```
GPU=nvidia0 gpu-docker run --rm nvidia/cuda nvidia-smi

Thu Dec 21 10:07:59 2017
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 381.22                 Driver Version: 381.22                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 0000:01:00.0     Off |                  N/A |
| 20%   29C    P8     8W / 250W |      2MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Start the gpu version of caffe docker and run the first example
```
GPU=nvidia0 gpu-docker run -it bvlc/caffe:gpu bash
```

Move to caffe folder
```
cd /opt/caffe
```

This is the location of GoogLeNet model, prototxt is the architecture of GoogLeNet model. The .caffemodel file need download manually.
```
ls -l models/bvlc_googlenet/
```

The script to download the GoogLeNet weights file (.caffemodel).
```
./scripts/download_model_binary.py models/bvlc_googlenet
```

The script to download the class label.
```
./data/ilsvrc12/get_ilsvrc_aux.sh
```

Run the pre-compiled cpp file to do the image classification. The image is https://github.com/BVLC/caffe/blob/master/examples/images/cat.jpg
```
./build/examples/cpp_classification/classification.bin models/bvlc_googlenet/deploy.prototxt models/bvlc_googlenet/bvlc_googlenet.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg
```

## Write the first python program to predict
```
# install vim
cd
touch /var/cache/apt/archives/lock
rmdir /var/cache/apt/archives/partial
mkdir -p /var/cache/apt/archives/partial
apt update
apt install vim -y
```

## Add a classify.py python program
```
vi classify.py
```

## Code:
```
# This example follow the sample code from http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import os
import caffe
import numpy as np
import datetime
import time
import sys

model = '/opt/caffe/models/bvlc_googlenet/deploy.prototxt'
weights = '/opt/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
if len(sys.argv)<2:
  print 'example: python classify.py image.jpg'
  sys.exit()

# Set Caffe to GPU mode and load the net from disk
caffe.set_mode_gpu()
caffe.set_device(0)
caffeNet = caffe.Net(model, weights, caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/opt/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': caffeNet.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

caffeNet.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227

img = caffe.io.load_image(sys.argv[1])
transformed_image = transformer.preprocess('data', img)

# copy the image data into the memory allocated for the net
caffeNet.blobs['data'].data[...] = transformed_image

# perform classification
tic = time.time()
res = caffeNet.forward()
toc = time.time()

elapsed = toc - tic

# sort top five predictions from softmax output
output_prob = caffeNet.blobs['prob'].data[0]
top_inds = output_prob.argsort()[::-1][:5]

# load ImageNet labels
labels_file = '/opt/caffe/data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')
result = np.vstack((labels[top_inds], output_prob[top_inds])).T

print('Elapsed: ' + str(elapsed*1000) + 'ms')
print(result.tolist())
```

## Run it:
```
python classify.py /opt/caffe/examples/images/cat.jpg
```

## Use CPU instead, mark below 2 lines, and run again
```
#caffe.set_mode_gpu()
#caffe.set_device(0)
```



