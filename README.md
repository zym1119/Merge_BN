# Merge_BN
Merge convolution and batchnorm layers in both Caffe and PyTorch using Python

Basic idea can be found in my CSDN Blog:

[Caffe merge bn with equations and explanations](https://blog.csdn.net/zym19941119/article/details/84635371)

[PyTorch merge bn](https://blog.csdn.net/zym19941119/article/details/84640433)

I wrote the Caffe merge script a long time ago while Pytorch merge script recently, thus may have many difference between them. 

Both of them have been tested yet.

# Usage

Before using either of them, you should have a **no-bn version model** first, it means you should have a caffe network prototxt without bn or a pytorch model with all bn layers commented.

After prepare your model and weights, you should modify the path and imports in python scripts

Finally, just run and merge~

BTW, i offered a no-bn version shufflenetv2 model for example
