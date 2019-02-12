import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class VggRand:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(VggRand)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "../vgg16.npy")
            vgg16_npy_path = path
            # print (vgg19_npy_path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        data = self.data_dict
        
        for k in data.keys():
            for i in range(2):
                tmp = np.shape(data[k][i])
                print(tmp)
                l = len(tmp)
                print(k,i)
                if i == 0:
                    if l > 2:
                        data[k][i] =np.float32(np.random.rand(tmp[0],tmp[1], tmp[2], tmp[3]))

                    else:
                        data[k][i] = np.float32(np.random.rand(tmp[0],tmp[1]))
                else:
                    data[k][i] = np.float32(np.random.rand(tmp[0]))

        print(np.shape(self.data_dict))
        
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        print(np.size(rgb_scaled), type(rgb_scaled))
        print(rgb_scaled)

        # Convert RGB to BGR
        red, green, blue = tf.split(rgb_scaled, [1,1,1 ] , 3 ) #tf.split(3, 3, rgb_scaled)

        print(red)
        print(red.get_shape())
        print(red.get_shape().as_list()[1:])
        print("************************************")
        assert red.get_shape().as_list()[1:] == [256, 256, 1]
        assert green.get_shape().as_list()[1:] == [256, 256, 1]
        assert blue.get_shape().as_list()[1:] == [256, 256, 1]
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 3)
        assert bgr.get_shape().as_list()[1:] == [256, 256, 3]
        self.bgr_in = bgr
        self.conv_list = []
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.conv_list.append(self.conv1_1)
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1')
    
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.conv_list.append(self.pool1)
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv_list.append(self.pool2)
        self.pool3 = self.avg_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        #self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.conv_list.append(self.pool3)
        self.pool4 = self.avg_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        #self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.conv_list.append(self.pool4)
        #self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
