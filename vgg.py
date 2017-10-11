from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io


def load_model_data(model_filename):
  model_data = scipy.io.loadmat(model_filename)
  mean = model_data['normalization'][0][0][0]
  mean_pixel = np.mean(mean, axis=(0, 1), dtype=np.float32)
  model_layers = model_data['layers'][0]
  return model_layers, mean_pixel


def load_weight_bias(model_layers):
  weights = []
  biases = []
  conv_layers = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
  for layer_num in conv_layers:
    weight, bias = model_layers[layer_num][0][0][0][0]
    # mat weights: [width, height, in_channels, out_channels]
    # tensorflow weights: [height, width, in_channels, out_channels]
    weight = np.transpose(weight, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    weights.append(weight)
    biases.append(bias)
  return weights, biases


def vgg_19(image, model_layers):
  weights, biases = load_weight_bias(model_layers)
  content_features = []
  style_features = []

  # conv1_1
  net = tf.nn.conv2d(input=image, filter=weights[0], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[0])
  # relu1_1
  net = tf.nn.relu(net)
  style_features.append(net)
  # conv1_2
  net = tf.nn.conv2d(input=net, filter=weights[1], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[1])
  # relu1_2
  net = tf.nn.relu(net)
  # pool1
  net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # conv2_1
  net = tf.nn.conv2d(input=net, filter=weights[2], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[2])
  # relu2_1
  net = tf.nn.relu(net)
  style_features.append(net)
  # conv2_2
  net = tf.nn.conv2d(input=net, filter=weights[3], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[3])
  # relu2_2
  net = tf.nn.relu(net)
  # pool2
  net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # conv3_1
  net = tf.nn.conv2d(input=net, filter=weights[4], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[4])
  # relu3_1
  net = tf.nn.relu(net)
  style_features.append(net)
  # conv3_2
  net = tf.nn.conv2d(input=net, filter=weights[5], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[5])
  # relu3_2
  net = tf.nn.relu(net)
  # conv3_3
  net = tf.nn.conv2d(input=net, filter=weights[6], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[6])
  # relu3_3
  net = tf.nn.relu(net)
  # conv3_4
  net = tf.nn.conv2d(input=net, filter=weights[7], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[7])
  # relu3_4
  net = tf.nn.relu(net)
  # pool3
  net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # conv4_1
  net = tf.nn.conv2d(input=net, filter=weights[8], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[8])
  # relu4_1
  net = tf.nn.relu(net)
  style_features.append(net)
  # conv4_2
  net = tf.nn.conv2d(input=net, filter=weights[9], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[9])
  # relu4_2
  net = tf.nn.relu(net)
  content_features.append(net)
  # conv4_3
  net = tf.nn.conv2d(input=net, filter=weights[10], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[10])
  # relu4_3
  net = tf.nn.relu(net)
  # conv4_4
  net = tf.nn.conv2d(input=net, filter=weights[11], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[11])
  # relu4_4
  net = tf.nn.relu(net)
  # pool4
  net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # conv5_1
  net = tf.nn.conv2d(input=net, filter=weights[12], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[12])
  # relu5_1
  net = tf.nn.relu(net)
  style_features.append(net)
  # conv5_2
  net = tf.nn.conv2d(input=net, filter=weights[13], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[13])
  # relu5_2
  net = tf.nn.relu(net)
  # conv5_3
  net = tf.nn.conv2d(input=net, filter=weights[14], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[14])
  # relu5_3
  net = tf.nn.relu(net)
  # conv5_4
  net = tf.nn.conv2d(input=net, filter=weights[15], strides=[1, 1, 1, 1], padding='SAME')
  net = tf.nn.bias_add(net, bias=biases[15])
  # relu5_4
  net = tf.nn.relu(net)
  # pool5
  net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  return content_features, style_features