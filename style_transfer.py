import os
from PIL import Image
import numpy as np
import tensorflow as tf
from operator import mul
import datetime
import image_utils
import vgg

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('content_image', 'data/content_image.jpg', 'content image')
tf.app.flags.DEFINE_string('style_image', 'data/style_image.jpg', 'style image')
tf.app.flags.DEFINE_string('model_filename', 'data/imagenet-vgg-verydeep-19.mat', 'vgg model')
tf.app.flags.DEFINE_string('tensorboard_path', 'tensorboard/', 'tensorboard output')
tf.app.flags.DEFINE_float('learning_rate', 10.0, 'learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9, 'learning rate decay factor')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'decay steps')
tf.app.flags.DEFINE_integer('max_iteration', 2000, 'max iteration')


def style_transfer(content_image_filename, style_image_filename, model_filename, tensorboard_path,
                   learning_rate, learning_rate_decay_factor, decay_steps, max_iteration):
  # image process
  resized_width = 1000
  raw_content_image = Image.open(content_image_filename)
  raw_style_image = Image.open(style_image_filename)
  if raw_style_image.mode == 'L':
    raw_content_image = raw_content_image.convert('L')
  raw_content_image = raw_content_image.convert('RGB')
  raw_style_image = raw_style_image.convert('RGB')
  raw_content_image = raw_content_image.resize((resized_width, int(resized_width * raw_content_image.height / raw_content_image.width)),
                                               resample=Image.LANCZOS)
  raw_style_image = raw_style_image.resize((resized_width, int(resized_width * raw_style_image.height / raw_style_image.width)),
                                           resample=Image.LANCZOS)
  content_image = np.array(raw_content_image, dtype=np.float32)
  style_image = np.array(raw_style_image, dtype=np.float32)
  if len(content_image.shape) != 3 or content_image.shape[2] != 3 or len(style_image.shape) != 3 or style_image.shape[2] != 3:
    print 'image format error!'
    return

  model_layers, mean_pixel = vgg.load_model_data(model_filename)

  # content image features
  mean_content_image = np.array([content_image - mean_pixel], dtype=np.float32)
  content_features, _ = vgg.vgg_19(mean_content_image, model_layers)

  # style image features
  mean_style_image = np.array([style_image - mean_pixel], dtype=np.float32)
  _, style_features = vgg.vgg_19(mean_style_image, model_layers)
  style_gram_features = []
  for features in style_features:
    features = tf.reshape(features, shape=[-1, features.get_shape()[-1].value])
    features_size = reduce(lambda x, y: x.value * y.value, features.get_shape())
    gram = tf.matmul(tf.transpose(features), features) / features_size
    style_gram_features.append(gram)

  # generated image features
  initial_image = tf.random_normal((1,) + content_image.shape, dtype=tf.float32) * 0.256
  generated_image = tf.Variable(initial_image)
  generated_content_features, generated_style_features = vgg.vgg_19(generated_image, model_layers)
  generated_gram_features = []
  for features in generated_style_features:
    features = tf.reshape(features, shape=[-1, features.get_shape()[-1].value])
    features_size = reduce(lambda x, y: x.value * y.value, features.get_shape())
    gram = tf.matmul(tf.transpose(features), features) / features_size
    generated_gram_features.append(gram)

  # content loss
  content_weight = 5.0
  content_loss = 0.0
  for (content_feature, generated_content_feature) in zip(content_features, generated_content_features):
    content_feature_size = reduce(lambda x, y: x * y, content_feature.get_shape()).value
    content_loss += 2 * tf.nn.l2_loss(generated_content_feature - content_feature) / content_feature_size
  content_loss *= content_weight

  # style loss
  style_weight = 500.0
  style_layer_weight = 0.2
  style_loss = 0.0
  for (style_gram_feature, generated_gram_feature) in zip(style_gram_features, generated_gram_features):
    style_gram_size = reduce(lambda x, y: x.value * y.value, style_gram_feature.get_shape())
    style_loss += style_layer_weight * 2 * tf.nn.l2_loss(generated_gram_feature - style_gram_feature) / style_gram_size
  style_loss *= style_weight

  # tv loss
  tv_weight = 100.0
  tv_x_size = reduce(mul, (x.value for x in generated_image[:, :, 1:, :].get_shape()), 1)
  tv_y_size = reduce(mul, (y.value for y in generated_image[:, 1:, :, :].get_shape()), 1)
  tv_loss = tv_weight * 2 * (
    (tf.nn.l2_loss(generated_image[:, :, 1:, :] - generated_image[:, :, :content_image.shape[1] - 1, :]) / tv_x_size) +
    (tf.nn.l2_loss(generated_image[:, 1:, :, :] - generated_image[:, :content_image.shape[0] - 1, :, :]) / tv_y_size))

  # photorealism regularization, mattion laplacian matrix
  matting_laplacian_weight = 50000.0
  laplacian_content_image = raw_content_image.resize((10, int(10 * raw_content_image.height / raw_content_image.width)),
                                                     resample=Image.LANCZOS)
  laplacian_content_image = np.array(laplacian_content_image, dtype=np.float32)
  laplacian_generated_image = tf.image.resize_bilinear(generated_image, size=laplacian_content_image.shape[0:2])

  matting_laplacian_matrix = image_utils.compute_matting_laplacian(laplacian_content_image)
  matting_laplacian_matrix1 = image_utils.getlaplacian(laplacian_content_image, np.zeros(shape=(laplacian_content_image.shape[0:2])))

  matting_laplacian_sparse_tensor = tf.SparseTensor(indices=np.array([matting_laplacian_matrix.row, matting_laplacian_matrix.col]).T,
                                                    values=matting_laplacian_matrix.data,
                                                    dense_shape=matting_laplacian_matrix.shape)
  matting_laplacian_tensor = tf.sparse_tensor_to_dense(matting_laplacian_sparse_tensor, default_value=0.0, validate_indices=False)
  matting_laplacian_loss = 0.0
  for dim in range(3):
    dim_generated_image = tf.slice(laplacian_generated_image, [0, 0, 0, dim], [-1, -1, -1, 1])
    dim_generated_image = tf.reshape(dim_generated_image, shape=[-1, 1])
    dim_generated_image_product = tf.matmul(tf.matmul(dim_generated_image, matting_laplacian_tensor, transpose_a=True), dim_generated_image)
    dim_generated_image_product = tf.reshape(dim_generated_image_product, shape=[])
    matting_laplacian_loss += dim_generated_image_product
  matting_laplacian_loss *= matting_laplacian_weight

  # total loss
  loss = content_loss + style_loss + tv_loss

  # optimizer
  with tf.device('/cpu:0'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
  lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)
  optimizer = tf.train.AdamOptimizer(lr)
  train_op = optimizer.minimize(loss, global_step=global_step)

  # summary
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('lr', lr)
  if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    sess.run(init_op)
    start_time = datetime.datetime.now()
    for i in range(max_iteration):
      _, loss_value, step, summary_value, lr_value = sess.run([train_op, loss, global_step, summary_op, lr])
      end_time = datetime.datetime.now()
      print('[{}] Step: {}, loss: {}, lr: {}'.format(end_time - start_time, step, loss_value, lr_value))
      writer.add_summary(summary_value, step)
      if step % 100 == 0 or i == max_iteration - 1:
        stylized_image = generated_image.eval()
        stylized_image = stylized_image.reshape(content_image.shape) + mean_pixel
        stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
        Image.fromarray(stylized_image).save('data/stylized_' + str(step) + '.jpg', quality=95)
        print 'success saved stylized_' + str(step) + '.jpg to data/'
      start_time = end_time
  print 'style transfer done!'


def main(_):
  content_image_filename = FLAGS.content_image
  style_image_filename = FLAGS.style_image
  model_filename = FLAGS.model_filename
  tensorboard_path = FLAGS.tensorboard_path
  learning_rate = FLAGS.learning_rate
  learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
  decay_steps = FLAGS.decay_steps
  max_iteration = FLAGS.max_iteration
  style_transfer(content_image_filename, style_image_filename, model_filename, tensorboard_path,
                 learning_rate, learning_rate_decay_factor, decay_steps, max_iteration)


if __name__ == '__main__':
  tf.app.run()
