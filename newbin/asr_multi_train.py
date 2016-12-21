# coding=utf-8
'''
训练语音识别
特征数据有eesen的脚本先生成好， 通过本目录的: format_eesen_data.py 程序将eesen生成的文本特征文件转成二进制的文件

asr训练程序通过tf.FixLengthReader读取二进制文件得到特征和label
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import os

import tensorflow as tf

try:
  import asr.train.bin.asr as asr
except ImportError:
  import asr

import logging

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9, 'gpu占用内存比例')
tf.app.flags.DEFINE_string('model_dir', "../data/model_multi/", '保存模型数据的文件夹')
tf.app.flags.DEFINE_string('calc_devices', "/gpu:0|/gpu:1|/gpu:2",
                           '分布式计算的所有device')

tf.app.flags.DEFINE_integer('reload_model', 0, '是否reload之前训练好的模型')
tf.app.flags.DEFINE_integer('print_loss_per_step', 10, '多少步计算后输出loss等信息')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./out.log',
                    filemode='a')

def _get_buckets(max_frame_size, max_label_size):
  """
    生成bucket， 每隔100个生成一个， 数据例子:
    [(200, 170), (300, 170), (400, 170), (500, 170)]
  """
  buckets = []
  start_frame_size = 200
  interval = 100
  
  frame_size = start_frame_size
  while frame_size < max_frame_size:
    buckets.append((frame_size, max_label_size))
    frame_size += 100
  buckets.append((max_frame_size, max_label_size))
  return buckets

def __train_file_open(data_dir):
  # train files
  train_feature_file = os.path.join(data_dir, FLAGS.train_feature_file)
  train_feature_len_file = os.path.join(data_dir, FLAGS.train_feature_len_file)
  train_label_file = os.path.join(data_dir, FLAGS.train_label_file)
  train_label_len_file = os.path.join(data_dir, FLAGS.train_label_len_file)

  # train file open
  train_feature_fr = open(train_feature_file, "r")
  train_feature_len_fr = open(train_feature_len_file, "r")
  train_label_fr = open(train_label_file, "r")
  train_label_len_fr = open(train_label_len_file, "r")

  return train_feature_fr, train_feature_len_fr, train_label_fr, train_label_len_fr

def __dev_file_open(data_dir):
  # dev files
  dev_feature_file = os.path.join(data_dir, FLAGS.dev_feature_file)
  dev_feature_len_file = os.path.join(data_dir, FLAGS.dev_feature_len_file)
  dev_label_file = os.path.join(data_dir, FLAGS.dev_label_file)
  dev_label_len_file = os.path.join(data_dir, FLAGS.dev_label_len_file)

  # dev file open
  dev_feature_fr = open(dev_feature_file, "r")
  dev_feature_len_fr = open(dev_feature_len_file, "r")
  dev_label_fr = open(dev_label_file, "r")
  dev_label_len_fr = open(dev_label_len_file, "r")
  return dev_feature_fr,dev_feature_len_fr,dev_label_fr,dev_label_len_fr

def __close_file(fr_list):
  for fr in fr_list:
    fr.close()

def __get_train_feed_dict(train_reader, train_feature_area,train_seq_len,train_label_area,train_label_len,
                          train_feature_fr,train_feature_len_fr,train_label_fr,train_label_len_fr):
  feed_dict={}
  calc_devices = FLAGS.calc_devices
  device_list = calc_devices.split("|")
  for i in range(len(device_list)):
    patch_data, bucket_id = train_reader.read_data(train_feature_fr, 
                                            train_feature_len_fr, 
                                            train_label_fr, 
                                            train_label_len_fr)
    feature_ids, seq_len_ids, label_ids, label_len_ids = patch_data
    feed_dict[train_feature_area[i]] = feature_ids
    feed_dict[train_seq_len[i]] = seq_len_ids
    feed_dict[train_label_area[i]] = label_ids
    feed_dict[train_label_len[i]] = label_len_ids
  return feed_dict
  
def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)

      grads.append(expanded_g)

    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def tower_loss(scope, train_data_config, batch_size,
                         train_feature_area, train_seq_len, 
                         train_label_area, train_label_len):
  train_ctc_in, train_targets, train_seq_len = asr.rnn(train_data_config,
                                                      batch_size,
                                                      train_feature_area, train_seq_len, 
                                                      train_label_area, train_label_len)

  _ = asr.loss_multi(train_ctc_in, train_targets, train_seq_len)

  losses = tf.get_collection('losses', scope)

  total_loss = tf.add_n(losses, name='total_loss')

  return total_loss


def train():
  logging.info("train start")

  # 得到计算设备列表
  calc_devices = FLAGS.calc_devices
  device_list = calc_devices.split("|")
  
  graph = tf.Graph()

  data_dir = FLAGS.data_dir
  batch_size = FLAGS.batch_size
  dev_batch_size = FLAGS.dev_batch_size
  
  dev_config_file = os.path.join(data_dir, FLAGS.dev_config_file)
  train_config_file = os.path.join(data_dir, FLAGS.train_config_file)
  train_data_config = asr.read_data_config(train_config_file)
  dev_data_config = asr.read_data_config(dev_config_file)

  # 初始化bucket的大小, 初始化reader
  _buckets = _get_buckets(train_data_config.frame_max_length, train_data_config.label_max_length)
  train_reader = asr.BucketReader(_buckets, train_data_config.feature_cols, batch_size)
  dev_reader = asr.BucketReader(_buckets, dev_data_config.feature_cols, dev_batch_size)

  # train file open
  train_feature_fr,train_feature_len_fr,train_label_fr, train_label_len_fr = __train_file_open(data_dir)
  # dev file open
  dev_feature_fr,dev_feature_len_fr,dev_label_fr,dev_label_len_fr = __dev_file_open(data_dir)
  
  # 数据的配置信息，多少个样本，一轮多少个batch
  dev_examples_num = dev_data_config.example_number
  dev_num_batches_per_epoch = int(dev_examples_num / dev_batch_size)
  train_num_examples = train_data_config.example_number
  train_num_batches_per_epoch = int(train_num_examples / (batch_size * len(device_list)))

  global_step = tf.get_variable('global_step', [],
                                initializer=tf.constant_initializer(0),
                                trainable=False, dtype=tf.int32)

  

  with graph.as_default():
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False, dtype=tf.int32)
    # placeholder
    train_seq_len,train_label_len,train_feature_area,train_label_area=[],[],[],[]
    for i in range(len(device_list)):
      train_seq_len.append(tf.placeholder(tf.int32, [batch_size], name="seq_len_placeholder"))
      train_label_len.append(tf.placeholder(tf.int32, [batch_size], name="label_len_placeholder"))
      train_feature_area.append(tf.placeholder(tf.float32,[None, None], name="feature_area_placeholder"))
      train_label_area.append(tf.placeholder(tf.float32,[batch_size,train_data_config.label_max_length],name="label_area_placeholder"))
 
    dev_seq_len = tf.placeholder(tf.int32, [dev_batch_size], name="dev_seq_len_placeholder")
    dev_label_len = tf.placeholder(tf.int32, [dev_batch_size], name="dev_label_len_placeholder")
    dev_feature_area=tf.placeholder(tf.float32,[None, None], name="dev_feature_area_placeholder")
    dev_label_area=tf.placeholder(tf.float32,[dev_batch_size,train_data_config.label_max_length],name="dev_label_area_placeholder")

    opt = tf.train.AdamOptimizer(FLAGS.initial_learning_rate, name="AdamOpt")
  
    tower_grads = []
    i = 0

    with tf.variable_scope("inference"):
      for device in device_list:
        with tf.device(device):
          with tf.name_scope("tower_%d" % i) as scope:
            loss = tower_loss(scope, train_data_config, batch_size,
                                                      train_feature_area[i], train_seq_len[i], 
                                                      train_label_area[i], train_label_len[i])
            tf.get_variable_scope().reuse_variables()
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)

        i += 1

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay,
                                                          global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op, variables_averages_op)
    tf.scalar_summary("train_cost", loss)

    # for cv
    with tf.variable_scope("inference"):
      with tf.device(device_list[0]):
        with tf.name_scope("tower_0") as scope:
          tf.get_variable_scope().reuse_variables()
          dev_ctc_in, dev_targets, dev_seq_len = asr.rnn(dev_data_config,
                                                      dev_batch_size,
                                                      dev_feature_area, dev_seq_len, 
                                                      dev_label_area, dev_label_len)
          dev_decode, _ = tf.nn.ctc_greedy_decoder(dev_ctc_in, dev_seq_len)
          dev_error_count = tf.reduce_sum(
            tf.edit_distance(tf.cast(dev_decode[0], tf.int32), dev_targets,
                             normalize=False))
          dev_label_value_shape = tf.shape(dev_targets.values)
          dev_batch_label_count = dev_label_value_shape[0]

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                               allow_soft_placement=True,
                                               gpu_options=gpu_options))

    num_examples_per_step = batch_size * len(device_list)

    step = 0
    if FLAGS.reload_model == 1:
      ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
      if ckpt:
        saver.restore(session, ckpt.model_checkpoint_path)
        global_step = int(
          ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        logging.info("从%s载入模型参数, global_step = %d",
                     ckpt.model_checkpoint_path, global_step)
        session.run(local_init)
        step = global_step
      else:
        logging.info("Created model with fresh parameters.")
        session.run(init)
        session.run(local_init)
    else:
      logging.info("Created model with fresh parameters.")
      session.run(init)
      session.run(local_init)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.model_dir, session.graph)

    epoch = 0

    while True:
      step = step + 1
      start_time = time.time()

      feed_dict = __get_train_feed_dict(train_reader, 
                train_feature_area,train_seq_len,train_label_area,train_label_len,
                train_feature_fr,train_feature_len_fr,train_label_fr,train_label_len_fr)

      _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

      duration = time.time() - start_time
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration / len(device_list))

      if step % 5 == 1:
        summary_str = session.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
      if step % FLAGS.print_loss_per_step == 0:
        format_str = (
          '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        logging.info(format_str % (
          datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

      #if step % train_num_batches_per_epoch == 0:
      if step % 5 == 0:
        # save model
        saver.save(session, FLAGS.model_dir + "model.ckpt", global_step=step)
        logging.info("保存模型参数.")

        epoch = epoch + 1
        dev_epoch_error_count = 0
        dev_epoch_label_count = 0
        for batch in range(dev_num_batches_per_epoch):
          patch_data, bucket_id = dev_reader.read_data(dev_feature_fr, 
                                            dev_feature_len_fr, 
                                            dev_label_fr, 
                                            dev_label_len_fr)
          feature_ids, seq_len_ids, label_ids, label_len_ids = patch_data
          feed_dict={}
          feed_dict[dev_feature_area] = feature_ids
          feed_dict[dev_seq_len] = seq_len_ids
          feed_dict[dev_label_area] = label_ids
          feed_dict[dev_label_len] = label_len_ids

          dev_error_count_value, dev_label_count_value = session.run(
              [dev_error_count, dev_batch_label_count], feed_dict=feed_dict)
          dev_epoch_error_count += dev_error_count_value
          dev_epoch_label_count += dev_label_count_value
        cv_acc_ratio = (
                         dev_epoch_label_count - dev_epoch_error_count) / dev_epoch_label_count
        logging.info("eval: step = %d epoch = %d eval_acc = %.3f " % (
          step, epoch, cv_acc_ratio))

  __close_file([train_feature_fr,train_feature_len_fr,train_label_fr, train_label_len_fr,
        dev_feature_fr,dev_feature_len_fr,dev_label_fr,dev_label_len_fr])

def main(argv=None):
  train()

if __name__ == '__main__':
  tf.app.run()
