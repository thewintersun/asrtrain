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
from os import path

import tensorflow as tf

try:
  import src.train.bin.asr as asr
except ImportError:
  import asr

import logging

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9, 'gpu占用内存比例')
tf.app.flags.DEFINE_string('model_dir', "../data/model_multi/", '保存模型数据的文件夹')
tf.app.flags.DEFINE_string('calc_devices', "/gpu:0|/gpu:1|/gpu:2",
                           '分布式计算的所有device')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 10, '多少个epoch之后学习率下降')
tf.app.flags.DEFINE_integer('reload_model', 0, '是否reload之前训练好的模型')
tf.app.flags.DEFINE_integer('print_loss_per_step', 10, '多少步计算后输出loss等信息')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./out.log',
                    filemode='a')


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


def tower_loss(scope, train_max_size_list, batch_size):
  train_data = asr.distort_inputs(train_max_size_list, batch_size)

  train_logits, train_targets, train_seq_len = asr.rnn(train_data,
                                                       train_max_size_list,
                                                       batch_size)

  _ = asr.loss_multi(train_logits, train_targets, train_seq_len)

  losses = tf.get_collection('losses', scope)

  total_loss = tf.add_n(losses, name='total_loss')

  return total_loss


def train():
  logging.info("train start")

  graph = tf.Graph()

  data_dir = FLAGS.data_dir
  cv_maxsize_file = path.join(data_dir, FLAGS.cv_maxsize_file)
  train_maxsize_file = path.join(data_dir, FLAGS.train_maxsize_file)

  train_max_size_list = asr.read_data_config(train_maxsize_file)
  train_num_examples = train_max_size_list.example_number

  cv_max_size_list = asr.read_data_config(cv_maxsize_file)
  cv_num_examples = cv_max_size_list.example_number

  batch_size = FLAGS.batch_size
  cv_batch_size = FLAGS.cv_batch_size

  num_epochs_per_decay = FLAGS.num_epochs_per_decay
  initial_learning_rate = FLAGS.initial_learning_rate
  learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
  moving_average_decay = FLAGS.moving_average_decay

  # 得到计算设备列表
  calc_devices = FLAGS.calc_devices
  device_list = calc_devices.split("|")

  num_batches_per_epoch = int(train_num_examples / batch_size)

  decay_steps = int(
    num_batches_per_epoch * num_epochs_per_decay / len(device_list))

  with graph.as_default():
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False, dtype=tf.int32)

    lr = tf.train.exponential_decay(initial_learning_rate, global_step,
                                    decay_steps, learning_rate_decay_factor,
                                    staircase=True)

    opt = tf.train.AdamOptimizer(lr)
    tower_grads = []
    i = 0

    with tf.variable_scope("inference"):
      for device in device_list:
        with tf.device(device):
          with tf.name_scope("tower_%d" % i) as scope:
            loss = tower_loss(scope, train_max_size_list, batch_size)
            tf.get_variable_scope().reuse_variables()
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)

        i += 1

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,
                                                          global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)
    tf.scalar_summary("train_cost", loss)

    # for cv
    cv_data = asr.get_dev_data(cv_max_size_list, cv_batch_size)
    with tf.variable_scope("inference"):
      with tf.device(device_list[0]):
        with tf.name_scope("tower_0") as scope:
          tf.get_variable_scope().reuse_variables()
          cv_logits, cv_targets, cv_seq_len = asr.rnn(cv_data, cv_max_size_list,
                                                      cv_batch_size)
          cv_decode, cv_log_prob = tf.nn.ctc_beam_search_decoder(cv_logits,
                                                                 cv_seq_len)
          cv_error_count = tf.reduce_sum(
            tf.edit_distance(tf.cast(cv_decode[0], tf.int32), cv_targets,
                             normalize=False))
          cv_label_value_shape = tf.shape(cv_targets.values)
          cv_batch_label_count = cv_label_value_shape[0]

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                               allow_soft_placement=True,
                                               gpu_options=gpu_options))

    num_examples_per_step = FLAGS.batch_size * len(device_list)

    step = 0
    if FLAGS.reload_model == 1:
      ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
      if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    try:
      epoch = 0

      cv_num_batches_per_epoch = int(cv_num_examples / cv_batch_size)
      train_num_batches_per_epoch = int(
        train_num_examples / (cv_batch_size * len(device_list)))

      while not coord.should_stop():
        step = step + 1
        start_time = time.time()
        # _, loss_value      = session.run([apply_gradient_op, loss])
        _, loss_value = session.run([train_op, loss])
        duration = time.time() - start_time
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration / len(device_list))

        if step % 5 == 1:
          summary_str = session.run(summary_op)
          summary_writer.add_summary(summary_str, step)
        if step % FLAGS.print_loss_per_step == 0:
          format_str = (
            '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
          print(format_str % (
            datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
          logging.info(format_str % (
            datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

        if step % train_num_batches_per_epoch == 0:
          # save model
          saver.save(session, FLAGS.model_dir + "model.ckpt", global_step=step)
          logging.info("保存模型参数.")

          epoch = epoch + 1
          cv_epoch_error_count = 0
          cv_epoch_label_count = 0
          for batch in range(cv_num_batches_per_epoch):
            cv_error_count_value, cv_label_count = session.run(
              [cv_error_count, cv_batch_label_count])
            cv_epoch_error_count += cv_error_count_value
            cv_epoch_label_count += cv_label_count
          cv_acc_ratio = (
                           cv_epoch_label_count - cv_epoch_error_count) / cv_epoch_label_count
          print("eval: step = %d epoch = %d eval_acc = %.3f " % (
            step, epoch, cv_acc_ratio))
          logging.info("eval: step = %d epoch = %d eval_acc = %.3f " % (
            step, epoch, cv_acc_ratio))

    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      coord.request_stop()

    coord.join(threads)


def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run()
