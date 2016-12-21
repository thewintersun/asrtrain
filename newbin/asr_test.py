# coding=utf-8

import logging
import os
from os import path

import tensorflow as tf

try:
  from . import asr
except SystemError:
  import asr

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.4, 'gpu占用内存比例')
tf.app.flags.DEFINE_string('model_dir', "../data/model_distribute/",
                           '保存模型数据的文件夹')


def test():
  """单独的，测试模型效果.

  """
  data_dir = FLAGS.data_dir
  cv_batch_size = FLAGS.cv_batch_size
  cv_maxsize_file = path.join(data_dir, FLAGS.cv_maxsize_file)
  dev_data_config = asr.read_data_config(cv_maxsize_file)
  dev_data = asr.get_dev_data(dev_data_config, cv_batch_size)
  dev_examples_num = dev_data_config.example_number
  dev_num_batches_per_epoch = int(dev_examples_num / cv_batch_size)

  with tf.variable_scope("inference") as scope:
    dev_ctc_in, dev_targets, dev_seq_len = asr.rnn(dev_data, dev_data_config,
                                                   cv_batch_size)

    dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in,
                                                         dev_seq_len)

  edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), dev_targets,
                                   normalize=False)

  batch_error_count = tf.reduce_sum(edit_distance, name="batch_error_count")
  batch_label_count = tf.shape(dev_targets.values)[0]

  local_init = tf.initialize_local_variables()
  saver = tf.train.Saver()

  gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver.restore(session, ckpt.model_checkpoint_path)

    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

    logging.info("从%s载入模型参数, global_step = %d",
                 ckpt.model_checkpoint_path, global_step)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    try:
      dev_error_count = 0
      dev_label_count = 0

      for batch in range(dev_num_batches_per_epoch):
        cv_error_count_value, cv_label_count = session.run(
          [batch_error_count, batch_label_count])

        dev_error_count += cv_error_count_value
        dev_label_count += cv_label_count

      dev_acc_ratio = (dev_label_count - dev_error_count) / dev_label_count

      logging.info("eval:  eval_acc = %.3f ", dev_acc_ratio)
    except tf.errors.OutOfRangeError:
      logging.info("训练完成.")
    finally:
      coord.request_stop()

    coord.join(threads)


def main(_):
  test()


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                      datefmt='%a, %d %b %Y %H:%M:%S',
                      filename='./test.log',
                      filemode='a')

  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  tf.app.run()
