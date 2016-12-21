# coding=utf-8
"""
训练语音识别
特征数据有eesen的脚本先生成好， 通过本目录的: format_eesen_data.py 程序将eesen生成的文本
特征文件转成二进制的文件.

asr训练程序通过tf.FixLengthReader读取二进制文件得到特征和label
"""
import logging
import os
import time
from os import path

import tensorflow as tf

try:
  from . import asr
except SystemError:
  import asr

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9, 'gpu占用内存比例')
tf.app.flags.DEFINE_string('model_dir', "../data/model/", '保存模型数据的文件夹')
tf.app.flags.DEFINE_integer('reload_model', 0, '是否reload之前训练好的模型')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 10, '多少个epoch之后学习率下降')


def train():
  """训练LSTM + CTC的语音识别系统.

  """
  data_dir = FLAGS.data_dir
  batch_size = FLAGS.batch_size
  cv_batch_size = FLAGS.cv_batch_size

  cv_maxsize_file = path.join(data_dir, FLAGS.cv_maxsize_file)
  train_maxsize_file = path.join(data_dir, FLAGS.train_maxsize_file)
  train_data_config = asr.read_data_config(train_maxsize_file)
  dev_data_config = asr.read_data_config(cv_maxsize_file)
  train_data = asr.distort_inputs(train_data_config, batch_size)
  dev_data = asr.get_dev_data(dev_data_config, cv_batch_size)

  dev_examples_num = dev_data_config.example_number
  dev_num_batches_per_epoch = int(dev_examples_num / cv_batch_size)
  train_num_examples = train_data_config.example_number
  train_num_batches_per_epoch = int(train_num_examples / batch_size)

  # 多少个step之后， 学习率下降
  decay_steps = int(train_num_batches_per_epoch * FLAGS.num_epochs_per_decay)

  global_step = tf.get_variable('global_step', [],
                                initializer=tf.constant_initializer(0),
                                trainable=False, dtype=tf.int32)

  lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step,
                                  decay_steps, FLAGS.learning_rate_decay_factor,
                                  staircase=True, name="decay_learning_rate")

  optimizer = tf.train.AdamOptimizer(lr, name="AdamOpt")
  # optimizer = tf.train.MomentumOptimizer(lr, 0.9)

  with tf.variable_scope("inference") as scope:
    ctc_input, train_targets, train_seq_len = asr.rnn(train_data,
                                                      train_data_config,
                                                      batch_size)

    scope.reuse_variables()
    dev_ctc_in, dev_targets, dev_seq_len = asr.rnn(dev_data, dev_data_config,
                                                   cv_batch_size)

  example_losses = tf.nn.ctc_loss(ctc_input, train_targets, train_seq_len)
  train_cost = tf.reduce_mean(example_losses, name="train_cost")
  grads_and_vars = optimizer.compute_gradients(train_cost)
  capped_grads_and_vars = [(tf.clip_by_value(gv[0], -50.0, 50.0), gv[1]) for gv
                           in grads_and_vars]
  train_op = optimizer.apply_gradients(capped_grads_and_vars,
                                       global_step=global_step)
  #train_op = optimizer.minimize(train_cost, global_step=global_step)

  tf.scalar_summary("train_cost", train_cost)

  dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in, dev_seq_len)

  edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), dev_targets,
                                   normalize=False)

  batch_error_count = tf.reduce_sum(edit_distance, name="batch_error_count")
  batch_label_count = tf.shape(dev_targets.values)[0]
  init = tf.global_variables_initializer()
  local_init = tf.local_variables_initializer()
  saver = tf.train.Saver()
  summary_op = tf.merge_all_summaries()

  gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    if FLAGS.reload_model == 1:
      ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
      saver.restore(session, ckpt.model_checkpoint_path)

      global_step = int(
        ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

      logging.info("从%s载入模型参数, global_step = %d",
                   ckpt.model_checkpoint_path, global_step)
    else:
      logging.info("Created model with fresh parameters.")
      session.run(init)
      session.run(local_init)

    summary_writer = tf.train.SummaryWriter(FLAGS.model_dir, session.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    step = 0
    epoch = 0

    try:
      while not coord.should_stop():
        step += 1
        start_time = time.time()
        train_cost_value, _ = session.run([train_cost, train_op])
        duration = time.time() - start_time
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        if step % 2 == 0:
          summary_str = session.run(summary_op)
          summary_writer.add_summary(summary_str, step)
        if step % 20 == 0:
          logging.info(
            'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)',
            step, train_cost_value, examples_per_sec, sec_per_batch)

        if step % train_num_batches_per_epoch == 0:
          saver.save(session, FLAGS.model_dir + "model.ckpt", global_step=step)
          logging.info("保存模型参数.")
          epoch += 1
          dev_error_count = 0
          dev_label_count = 0

          for batch in range(dev_num_batches_per_epoch):
            cv_error_count_value, cv_label_count = session.run(
              [batch_error_count, batch_label_count])

            dev_error_count += cv_error_count_value
            dev_label_count += cv_label_count

          dev_acc_ratio = (dev_label_count - dev_error_count) / dev_label_count

          logging.info("eval: step = %d epoch = %d eval_acc = %.3f ",
                       step, epoch, dev_acc_ratio)
    except tf.errors.OutOfRangeError:
      logging.info("训练完成.")
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    coord.join(threads)


def main(_):
  train()


if __name__ == '__main__':
  logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s",
                      level=logging.INFO)

  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  tf.app.run()
