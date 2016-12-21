# coding=utf-8
# 分布式的方式训练语音识别的声学模型
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from os import path

import asr
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.95, 'gpu占用内存比例')
tf.app.flags.DEFINE_string('model_dir', "../data/model_distribute/",
                           '保存模型数据的文件夹')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 12, '多少个epoch之后学习率下降')
tf.app.flags.DEFINE_string('cuda_visible_devices', "0", '使用第几个GPU')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")


def train():
  """训练LSTM + CTC的语音识别系统.
  
  """

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

  server = tf.train.Server(cluster,
                           config=tf.ConfigProto(gpu_options=gpu_options),
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  issync = FLAGS.issync

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    data_dir = FLAGS.data_dir
    cv_maxsize_file = path.join(data_dir, FLAGS.cv_maxsize_file)
    train_maxsize_file = path.join(data_dir, FLAGS.train_maxsize_file)

    batch_size = FLAGS.batch_size
    cv_batch_size = FLAGS.cv_batch_size

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

    initial_learning_rate = FLAGS.initial_learning_rate
    learning_rate_decay_factor = FLAGS.learning_rate_decay_factor

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
      global_step = tf.get_variable('global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False, dtype=tf.int32)
      #optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(initial_learning_rate)

      with tf.variable_scope("inference") as scope:
        ctc_input, train_targets, train_seq_len = asr.rnn(train_data,
                                                          train_data_config,
                                                          batch_size)

        scope.reuse_variables()
        dev_ctc_in, dev_targets, dev_seq_len = asr.rnn(dev_data,
                                                       dev_data_config,
                                                       cv_batch_size)

      example_losses = tf.nn.ctc_loss(ctc_input, train_targets, train_seq_len)
      train_cost = tf.reduce_mean(example_losses)

      if issync == 1:
        rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                replicas_to_aggregate=len(
                                                  worker_hosts),
                                                replica_id=FLAGS.task_index,
                                                total_num_replicas=len(
                                                  worker_hosts),
                                                use_locking=True)
        train_op = rep_op.minimize(train_cost, global_step=global_step)
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
      else:
        train_op = optimizer.minimize(train_cost, global_step=global_step)

      tf.scalar_summary("train_cost", train_cost)

      train_decoded, train_log_prob = tf.nn.ctc_greedy_decoder(ctc_input,
                                                               train_seq_len)
      dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in,
                                                           dev_seq_len)

      train_edit_distance = tf.edit_distance(tf.to_int32(train_decoded[0]),
                                             train_targets,
                                             normalize=False)
      edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), dev_targets,
                                       normalize=False)

      train_batch_error_count = tf.reduce_sum(train_edit_distance)
      train_batch_label_count = tf.shape(train_targets.values)[0]
      batch_error_count = tf.reduce_sum(edit_distance)
      batch_label_count = tf.shape(dev_targets.values)[0]

      init_op = tf.global_variables_initializer()
      local_init = tf.local_variables_initializer()
      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()

      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                               logdir=FLAGS.model_dir,
                               init_op=init_op,
                               local_init_op=local_init,
                               summary_op=summary_op,
                               saver=saver,
                               global_step=global_step,
                               save_model_secs=600)
      with sv.prepare_or_wait_for_session(server.target) as sess:

        if FLAGS.task_index == 0 and issync == 1:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_token_op)

        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir, sess.graph)

        step = 0
        valid_step = 0
        train_acc_step = 0
        epoch = 0
        while not sv.should_stop() and step < 100000000:

          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(sess=sess, coord=coord)

          try:
            while not coord.should_stop():
              train_cost_value, _, step  = sess.run(
                [train_cost, train_op, global_step ])

              if step % 100 == 0:
                logging.info("step: %d,  loss: %f" % (
                step,  train_cost_value))

              if step - train_acc_step > 1000 and FLAGS.task_index == 0:
                train_acc_step = step
                train_error_count_value, train_label_count = sess.run(
                  [train_batch_error_count, train_batch_label_count])
                train_acc_ratio = (
                                  train_label_count - train_error_count_value) / train_label_count
                logging.info("eval: step = %d train_acc = %.3f ", step,
                             train_acc_ratio)

              # 当跑了steps_to_validate个step，并且是主的worker节点的时候， 评估下数据
              # 因为是分布式的，各个节点分配了不同的step，所以不能用 % 是否等于0的方法
              if step - valid_step > train_num_batches_per_epoch and FLAGS.task_index == 0:
                epoch += 1

                valid_step = step
                dev_error_count = 0
                dev_label_count = 0

                for batch in range(dev_num_batches_per_epoch):
                  cv_error_count_value, cv_label_count = sess.run(
                    [batch_error_count, batch_label_count])

                  dev_error_count += cv_error_count_value
                  dev_label_count += cv_label_count

                dev_acc_ratio = (
                                dev_label_count - dev_error_count) / dev_label_count

                logging.info("epoch: %d eval: step = %d eval_acc = %.3f ",
                             epoch,
                             step, dev_acc_ratio)

          except tf.errors.OutOfRangeError:
            print("Done training after reading all data")
          finally:
            coord.request_stop()

          # Wait for threads to exit
          coord.join(threads)


def main(_):
  train()


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                      datefmt='%a, %d %b %Y %H:%M:%S',
                      filename='./out.log',
                      filemode='a')
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices

  tf.app.run()
