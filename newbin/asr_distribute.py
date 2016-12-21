# coding=utf-8
# 分布式的方式训练语音识别的声学模型
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from os import path

import tensorflow as tf

import asr

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.45, 'gpu占用内存比例')
tf.app.flags.DEFINE_string('model_dir', "../data/model_distribute/",
                           '保存模型数据的文件夹')
tf.app.flags.DEFINE_string('cuda_visible_devices', "0", '使用第几个GPU')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")


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
    batch_size = FLAGS.batch_size
    dev_batch_size = FLAGS.dev_batch_size

    dev_config_file = path.join(data_dir, FLAGS.dev_config_file)
    train_config_file = path.join(data_dir, FLAGS.train_config_file)
    train_data_config = asr.read_data_config(train_config_file)
    dev_data_config = asr.read_data_config(dev_config_file)

    # 初始化bucket的大小, 初始化reader
    _buckets = _get_buckets(train_data_config.frame_max_length, train_data_config.label_max_length)
    train_reader = asr.BucketReader(_buckets, train_data_config.feature_cols, batch_size)
    dev_reader = asr.BucketReader(_buckets, dev_data_config.feature_cols, dev_batch_size)

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

    # 数据的配置信息，多少个样本，一轮多少个batch
    dev_examples_num = dev_data_config.example_number
    dev_num_batches_per_epoch = int(dev_examples_num / dev_batch_size)
    train_num_examples = train_data_config.example_number
    train_num_batches_per_epoch = int(train_num_examples / batch_size)

    initial_learning_rate = FLAGS.initial_learning_rate

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
      global_step = tf.get_variable('global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False, dtype=tf.int32)

      #optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(initial_learning_rate)

      # placeholder
      train_seq_len = tf.placeholder(tf.int32, [batch_size], name="seq_len_placeholder")
      train_label_len = tf.placeholder(tf.int32, [batch_size], name="label_len_placeholder")
      train_feature_area=tf.placeholder(tf.float32,[None, None], name="feature_area_placeholder")
      train_label_area=tf.placeholder(tf.float32,[batch_size,train_data_config.label_max_length],name="label_area_placeholder")
 
      dev_seq_len = tf.placeholder(tf.int32, [dev_batch_size], name="dev_seq_len_placeholder")
      dev_label_len = tf.placeholder(tf.int32, [dev_batch_size], name="dev_label_len_placeholder")
      dev_feature_area=tf.placeholder(tf.float32,[None, None], name="dev_feature_area_placeholder")
      dev_label_area=tf.placeholder(tf.float32,[dev_batch_size,train_data_config.label_max_length],name="dev_label_area_placeholder")

      with tf.variable_scope("inference") as scope:
        train_ctc_in, train_targets, train_seq_len = asr.rnn(train_data_config,
                                                      batch_size,
                                                      train_feature_area, train_seq_len, 
                                                      train_label_area, train_label_len)

        scope.reuse_variables()
        dev_ctc_in, dev_targets, dev_seq_len = asr.rnn(dev_data_config,
                                                   dev_batch_size,
                                                    dev_feature_area, dev_seq_len, 
                                                      dev_label_area, dev_label_len)

      train_ctc_losses = tf.nn.ctc_loss(train_ctc_in, train_targets, train_seq_len)
      train_cost = tf.reduce_mean(train_ctc_losses, name="train_cost")
      # 限制梯度范围
      grads_and_vars = optimizer.compute_gradients(train_cost)
      capped_grads_and_vars = [(tf.clip_by_value(gv[0], -50.0, 50.0), gv[1]) for gv
                           in grads_and_vars]

      
      if issync == 1:
        #同步模式计算更新梯度
        rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                replicas_to_aggregate=len(
                                                  worker_hosts),
                                                replica_id=FLAGS.task_index,
                                                total_num_replicas=len(
                                                  worker_hosts),
                                                use_locking=True)
        train_op = rep_op.apply_gradients(capped_grads_and_vars,
                                       global_step=global_step)
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
      else:
        #异步模式计算更新梯度
        train_op = optimizer.apply_gradients(capped_grads_and_vars,
                                       global_step=global_step)

      #记录loss值，显示到tensorboard上
      #tf.scalar_summary("train_cost", train_cost)

      #dev 评估
      dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in, dev_seq_len)
      dev_edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), dev_targets,
                                   normalize=False)
      dev_batch_error_count = tf.reduce_sum(dev_edit_distance)
      dev_batch_label_count = tf.shape(dev_targets.values)[0]
  
      # train 评估
      train_decoded, train_log_prob = tf.nn.ctc_greedy_decoder(train_ctc_in, train_seq_len)
      train_edit_distance = tf.edit_distance(tf.to_int32(train_decoded[0]), train_targets,
                                   normalize=False)
      train_batch_error_count = tf.reduce_sum(train_edit_distance)
      train_batch_label_count = tf.shape(train_targets.values)[0]

      # 初始化各种
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
        # 如果是同步模式
        if FLAGS.task_index == 0 and issync == 1:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_token_op)

        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir, sess.graph)

        step = 0
        valid_step = 0
        train_acc_step = 0
        epoch = 0
        while not sv.should_stop() and step < 100000000:
          patch_data, bucket_id = train_reader.read_data(train_feature_fr, 
                                              train_feature_len_fr, 
                                              train_label_fr, 
                                              train_label_len_fr)
          feature_ids, seq_len_ids, label_ids, label_len_ids = patch_data
          feed_dict={train_feature_area: feature_ids, train_seq_len: seq_len_ids, 
                train_label_area:label_ids, train_label_len:label_len_ids}
          _,loss, step = sess.run([train_op,train_cost,global_step], feed_dict=feed_dict)

          if (step - train_acc_step) > 1000 and FLAGS.task_index == 0:
            train_acc_step = step
            patch_data, bucket_id = train_reader.read_data(train_feature_fr,
                                              train_feature_len_fr,
                                              train_label_fr,
                                              train_label_len_fr)
            feature_ids, seq_len_ids, label_ids, label_len_ids = patch_data
            feed_dict={train_feature_area: feature_ids, train_seq_len: seq_len_ids,
                train_label_area:label_ids, train_label_len:label_len_ids}
            train_error_count_value, train_label_count = sess.run(
              [train_batch_error_count, train_batch_label_count],
              feed_dict=feed_dict)
            train_acc_ratio = (train_label_count - train_error_count_value) / train_label_count
            logging.info("eval: step = %d loss = %.3f train_acc = %.3f ", step, loss,
						                 train_acc_ratio)

		      # 当跑了steps_to_validate个step，并且是主的worker节点的时候， 评估下数据
		      # 因为是分布式的，各个节点分配了不同的step，所以不能用 % 是否等于0的方法
          if step - valid_step > train_num_batches_per_epoch and FLAGS.task_index == 0:
            epoch += 1
            valid_step = step
            dev_error_count = 0
            dev_label_count = 0

            for batch in range(dev_num_batches_per_epoch):
              patch_data, bucket_id = dev_reader.read_data(dev_feature_fr,
                                              dev_feature_len_fr,
                                              dev_label_fr,
                                              dev_label_len_fr)
              feature_ids, seq_len_ids, label_ids, label_len_ids = patch_data
              feed_dict={dev_feature_area: feature_ids, dev_seq_len: seq_len_ids, 
                dev_label_area:label_ids, dev_label_len:label_len_ids}

              dev_error_count_value, dev_label_count_value = sess.run(
                [dev_batch_error_count, dev_batch_label_count],
                feed_dict=feed_dict)
              dev_error_count += dev_error_count_value
              dev_label_count += dev_label_count_value

            dev_acc_ratio = (dev_label_count - dev_error_count) / dev_label_count
            logging.info("epoch: %d eval: step = %d eval_acc = %.3f ",
						 epoch,
						 step, dev_acc_ratio)


    train_feature_fr.close()
    train_feature_len_fr.close()
    train_label_fr.close()
    train_label_len_fr.close()

    dev_feature_fr.close()
    dev_feature_len_fr.close()
    dev_label_fr.close()
    dev_label_len_fr.close()

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
