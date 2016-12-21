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

  global_step = tf.get_variable('global_step', [],
                                initializer=tf.constant_initializer(0),
                                trainable=False, dtype=tf.int32)

  optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate, name="AdamOpt")
  # optimizer = tf.train.MomentumOptimizer(lr, 0.9)

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
  train_op = optimizer.apply_gradients(capped_grads_and_vars,
                                       global_step=global_step)

  tf.scalar_summary("train_cost", train_cost)

  #dev
  dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in, dev_seq_len)
  dev_edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), dev_targets,
                                   normalize=False)
  dev_batch_error_count = tf.reduce_sum(dev_edit_distance)
  dev_batch_label_count = tf.shape(dev_targets.values)[0]
  
  # train
  train_decoded, train_log_prob = tf.nn.ctc_greedy_decoder(train_ctc_in, train_seq_len)
  train_edit_distance = tf.edit_distance(tf.to_int32(train_decoded[0]), train_targets,
                                   normalize=False)
  train_batch_error_count = tf.reduce_sum(train_edit_distance)
  train_batch_label_count = tf.shape(train_targets.values)[0]

  #init
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

    step = 0
    epoch = 0
    while True:
      step += 1
      patch_data, bucket_id = train_reader.read_data(train_feature_fr, 
                                              train_feature_len_fr, 
                                              train_label_fr, 
                                              train_label_len_fr)
      feature_ids, seq_len_ids, label_ids, label_len_ids = patch_data
      feed_dict={train_feature_area: feature_ids, train_seq_len: seq_len_ids, 
              train_label_area:label_ids, train_label_len:label_len_ids}
      _,loss = session.run([train_op,train_cost], feed_dict=feed_dict)

      
      if step % 20 ==0:
        start_time = time.time()
        patch_data, bucket_id = train_reader.read_data(train_feature_fr,
                                              train_feature_len_fr,
                                              train_label_fr,
                                              train_label_len_fr)
        feature_ids, seq_len_ids, label_ids, label_len_ids = patch_data
        feed_dict={train_feature_area: feature_ids, train_seq_len: seq_len_ids,
                train_label_area:label_ids, train_label_len:label_len_ids}
        train_error_count_value, train_label_count = session.run(
              [train_batch_error_count, train_batch_label_count],
              feed_dict=feed_dict)
        train_acc_ratio = (train_label_count - train_error_count_value) / train_label_count
        duration = time.time() - start_time
        examples_per_sec = batch_size / duration

        logging.info(
            'step %d, loss = %.2f (%.1f examples/sec) bucketid=%d train_acc= %.3f',
            step, loss, examples_per_sec, bucket_id, train_acc_ratio)

      if step % train_num_batches_per_epoch == 0:
          saver.save(session, FLAGS.model_dir + "model.ckpt", global_step=step)
          logging.info("保存模型参数.")
          epoch += 1
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

            dev_error_count_value, dev_label_count_value = session.run(
              [dev_batch_error_count, dev_batch_label_count],
              feed_dict=feed_dict)

            dev_error_count += dev_error_count_value
            dev_label_count += dev_label_count_value

          dev_acc_ratio = (dev_label_count - dev_error_count) / dev_label_count

          logging.info("eval: step = %d epoch = %d eval_acc = %.3f ",
                       step, epoch, dev_acc_ratio)

    
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
  logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s",
                      filename='./out.log',
                      filemode='a',
                      level=logging.INFO)

  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  tf.app.run()
