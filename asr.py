# coding=utf-8
# 定义模型结构, 数据预处理.
from os import path

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 10, """batch size""")
tf.app.flags.DEFINE_integer('cv_batch_size', 10, """cv的测试的batch size""")

tf.app.flags.DEFINE_string('data_dir', '/data/700h2/',
                           """train data dir""")

tf.app.flags.DEFINE_string("train_files",
                           "train.0,train.1,train.2",
                           """在data_dir文件下，用来训练的文件列表，用英文逗号分隔""")
tf.app.flags.DEFINE_string("train_maxsize_file",
                           "train.config",
                           """记录每个小文件的每行的最大的值的个数""")

tf.app.flags.DEFINE_string('eval_file', 'cv.0',
                           """cv的样本文件,只有一个""")

tf.app.flags.DEFINE_string('cv_maxsize_file', "cv.config",
                           """记录每个小文件的每行的最大的值的个数""")

tf.app.flags.DEFINE_integer('num_layers', 5, """lstm网络的lstm的层数""")
tf.app.flags.DEFINE_integer('num_hidden', 320, "lstm网络每层的节点数")
tf.app.flags.DEFINE_integer('num_epochs', 200, """迭代的次数""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, """学习率""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.8, "学习率下降的百分比")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """""")


class TrainDataInfo:
  """存放训练数据的一些信息，比如一个样本的长度
  一共有多少个样本等信息
  """


  def __init__(self):
    self.example_max_length = 0  # 一个样本所有部分占用固定最大的float32的长度
    self.example_feature_max_length = 0  # 一个样本的特征部分的固定占用最大float32的长度
    self.example_label_max_len = 0  # 一个样本的label部分的固定占用最大float32的长度
    self.example_number = 0  # 一共有多少个样本
    self.example_label_count = 0  # 所有的label的个数， 一般是70-90左右
    self.feature_cols = 0  # 一帧包含多少个特征值


def read_data_config(config_path):
  """读取数据配置文件.

  Args:
    config_path: 配置文件路径.

  Returns:
    (一个样本最大的占多少个float32,
    特征部分占多少个float32的个数,
    label最大占多少个float32的个数,
    样本的个数,
    label的类型的个数,
    一帧数据的特征数个数)

  """
  config = list()

  with open(config_path) as config_file:
    for line in config_file:
      config.append(int(line.strip()))

  train_data_info = TrainDataInfo()
  train_data_info.example_max_length = config[0]
  train_data_info.example_feature_max_length = config[1]
  train_data_info.example_label_max_length = config[2]
  train_data_info.example_number = config[3]
  train_data_info.example_label_count = config[4]
  train_data_info.feature_cols = config[5]
  return train_data_info


def get_dev_data(dev_data_config, batch_size):
  """获取开发集数据.

  Args:
    dev_data_config: 开发集数据配置.
    batch_size: batch size.

  Returns:
    开发集数据.

  """
  eval_file = FLAGS.data_dir + FLAGS.eval_file
  if not eval_file:
    raise ValueError('Please supply a eval data_file')

  if not tf.gfile.Exists(eval_file):
    raise ValueError('Failed to find file: ' + eval_file)
  file_names = [eval_file]

  filename_queue = tf.train.string_input_producer(file_names)
  record_bytes = dev_data_config.example_max_length * 4
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  key, value = reader.read(filename_queue)
  value_n = tf.decode_raw(value, tf.float32)
  value_n = tf.reshape(value_n, [dev_data_config.example_max_length])

  min_queue_examples = 300

  inputs = tf.train.shuffle_batch([value_n], batch_size=batch_size,
                                  num_threads=16,
                                  capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples)

  return inputs


def distort_inputs(data_config, batch_size):
  """乱序读入训练数据.

  Args:
    data_config: 数据相关的参数. 结构为:
    (一个样本最大的占多少个float32, 特征部分占多少个float32的个数,
    label最大占多少个float32的个数, 样本的个数)
    batch_size: 批大小.

  Returns:
    读取数据的op.

  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  data_dir = FLAGS.data_dir
  train_files = FLAGS.train_files

  # 组成读入的文件列表
  train_file_list = train_files.strip().split(",")
  train_file_number = len(train_file_list)
  data_paths = list()

  for train_file in train_file_list:
    data_paths.append(path.join(data_dir, train_file))

  for f in data_paths:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(data_paths)
  record_bytes = data_config.example_max_length * 4
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

  # 这里是读取tf.float32的, 如果二进制写入的格式变了, 下面需要对应的修改.
  key, value = reader.read(filename_queue)
  value_n = tf.decode_raw(value, tf.float32)
  value_n = tf.reshape(value_n, [data_config.example_max_length])

  min_queue_examples = 300

  return tf.train.shuffle_batch([value_n], batch_size=batch_size,
                                num_threads=4,
                                capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples)


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


def inference(num_hidden, num_layers, num_classes, feature_data, seq_len,
              batch_size):
  """双向LSTM网络结构，计算向前结果.

  Args:
    num_hidden: 每层宽度.
    num_layers: 层数.
    num_classes: 分类的数量.
    feature_data: 特征数据, 用于训练. 规模为[batch_size, seq_length, layer_width]
    seq_len: 序列长度.
    batch_size: batch size.

  Returns:
    在目标函数之前的模型流程.

  """
  with tf.device('/cpu:0'):
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden,  use_peepholes=True,
                                      initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                                      state_is_tuple=True)
    stack_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers,
                                           state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True,
                                      initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),
                                      state_is_tuple=True)
    stack_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers,
                                           state_is_tuple=True)

    W = tf.get_variable("weights", [num_hidden * 2, num_classes],
                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                 stddev=0.1))
    b = tf.get_variable("biases", [num_classes],
                        initializer=tf.constant_initializer(0.0))
  outputs, statuses = tf.nn.bidirectional_dynamic_rnn(stack_fw, stack_bw,
                                                      feature_data, seq_len,
                                                      dtype=tf.float32)

  # 由于outputs是由双向LSTM生成的, 因此有两个输出, 一个是正向的输出, 一个是反向的输出. 这
  # 一步将两个LSTM输出的结果合并成一个.
  outputs = tf.concat(2, outputs, name="output_concat")

  # 做一个全连接映射到label_num个数的输出
  outputs = tf.reshape(outputs, [-1, num_hidden * 2])
  logits = tf.add(tf.matmul(outputs, W), b, name="logits_add")
  logits = tf.reshape(logits, [batch_size, -1, num_classes])
  ctc_input = tf.transpose(logits, (1, 0, 2))
  return ctc_input


def rnn(inputs, data_config, batch_size):
  """构建训练模型的RNN网络.

  Args:
    inputs: 特征数据, 第一维是batch size, 后面记录了各种标签数据.
    data_config: 数据相关配置.

  Returns:
    返回三元组, 分别是
    ctc_input: ctc的输入数据.
    targets: 训练数据的正确标签, 以稀疏矩阵的形式表示.
    seq_len:  每个batch的长度.

  """
  num_layers = FLAGS.num_layers
  num_hidden = FLAGS.num_hidden
  feature_cols = data_config.feature_cols
  num_classes = data_config.example_label_count + 1  # label的个数+1

  # 处理输入的数据，提取特征数据
  seq_len = tf.to_int32(tf.div(inputs[:, 0], feature_cols))
  label_len = tf.to_int32(inputs[:, 1])

  feature_slice_length = int(
    data_config.example_feature_max_length / feature_cols) * feature_cols
  feature_area = tf.slice(inputs, [0, 2],
                          [batch_size, feature_slice_length])

  feature_area = tf.reshape(feature_area, [batch_size, -1, feature_cols],
                            name="feature_area")

  label_area = tf.slice(inputs, [0, 2 + data_config.example_feature_max_length],
                        [batch_size, data_config.example_label_max_length],
                        name="label_area")

  ctc_input = inference(num_hidden, num_layers, num_classes, feature_area,
                        seq_len, batch_size)

  # label的所有值，将label的数据转换成SparseTensor的形式
  # TODO: 可以简化, 用batch中的样例id pack label本身即可.
  label_value = tf.slice(label_area, [0, 0], [1, label_len[0]])

  for i in range(1, batch_size):
    v1 = tf.slice(label_area, [i, 0], [1, label_len[i]])
    label_value = tf.concat(1, [label_value, v1])

  label_value = tf.reshape(label_value, [-1])
  indices = tf.range(data_config.example_label_max_length)

  indice1 = tf.fill([label_len[0]], 0)
  indice2 = tf.slice(indices, [0], [label_len[0]])
  indices_array = tf.pack([indice1, indice2], axis=1)

  for i in range(1, batch_size):
    indice1 = tf.fill([label_len[i]], i)
    indice2 = tf.slice(indices, [0], [label_len[i]])
    temp_array = tf.pack([indice1, indice2], axis=1)
    indices_array = tf.concat(0, [indices_array, temp_array])

  sparse_shape = [batch_size, data_config.example_label_max_length]
  sparse_shape = tf.to_int64(sparse_shape)
  indices_array = tf.to_int64(indices_array)
  label_value = tf.to_int32(label_value)
  targets = tf.SparseTensor(indices_array, label_value, sparse_shape)
  return ctc_input, targets, seq_len


def loss_multi(logits, targets, seq_len):
  loss = tf.nn.ctc_loss(logits, targets, seq_len)
  cost = tf.reduce_mean(loss)
  tf.add_to_collection('losses', cost)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == "__main__":
  pass
