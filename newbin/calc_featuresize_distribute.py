#!/usr/bin/env python
#coding=utf-8
#生成一个高斯密度函数

import codecs
import os

'''
  将eesen的特征文件归一到一个文件中， 方便tensorflow或者其他的框架读取
  输出后的文件是二进制格式：
'''

def format_data():
  TRAINFEATUREFILE="./eesen/exp/train_phn_l2_c120/train2.ark"
  TRAINLABELFILE="./eesen/exp/train_phn_l2_c120/labels.tr"
  
  CVFEATUREFILE="./eesen/exp/train_phn_l2_c120/cv2.ark"
  CVLABELFILE="./eesen/exp/train_phn_l2_c120/labels.cv"

  format_eesen_data(TRAINFEATUREFILE, TRAINLABELFILE, "../data/traindata/train.eesen","../data/traindata/train.maxsize", 10)
  format_eesen_data(CVFEATUREFILE, CVLABELFILE, "../data/traindata/cv.eesen", "../data/traindata/cv.maxsize",1)
  pass


def format_eesen_data(ark_file, label_file, output_file_prefix, max_size_output, output_file_number=1):

  feature_number = 120
 
  print("start format eesen feature data, %s...." % ark_file)

  if not os.path.exists(ark_file):
    print("%s not exist" % ark_file)
    return -1
  if not os.path.exists(label_file):
    print("%s not exist" % label_file)
    return -1

  label_dict={}

  #一行最多包含多少个值， 不足的用0补齐
  max_values_per_line = []

  max_line_number = 0


  #读取的的文件，样本的包含的特征数量，已经按照从小到大排序好了。 
  #先读取label的数据到内存， 后读取feature的数据去匹配
  with codecs.open(label_file, 'r', 'utf-8') as label_fr:
    while True:
      line = label_fr.readline()
      if line =='':
        break
      words = line.strip().split(' ')
      words_number = len(words)
      if words_number >=2:
        label_dict[words[0]] = words[1:]


  # 样本个数
  example_number = 0

  #最大的特征的个数
  max_feature_size = 0

  #最大的label的个数
  max_label_size = 0

  # 总共一行的最大的size 2+ max_feature_size + max_label_size
  max_total_size = 0

  countflag = 10000
  # 分成10份，计算每一份的max_values_per_line
  with codecs.open(ark_file, 'r', 'utf-8') as ark_fr:
    feature_size= 0
    example_id = 0
    count_num = 0
    while True:
      line = ark_fr.readline()
      if line=='':
        max_feature_size = feature_size
        break
      words = line.split()
      if len(words)==2 and words[1]=='[':
        key = words[0]
        label_value = label_dict[key]
        feature_size = 0
        continue
      if len(words) == feature_number:
        feature_size = feature_size + feature_number
        continue
      if len(words) == feature_number+1 and words[-1]==']':
        feature_size = feature_size + feature_number
        label_size   = len(label_value)
        
        count_num = count_num + 1
        if feature_size > countflag:
          print(countflag, count_num)
          count_num = 0 
          countflag = countflag + 10000
        if label_size > max_label_size:
          max_label_size = label_size

        example_number = example_number + 1

  #每个小文件的保存的样本的个数
  example_number_per_file = int(example_number / output_file_number)


  #加2 是因为前面2个数字保存feature的个数和label的个数
  max_total_size = max_feature_size + max_label_size + 2

  print("the size info:")
  print(max_total_size)
  print(max_feature_size)
  print(max_label_size)
  print(example_number)


  
  print("successed process eesen feature data")


if __name__ == "__main__":
  format_data()

