# coding=utf-8

import codecs
import os

# 分隔文件，把一个大文件分隔为几个小文件
'''
  input_file: 输入文件
  output_dir： 输出的目录
  output_filename： 输出的目录下的文件的前缀
  number： 分隔成几份
'''


def split_file(input_file, output_dir, output_filename, number):
  # 建立文件夹
  os.makedirs(output_dir)

  all_number = 0
  with codecs.open(input_file, 'r', 'utf-8') as input_fr:
    while True:
      all_number = all_number + 1
      if input_fr.readline() == '':
        break

  number_perfile = int(all_number / number) + 1

  file_id = 0
  write_lines = 0
  fw = codecs.open(output_dir + "/" + output_filename + "." + str(file_id), 'w',
                   'utf-8')
  with codecs.open(input_file, 'r', 'utf-8') as input_fr:
    while True:
      write_lines = write_lines + 1
      line = input_fr.readline()
      if line == '':
        break
      fw.write(line)
      if write_lines == number_perfile:
        fw.close()
        write_lines = 0
        file_id = file_id + 1
        fw = codecs.open(
          output_dir + "/" + output_filename + "." + str(file_id), 'w', 'utf-8')

  fw.close()


if __name__ == "__main__":
  split_file("../data/train.eesen", "../data/train/", "train", 5)
