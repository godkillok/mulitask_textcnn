#!/usr/bin/env python
import tensorflow as tf
import os
import re
import numpy as np
import random

import random
import json
from common_tool import per_line
flags = tf.app.flags
flags.DEFINE_string("data_dir", "/data/tanggp/youtube8m/", "Directory containing the dataset")
flags.DEFINE_string("pad_word", "0", "used for pad sentence")
flags.DEFINE_string("path_vocab", "/data/tanggp/youtube8m/textcnn_words.txt", "used for word index")
flags.DEFINE_string("path_author",  os.path.join("/data/tanggp/youtube8m/", 'textcnn_author_sort'))
flags.DEFINE_string("path_label",  os.path.join("/data/tanggp/youtube8m/", 'textcnn_label_sort'))
FLAGS = flags.FLAGS

sentence_max_len = 200
pad_word = FLAGS.pad_word

label_class=[]
author_calss=[]
def feature_auto(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, list):
        if isinstance(value[0],int):
            try:
                tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            except:
                print(value)
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        elif isinstance(value[0],float):
            try:
                tf.train.Feature(int64_list=tf.train.FloatList(value=value))
            except:
                print(value)
            return tf.train.Feature(int64_list=tf.train.FloatList(value=value))
        else:
            print("list type error")

    elif isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    elif isinstance(value, float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def parse_line_dict(record,vocab_dict,author_dict,label_dict):
    tokens=per_line(record)
    tokens=tokens.split()
    text = [vocab_dict.get(r) for r in tokens if vocab_dict.get(r, '-99') != '-99']
    record=json.loads(record)
    label=record.get("label")
    author=record.get("author")

    return [text, label_dict.get(label),author_dict.get(author)]


def per_thouds_lines_dict(result_lines, path_text, count,flag_name=''):
    tf_lines = []

    rl_num=0
    for rl in result_lines:
        text=rl[0]
        label=rl[1]
        author=rl[2]
        if len(text) >= sentence_max_len:
            text = text[0: sentence_max_len]
        else:
            text += [pad_word] * (sentence_max_len - len(text))
        g={"text":text,"label":label,"author":author}
        tf_lines.append(g)
        if rl_num%3000==0:
            flag_name=str(rl_num)
            write_tfrecords(tf_lines, path_text, count)
            tf_lines = []
    if len(tf_lines)>0:
        flag_name = str(rl_num)
        write_tfrecords(tf_lines, path_text, count)
            # tf_lines=[]

def generate_tf_dic(path_text):
    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {l.strip(): i for i, l in enumerate(lines)}

    with open(FLAGS.path_author, 'r', encoding='utf8') as f:
        lines = f.readlines()
        author_dict = {l.strip(): i for i, l in enumerate(lines)}

    with open(FLAGS.path_label, 'r', encoding='utf8') as f:
        lines = f.readlines()
        label_dict = {l.strip(): i for i, l in enumerate(lines)}

    result_lines = []
    count = 0
    with open(path_text, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            result_lines.append(parse_line_dict(line,author_dict,label_dict))
            if count % 50000 == 0:
                print(count)
                per_thouds_lines_dict(result_lines, path_text, count)
                result_lines = []
        if len(result_lines)>0:
            per_thouds_lines_dict(result_lines, path_text, count)


def write_tfrecords(tf_lines, path_text, count):
    (root_path, output_filename) = os.path.split(path_text)
    output_filename = output_filename.split('.')[0]
    output_filename+='text_cnn_'
    output_file = output_filename + '_' + str(count)+ '.tfrecords'

    print("Start to convert {} to {}".format(len(tf_lines), os.path.join(root_path, output_file)))

    writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
    random.shuffle(tf_lines)
    num = 0
    for data in tf_lines:
        text = data["text"]
        label = data["label"]
        author=data["author"]
        example = tf.train.Example(features=tf.train.Features(feature={
            'text': feature_auto(list(text)),
            'label': feature_auto(int(label)),
            'author': feature_auto(int(author))
        }))

        writer.write(example.SerializeToString())
        num += 1
        # if num % 1000 == 0:
        #     output_file = output_filename + '_' + str(count) + '_' + str(num)+'_' + flag_name + '.tfrecords'
        #     writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
        #     print("Start convert to {}".format(output_file))


def main():
    s3_input = FLAGS.data_dir
    for root, dirs, files in os.walk(s3_input):
        for file in files:
            # if file.endswith("ain_set.csv"):
            #     print('start to process file {}'.format(file))
            generate_tf_dic(os.path.join(root, file))
    # os.system('cd {}'.format(s3_input))
    # os.system('find . -name "*" -type f -size 0c | xargs -n 1 rm -f')


if __name__ == "__main__":
    main()
