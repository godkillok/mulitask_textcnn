#!/usr/bin/env python
import tensorflow as tf
import os
import re
import numpy as np
import random

import random
import json
import time

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/data/tanggp/company_data/", "Directory containing the dataset")
flags.DEFINE_string("pad_word", '<pad>', "used for pad sentence")
flags.DEFINE_string("OOV", 'v_oov', "used for pad sentence")
flags.DEFINE_string("path_vocab", "/data/tanggp/company_data/words.txt", "used for word index")

FLAGS = flags.FLAGS

sentence_max_len = 200
pad_word = FLAGS.pad_word
OOV =FLAGS.OOV
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

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\`", "\'", text)
    text = text.strip().lower()
    return text

def parse_line_dict(record,vocab_dict,author_dict,label_dict,categories_dict):

    fields = record.strip().split(',')
    text = clean_str(fields[2])
    tokens = text.split()
    tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]
    text = [vocab_dict.get(r,OOV) for r in tokens]
    label=int(fields[0])

    return [text, label,1,1]


def per_thouds_lines_dict(result_lines, path_text, count,flag_name=''):
    tf_lines = []


    for rl_num,rl in enumerate(result_lines):
        text=rl[0]
        label=rl[1]
        author=rl[2]
        categories=rl[3]
        if len(text) >= sentence_max_len:
            text = text[0: sentence_max_len]
        else:
            text += [pad_word] * (sentence_max_len - len(text))
        g={"text":text,"label":label,"author":author,"categories":categories}
        tf_lines.append(g)
        # if rl_num>1 and rl_num%10000==0:
        #     flag_name=str(rl_num)
        #     write_tfrecords(tf_lines, path_text, count)
        #     tf_lines = []
    if len(tf_lines)>0:
        flag_name = str(rl_num)
        print(len(tf_lines))
        #time.sleep(5)
        write_tfrecords(tf_lines, path_text, count)
            # tf_lines=[]
def ini():
    global pad_word,OOV

    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab_dict = {l.strip(): (i) for i, l in enumerate(lines)}
        pad_word=vocab_dict.get(pad_word)
        OOV=vocab_dict.get(OOV)
        print("pad_word {},OOV {}".format(pad_word,OOV))

    label_dict ={}
    author_dict={}
    categories_dict={}
    return vocab_dict,author_dict,label_dict,categories_dict

def generate_tf_dic(path_text,vocab_dict,author_dict,label_dict,categories_dict):


    result_lines = []
    count = 0
    with open(path_text, 'r', encoding='utf8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            count+=1
            result_lines.append(parse_line_dict(line,vocab_dict,author_dict,label_dict,categories_dict))
            if count>0 and count % 50000 == 0:
                print(count)
                per_thouds_lines_dict(result_lines, path_text, count)
                result_lines = []
        if len(result_lines)>0:
            per_thouds_lines_dict(result_lines, path_text, count)


def write_tfrecords(tf_lines, path_text, count):
    (root_path, output_filename) = os.path.split(path_text)
    output_filename = output_filename.split('.')[0]
    output_filename='author_text_cnn_'+output_filename
    output_file = output_filename + '_' + str(count)+ '.tfrecords'

    print("Start to convert {} to {}".format(len(tf_lines), os.path.join(root_path, output_file)))

    writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
    random.shuffle(tf_lines)
    num = 0
    for i,data in enumerate(tf_lines):
        if i==0:
            print("Start to convert {}".format(data))
        text = data["text"]
        label = data["label"]
        author=data["author"]
        categories=data["categories"]
        example = tf.train.Example(features=tf.train.Features(feature={
            'text': feature_auto(list(text)),
            'label': feature_auto(int(label)),
            'author': feature_auto(int(author)),
            "categories": feature_auto(int(categories)),
        }))

        writer.write(example.SerializeToString())
        num += 1
        # if num % 1000 == 0:
        #     output_file = output_filename + '_' + str(count) + '_' + str(num)+'_' + flag_name + '.tfrecords'
        #     writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
        #     print("Start convert to {}".format(output_file))


def main():
    vocab_dict, author_dict, label_dict, categories_dict= ini()
    generate_tf_dic(os.path.join(FLAGS.data_dir, 'train.csv'),vocab_dict,author_dict,label_dict,categories_dict)
    generate_tf_dic(os.path.join(FLAGS.data_dir, 'test.csv'),vocab_dict,author_dict,label_dict,categories_dict)
    #generate_tf_dic(os.path.join(FLAGS.data_dir, 'txt_valid'), vocab_dict, author_dict, label_dict, categories_dict)
    # s3_input = FLAGS.data_dir
    # for root, dirs, files in os.walk(s3_input):
    #     for file in files:
    #         # if file.endswith("ain_set.csv"):
    #         #     print('start to process file {}'.format(file))
    #         generate_tf_dic(os.path.join(root, file))
    # os.system('cd {}'.format(s3_input))
    # os.system('find . -name "*" -type f -size 0c | xargs -n 1 rm -f')


if __name__ == "__main__":
    main()
