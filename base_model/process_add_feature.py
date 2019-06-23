#!/usr/bin/python3
# -*- coding: utf-8 -*-


import argparse
import os
import time
from cnn_data_process import CnnDataProcessor
from logger import get_logger
import pickle
log_file_name = os.path.basename(__file__).split('.', 1)[0] + '.log'
logger = get_logger(log_file_name)

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=4, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--max_length', default=40, help="Maximum length allowed in the dataset", type=int)
parser.add_argument("--max_keyword_length", default=5, help="The maximum length of keyword.")
parser.add_argument('--train_dir', default='./train_dir', help="File containing the postive file")
parser.add_argument('--valid_dir', default='./valid_dir', help="File containing the negtive file")

parser.add_argument('--data_dir', default='./baseline_data/', help="Directory containing the dataset")
parser.add_argument('--target_train_dir', default='train', help="File saving the train file")
parser.add_argument('--target_valid_dir', default='valid', help="File saving the valid file")

if __name__ == '__main__':
    start  = time.time()
    args = parser.parse_args()
    process = CnnDataProcessor()
    train_sentence_list, train_label_list = process.get_examples(args.train_dir, 'train')
    process.create_label_dict(train_label_list) # 构建字典
    train_word_count, train_author_count, train_category_count, train_keyword_count, train_max_length = process.count_word_num(train_sentence_list)
    average_len = process.count_average_length(train_sentence_list)
    logger.info('average length is {}'.format(average_len))

    valid_sentence_list, valid_label_list = process.get_examples(args.valid_dir, 'valid')

    valid_word_count, valid_author_count, valid_category_count,valid_keyword_count, valid_max_length = process.count_word_num(valid_sentence_list)

    process.create_word_dict(train_word_count, args.min_count_word)
    process.update_word_dict(valid_word_count)
    process.append_unk_pad_dict()
    
    process.create_author_dict(train_author_count)
    process.update_author_dict(valid_author_count)
    process.append_author_unk_dict()

    process.create_category_dict(train_category_count)
    process.update_category_dict(valid_category_count)
    process.append_category_unk_dict()

    process.create_keyword_dict(train_keyword_count)
    process.update_keyword_dict(valid_keyword_count)
    process.append_keyword_pad_dict()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    words_list = process.get_words()
    process.save_vocab_to_txt_file(words_list, os.path.join(args.data_dir, 'words.txt'))

    id_word = process.get_id_words()

    max_length = min(args.max_length, max(train_max_length, valid_max_length))


    label_list= process.get_labels()
    print(label_list,len(label_list), 'DA')
    process.append_label_pad_dict()  # 添加label做特征
    label_id = process.label_id_dict
    with open(args.data_dir + '/label2id.pkl', 'wb') as w:
        pickle.dump(label_id, w)
    print(len(label_list))
    train_target_dir = args.data_dir + args.target_train_dir
    process.convert2tfrecord(train_sentence_list, train_label_list, train_target_dir, max_length, args.max_keyword_length)

    valid_target_dir = args.data_dir + args.target_valid_dir
    process.convert2tfrecord(valid_sentence_list, valid_label_list, valid_target_dir, max_length, args.max_keyword_length)

    sizes = {
        'train_size': len(train_sentence_list),
        'valid_size': len(valid_sentence_list),
        'vocab_size': len(words_list),
        'author_size': len(process.author_id_dict),
        'category_size': len(process.category_id_dict),
        'keyword_size': len(process.keyword_id_dict),
        'label_size': len(label_list),
        'id_word': id_word,
        'max_length': max_length,
        'max_keyword_length': args.max_keyword_length,
    }

    process.save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))
    elapsed = (time.time() - start)/60
    logger.info("The total program takes {} minutes".format(elapsed))

