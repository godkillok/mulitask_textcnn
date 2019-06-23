#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from logger import get_logger
import json
import random

log_file_name = os.path.basename(__file__).split('.', 1)[0] + '.log'
# Save params
# 当日志文件大小小于1M时，则以追加模式写
if os.path.exists(log_file_name) is False or os.path.getsize(log_file_name) / 1024 / 1024 < 5:
    logger = get_logger(log_file_name, mode='a')
else:
    # 否则删除以前的日志
    logger = get_logger(log_file_name)

target_num = 27458
train1_label_dict = {}
train2_label_dict = {}

with open('./google_item_mapping_level_0_replace', 'r', encoding='utf8') as f:
    for line in f:
        json1 = json.loads(line)
        label = json1['label']
        if label not in train1_label_dict:
            train1_label_dict[label] = []
        train1_label_dict[label].append(json1)

new_train = {}

with open('./8M_filted.json', 'r', encoding='utf8') as f:
    for line in f:
        json1 = json.loads(line)
        label = json1['label']
        if label not in train2_label_dict:
            train2_label_dict[label] = []
        train2_label_dict[label].append(json1)

for key in train1_label_dict:
    logger.info('{}, {}'.format(len(train1_label_dict[key]), key))

logger.info('\n')
for key in train2_label_dict:
    logger.info('{}, {}'.format(len(train2_label_dict[key]), key))

logger.info('\n')
for key in train1_label_dict:
    if len(train1_label_dict[key]) >= target_num:
        new_train[key] = random.sample(train1_label_dict[key], target_num)
    else:
        if key not in train2_label_dict:
            if target_num//len(train1_label_dict[key]) == 3:
                train1_label_dict[key].extend(random.sample(train1_label_dict[key] * 3, target_num - len(train1_label_dict[key])))

            elif target_num // len(train1_label_dict[key]) == 2:
                train1_label_dict[key].extend(random.sample(train1_label_dict[key]*2, target_num - len(train1_label_dict[key])))

            elif target_num // len(train1_label_dict[key]) == 1:
                train1_label_dict[key].extend(random.sample(train1_label_dict[key], target_num-len(train1_label_dict[key])))

        else:
            if len(train1_label_dict[key]) + len(train2_label_dict[key]) >= target_num:
                train1_label_dict[key].extend(random.sample(train2_label_dict[key], target_num-len(train1_label_dict[key])))
            else:
                train1_label_dict[key].extend(train2_label_dict[key])
                if target_num // len(train2_label_dict[key]) == 1:
                    train1_label_dict[key].extend(random.sample(train2_label_dict[key], target_num-len(train1_label_dict[key])))
        new_train[key] = train1_label_dict[key]

for key in new_train:
    try:
        logger.info('{}, {}'.format(len(new_train[key]), key))
    except Exception as e:
        logger.info('{}, {}'.format(key, e))

logger.info('\n')
with open('./all_train.json', 'w', encoding='utf8') as f_train:
    for key in new_train:
        for content in new_train[key]:
            f_train.write(json.dumps(content) + '\n')





