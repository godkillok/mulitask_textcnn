#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : chenzhikuo
# @Time :  2019/5/27
# @Filename : unt.py
import random

target_num = 8
new_train = {}
train1_label_dict = {'health':[2,3,4,5,6]}
for key in train1_label_dict:
    if target_num // len(train1_label_dict[key]) == 1:
        # temp_list = train1_label_dict[key]
        train1_label_dict[key].extend(random.sample(train1_label_dict[key], target_num-len(train1_label_dict[key]))).extend([76,32])
        new_train[key] = train1_label_dict[key]
print(new_train)