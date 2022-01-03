# coding=utf-8
from __future__ import unicode_literals
import gc
import io
import json
from model.FANCE import FANCE
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
from keras.utils.np_utils import to_categorical
import operator
import codecs
import os
from keras import backend as K
import argparse
import random
import sys
# when gpu>1 choose a gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def numpy_vector_str(n):
    s = ""
    for i in range(len(n)-1):
        s += str(n[i])+" "
    s += str(n[len(n)-1])
    return s



if __name__ == '__main__':
    # dataset used for training
    # platform = 'gossipcop'
    platform = 'politifact'
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--sent_len', '-sl', help='sentence_length')
    parser.add_argument('--sent_cout', '-sc', help='sentence_count')
    parser.add_argument('--comt_len', '-cl', help='comment_length')
    parser.add_argument('--comt_cout', '-cc', help='comment_count')
    parser.add_argument('--user_comt', '-uc', help='user_comment')
    parser.add_argument('--platform', '-pf', help='platform')
    parser.add_argument('--learning', '-lr', help='learning_rate')
    parser.add_argument('--bat_size', '-bs', help='batch_size')
    parser.add_argument('--explain_file_name', '-efn', help='explain_file_name')
    args = parser.parse_args()

    if args.platform:
        platform = args.platform

    contents_file = io.open('data/' + platform + '_content.txt', "r")
    VALIDATION_SPLIT = 0.25
    contents = []
    labels = []
    texts = []
    ids = []
    for i in contents_file:
        i = i.strip("\n")
        e = i.split("$:$:")
        ids.append(str(e[0]))
        labels.append(str(e[1]))

        tmp_contents_vector = np.empty((0, 200))
        for each in e[2:]:
            v = each.split(" ")
            v = [float(x) for x in v]
            temp = np.empty((1, 200))
            for i in range(200):
                temp[0][i] = v[i]
            tmp_contents_vector = np.append(tmp_contents_vector, temp, axis=0)
        contents.append(tmp_contents_vector)

    labels = np.asarray(labels)
    labels = to_categorical(labels)
    print("content finished!")

    sentiment_file = open('data/' + platform + '_sentiment.txt', "r")
    sentiment_vec = []
    for i in sentiment_file:
        i = i.strip("\n")
        e = i.split("$:$:")
        v = e[1].split(" ")
        v = [float(x) for x in v]
        temp = np.empty((1, 200))
        for i in range(200):
            temp[0][i] = v[i]
        sentiment_vec.append(temp)
    print("sentiment finished!")

    comments_file = io.open('data/' + platform + '_comment.txt', "r")
    comments = []
    for i in comments_file:
        i = i.strip("\n")
        e = i.split("$:$:")
        tmp_comments_vector = np.empty((0, 200))
        for each in e[2:]:
            v = each.split(" ")
            v = [float(x) for x in v]
            temp = np.empty((1, 200))
            for i in range(200):
                temp[0][i] = v[i]
            tmp_comments_vector = np.append(tmp_comments_vector, temp, axis=0)
        comments.append(tmp_comments_vector)
    print("comments finished!")

    users_file = io.open('data/' + platform + '_user.txt', "r")
    users = []
    for i in users_file:
        i = i.strip("\n")
        e = i.split("$:$:")
        tmp_users_vector = np.empty((0, 200))
        for each in e[2:]:
            v = each.split(" ")
            v = [float(x) for x in v]
            temp = np.empty((1, 200))
            for i in range(200):
                temp[0][i] = v[i]
            tmp_users_vector = np.append(tmp_users_vector, temp, axis=0)
        users.append(tmp_users_vector)
    print("users finished!")
        


    id_train, id_test, x_train, x_val, x_train_senti, x_val_senti, y_train, y_val, c_train, c_val, cid_train, cid_val = train_test_split(ids,
                                                                                                             contents,
                                                                                                             sentiment_vec,
                                                                                                             labels,
                                                                                                             comments,
                                                                                                             users,
                                                                                                             test_size=VALIDATION_SPLIT,
                                                                                                             random_state=42,
                                                                                                             stratify=labels)
    
    #Train and save the model
    batch_size = 20
    if args.sent_len:
        h = FANCE(platform, MAX_SENTENCE_LENGTH=int(args.sent_len), alter_params='sent_len')
    elif args.comt_len:
        h = FANCE(platform, MAX_COMS_LENGTH=int(args.comt_len), alter_params='comt_len')
    elif args.comt_cout:
        h = FANCE(platform, MAX_COMS_COUNT=int(args.comt_cout), alter_params='comt_cout')
    elif args.sent_cout:
        h = FANCE(platform, MAX_SENTENCE_COUNT=int(args.sent_cout), alter_params='sent_cout')
    elif args.user_comt:
        h = FANCE(platform, USER_COMS=int(args.user_comt), alter_params='user_comt')
    elif args.learning:
        h = FANCE(platform, lr= round(float(args.learning)/10000,6), alter_params='lr')
    elif args.bat_size:
        h = FANCE(platform, alter_params='batch_size')
        batch_size = int(args.bat_size)
    else:
        h = FANCE(platform, alter_params='null')
    
    h.train(x_train, x_train_senti, y_train, c_train, cid_train, cid_val, c_val, x_val, x_val_senti, y_val,
            batch_size=batch_size,
            epochs=30)


    # for explain
    
    # result = h.predict(x_val,c_val,cid_val)
    # print(result)
    # Get the attention weights for sentences in the news contents as well as comments
    # activation_maps = h.activation_maps(x_val, c_val, cid_val)
    
    #######
    # activation_maps = h.activation_maps(x_val, c_val, cid_val)
    # with codecs.open('./explain/'+str(platform)+'_results_'+str(args.explain_file_name)+'.txt', 'w', encoding='utf-8') as f:
    #     for i in range(len(activation_maps[0])):
    #             news_s_attention = ""
    #             for j in range(len(activation_maps[0][i])-1):
    #                 news_s_attention += str(round(float(activation_maps[0][i][j]),6)) + "::"
    #             news_s_attention += str(activation_maps[0][i][len(activation_maps[0][i])-1])
    #             f.write(str(id_test[i]) + '\t' + news_s_attention + '\n')
    #######

    # for i in range(len(activation_maps[1])):
    #     if len(activation_maps[1][i]) >= 30:
    #         with codecs.open('./explain/{}_result_comment_{}.txt'.format(platform, str(i)), 'w', encoding='utf-8') as f:
    #             for j in range(len(activation_maps[1][i])):
    #                 f.write(" ".join(activation_maps[1][i][j][0]) + '\t' + str(activation_maps[1][i][j][1]) + '\n')
    