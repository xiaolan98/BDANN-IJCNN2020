# encoding=utf-8
try:
    import cPickle as pickle
except ImportError:
    import pickle
import random
from random import *
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
import math
from types import *
#import jieba
import os.path
#from googletrans import Translator
#from langdetect import detect
#from langdetect.lang_detect_exception import LangDetectException


def stopwordslist(filepath='../Data/weibo/stop_words.txt'):
    stopwords = {}
    for line in open(filepath, 'r').readlines():
        # line = unicode(line, "utf-8").strip()
        line = line.strip()
        stopwords[line] = 1
    # stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
#
def read_image():
    image_list = {}
    # file_list = ['../Data/twitter/images_all/']
    file_list = ['../Data/twitter/images_train/', '../Data/twitter/images_validation/', '../Data/twitter/images_test/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                # im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    # print("image names are " + str(image_list.keys()))
    return image_list


def write_txt(data):
    f = open("../Data/weibo/top_n_data.txt", 'wb')
    for line in data:
        for l in line:
            f.write(l + "\n")
        f.write("\n")
        f.write("\n")
    f.close()


def clean_text(text):

    try:
        text = text.decode('utf-8').lower()
    except:
        text = text.encode('utf-8').decode('utf-8').lower()
    text = re.sub(u"\u2019|\u2018", "\'", text)
    text = re.sub(u"\u201c|\u201d", "\"", text)
    text = re.sub(u"[\u2000-\u206F]", " ", text)
    text = re.sub(u"[\u20A0-\u20CF]", " ", text)
    text = re.sub(u"[\u2100-\u214F]", " ", text)
    text = re.sub(r"http:\ ", "http:", text)
    text = re.sub(r"http[s]?:[^\ ]+", " ", text)
    text = re.sub(r"&gt;", " ", text)
    text = re.sub(r"&lt;", " ", text)
    text = re.sub(r"&quot;", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"#\ ", "#", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"[\(\)\[\]\{\}]", r" ", text)
    text = re.sub(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+',
    r" ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " had ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"#", " #", text)
    text = re.sub(r"\@", " \@", text)
    text = re.sub(r"[\!\?\.\,\+\-\$\%\^\>\<\=\:\;\*\(\)\{\}\[\]\/\~\&\'\|]", " ", text)
    text = text.strip()

    return text


def get_text_dict(flag):
    pre_path = "../Data/twitter/"
    if flag == 'train':
        text_dict = pickle.load(open(pre_path + 'cleaned_train_text.pkl', 'rb'))
    elif flag == 'validate':
        text_dict = pickle.load(open(pre_path + 'cleaned_train_text.pkl', 'rb'))
    elif flag == 'test':
        text_dict = pickle.load(open(pre_path + 'cleaned_test_text.pkl', 'rb'))
    else:
        text_dict = {}
        print('Error of getting text dict, because of the wrong flag')
    return text_dict


def write_data(flag, image, text_only):
    text_dict = get_text_dict(flag)
    def read_post(flag):
        pre_path = "../Data/twitter/"
        if flag == "train":
            texts = open(pre_path + 'train_posts.txt', 'r').readlines()
            images = [image.split('.')[0] for image in os.listdir(pre_path + 'images_train')]
        elif flag == "validate":
            texts = open(pre_path + 'train_posts.txt', 'r').readlines()
            images = [image.split('.')[0] for image in os.listdir(pre_path + 'images_validation')]

        elif flag == "test":
            texts = open(pre_path + 'test_posts.txt', 'r').readlines()
            images = [image.split('.')[0] for image in os.listdir(pre_path + 'images_test')]
        else:
            print('Error')
            return
        # cleaned_text = pickle.load(open(pre_path + 'cleaned_text.pkl', 'rb'))

        post_content = []
        data = []
        column = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        map_id = {}
        for i, line in enumerate(texts):
            if i == 0: continue
            line_data = []
            if flag == 'train' or flag == 'validate':
                image_id = line.split('\t')[3].lower().split(',')
            else:
                image_id = line.split('\t')[4].lower().split(',')
            continue_flag = False
            for image in image_id:
                if image in images:
                    image_id = image
                    continue_flag = True
                    break
            if not continue_flag:
                continue
            # text = cleaned_text[post_id]
            # text = clean_text(line.split('\t')[1])
            post_id = line.split('\t')[0]
            text = text_dict[post_id]

            if len(text.split(' ')) <= 2:
                # print(text)
                continue
            label = 0 if line.split('\t')[-1].strip() == 'real' else 1
            event_name = re.sub(u'fake', '', image_id)
            event_name = re.sub(u'real', '', event_name)
            event_name = re.sub(u'[0-9_]', '', event_name)
            if event_name not in map_id:
                map_id[event_name] = len(map_id)
                event = map_id[event_name]
            else:
                event = map_id[event_name]
            line_data.append(post_id)
            line_data.append(image_id)
            post_content.append(text)
            line_data.append(text)
            line_data.append([])
            line_data.append(label)
            line_data.append(event)

            data.append(line_data)

        data_df = pd.DataFrame(np.array(data), columns=column)

        return post_content, data_df, len(map_id)

    post_content, post, event_num = read_post(flag)
    print("Original " + flag + " post length is " + str(len(post_content)))
    print("Original " + flag + " data frame is " + str(post.shape))
    print("Original " + flag + " Event number is " + str(event_num))

    def paired(text_only=False):
        ordered_image = []
        ordered_text = []
        ordered_post = []
        ordered_event = []
        label = []
        post_id = []
        image_id_list = []

        image_id = ""
        for i, id in enumerate(post['post_id']):
            image_id = post.iloc[i]['image_id']

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_text.append(post.iloc[i]['original_post'])
                ordered_post.append(post.iloc[i]['post_text'])
                ordered_event.append(post.iloc[i]['event_label'])
                post_id.append(id)

                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=np.int)
        ordered_event = np.array(ordered_event, dtype=np.int)

        print("Label number is " + str(len(label)))
        print("Rummor number is " + str(sum(label)))
        print("Non rummor is " + str(len(label) - sum(label)))

        data = {"post_text": np.array(ordered_post),
                "original_post": np.array(ordered_text),
                "image": ordered_image, "social_feature": [],
                "label": np.array(label), \
                "event_label": ordered_event, "post_id": np.array(post_id),
                "image_id": image_id_list}
        # print(data['image'][0])

        print("data size is " + str(len(data["post_text"])))

        return data

    paired_data = paired(text_only)

    print("paired post length is " + str(len(paired_data["post_text"])))
    print("paried data has " + str(len(paired_data)) + " dimension")
    return paired_data, event_num


def load_data(train, validate, test):
    vocab = defaultdict(float)
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


def get_data(text_only):
    # text_only = False

    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = read_image()

    train_data, event_num = write_data("train", image_list, text_only)
    # valiate_data, _ = write_data("validate", image_list, text_only)
    # test_data, _ = write_data("test", image_list, text_only)
    valiate_data, _ = write_data("validate", image_list, text_only)
    test_data, _ = write_data("validate", image_list, text_only)

    return train_data, valiate_data, test_data, event_num
