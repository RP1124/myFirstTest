#!/user/bin/python
# codeing=utf-8

import datetime
from app.model.POS_tagging_GLM.config import config
import numpy as np
import random

class sentence():
    def __init__(self):
        self.words = []
        self.tags = []

class dataSet(object):
    def __init__(self,filename):
        self.sentences = []
        self.name = filename

    def open_file(self):
        self.file = open(self.name, mode='r', encoding='utf-8')

    def close_file(self):
        self.file.close()

    def read_data(self,sentenceLen):
        sentenceCount = 0 # 给句子计数
        wordCount = 0 # 给词计数
        sen = sentence()
        for s in self.file:
            if(s == '\n'):
                self.sentences.append(sen)
                sentenceCount += 1
                sen = sentence()
                if(sentenceLen != -1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')
            word = list_s[1]
            tag = list_s[3]
            sen.words.append(word)
            sen.tags.append(tag)
            wordCount += 1
        print(self.name + " contains " + str(sentenceCount) + " sentences ")
        print(self.name + " contains " + str(wordCount) + " words ")

    def shuffle(self):
        random(self.sentences)



class global_linear_model(object):
    def __init__(self, train_data, dev_data, test_data):
        self.train_data = dataSet(train_data) if train_data != None else None
        self.dev_data = dataSet(dev_data) if dev_data != None else None
        # self.test_data = dataSet(test_data) if test_data != None else None
        # 读取数据
        self.train_data.open_file()
        self.train_data.read_data(-1)
        self.train_data.close_file()
        self.dev_data.open_file()
        self.dev_data.read_data(-1)
        self.dev_data.close_file()
        # 属性：特征，权重，v，标签-索引词典，索引-标签词典，标签列表，开始标志
        self.features = {}
        self.w = []
        self.v = []
        self.tag_id = {}
        self.id_tag = {}
        self.tagsList = []
        self.BOS = "BOS"
        self.EOS = "EOS"

    def create_bi_feature(self, pre_tag):
        return ['01:'+ pre_tag]

    def create_uni_feature(self, sen, pos):
        template = []
        cur_word = sen.words[pos]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if pos == 0:
            pre_word = '##'
            pre_word_last_char = "#"
        else:
            pre_word = sen.words[pos-1]
            pre_word_last_char = pre_word[-1]

        if pos == len(sen.words)-1:
            next_word = "$$"
            next_word_first_char = "$"
        else:
            next_word = sen.words[pos+1]
            next_word_first_char = next_word[0]

        template.append("02:" + cur_word)
        template.append("03:" + pre_word)
        template.append("04:" + next_word)
        template.append("05:" + cur_word + "*" + pre_word_last_char)
        template.append("06:" + cur_word + "*" + next_word_first_char)
        template.append("07:" + cur_word_first_char)
        template.append("08:" + cur_word_last_char)

        for i in range(1, len(cur_word)-1):
            template.append("09:" + cur_word[i])
            template.append("10:" + cur_word[0] + "*" + cur_word[i])
            template.append("11:" + cur_word[-1] + "*" + cur_word[i])
            if cur_word[i] == cur_word[i+1]:
                template.append("13:" + cur_word[i] + "*" + "consecutive")
        if len(cur_word)>1 and cur_word[0] == cur_word[1]:
            template.append("13:" + cur_word[0] + "*" + "consecutive")

        if len(cur_word) == 1:
            template.append("12:" + cur_word + "*" + pre_word_last_char + "*" + next_word_first_char)

        for i in range(0, 4):
            if i > len(cur_word)-1:
                break
            template.append("14:" + cur_word[0:i+1])
            template.append("15:" + cur_word[-(i+1)::])
        return template

    '''构造单个特征向量'''
    def create_feature_template(self, sen, pos, pre_tag):
        template = []
        template.extend(self.create_bi_feature(pre_tag))
        template.extend(self.create_uni_feature(sen, pos))
        return template

    '''构造特征空间，对每个句子构造特征向量'''
    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sen = self.train_data.sentences[i]
            for j in range(len(sen.words)):
                if j == 0:
                    pre_tag = self.BOS
                else:
                    pre_tag = sen.tags[j-1]
                template = self.create_feature_template(sen, j, pre_tag)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in sen.tags:
                    if tag not in self.tagsList:
                        self.tagsList.append(tag)
        self.tagsList = sorted(self.tagsList)
        self.tag_id = {t: i for i, t in enumerate(self.tagsList)}
        self.id_tag = {i: t for i, t in enumerate(self.tagsList)}
        self.w = np.zeros((len(self.features),len(self.tagsList)))
        self.v = np.zeros((len(self.features),len(self.tagsList)))
        self.update_times = np.zeros((len(self.features),len(self.tagsList)))
        self.bi_features = [self.create_bi_feature(pre_tag) for pre_tag in self.tagsList]


        print("the total number of features is %d" % (len(self.features)))

    '''得到分值'''
    def score(self, feature, averaged = True):
        if averaged:
            scores = [self.v[self.features[f]]
                      for f in feature if f in self.features]
        else:
            scores = [self.w[self.features[f]] for f in feature if f in self.features]
        return np.sum(scores, axis = 0)

    '''预测一个句子的词性结果''' ######
    def predict(self, sen, averaged = False):
        max_score = np.zeros((len(sen.words), len(self.tagsList)))
        paths = np.zeros((len(sen.words), len(self.tagsList)), dtype="int")

        feature = self.create_bi_feature(self.BOS)
        feature.extend(self.create_uni_feature(sen, 0))
        max_score[0] = self.score(feature, averaged) # 得到第一行为第一个词为各个tag的分值

        bi_scores = [   #有问题，为什么是一个字一个字
            self.score(f, averaged)
            for f in self.bi_features
        ]

        for i in range(1, len(sen.words)):
            uni_feature = self.create_uni_feature(sen, i)
            uni_scores = self.score(uni_feature, averaged)
            scores = [[max_score[i-1][j] + bi_scores[j] + uni_scores]
                      for j in range(len(self.tagsList))]
            paths[i] = np.argmax(scores, axis=0) # 存的是前一个tag的索引
            max_score[i] = np.max(scores, axis=0) # 存的到目前为止的分值
        prev = np.argmax(max_score[-1]) #得到最后最高的值的索引

        predict = [prev]
        for i in range(len(sen.words)-1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tagsList[i] for i in reversed(predict)]

    '''根据w计算v'''
    def update_v(self, findex, tindex, update_time, last_w):
        last_update_time = self.update_times[findex][tindex]
        self.update_times[findex][tindex] = update_time
        self.v[findex][tindex] += (update_time - last_update_time - 1) * last_w + self.w[findex][tindex]

    '''评价准确度'''
    def evaluate(self, data, averaged=False):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sen = data.sentences[i]
            total_num += len(sen.words)
            predict = self.predict(sen, averaged)
            for j in range(len(sen.words)):
                if predict[j] == sen.tags[j]:
                    correct_num += 1
        return correct_num, total_num, correct_num/total_num




    def online_train(self,iteration=20, averaged=False, shuffle=False, exitor = 10):
        max_dev_precision = 0
        update_time = 0
        counter = 0
        if averaged:
            print("Using V to predict dev data", flush=True)
        for iter in range(iteration):
            print("iterator: %d" % (iter), flush=True)
            if shuffle:
                print("\tshuffle the train data...", flash=True)
                self.train_data.shuffle()
            startTime = datetime.datetime.now()
            bi_feature = self.bi_features
            for i in range(len(self.train_data.sentences)):
                sen = self.train_data.sentences[i]
                predict = self.predict(sen, averaged)
                if predict != sen.tags:
                    update_time += 1
                    for j in range(len(sen.words)):
                        uni_feature = self.create_uni_feature(sen, j)
                        if j == 0:
                            gold_bi_feature = self.create_bi_feature(self.BOS)
                            predict_bi_feature = self.create_bi_feature(self.BOS)
                        else:
                            gold_pre_tag = sen.tags[j-1]
                            predict_pre_tag = predict[j-1]
                            gold_bi_feature = bi_feature[self.tag_id[gold_pre_tag]]
                            predict_bi_feature = bi_feature[self.tag_id[predict_pre_tag]]

                        for f in uni_feature:
                            if f in self.features:
                                findex = self.features[f]
                                tindex = self.tag_id[sen.tags[j]]
                                last_w = self.w[findex][tindex]
                                self.w[findex][tindex] += 1
                                self.update_v(findex, tindex, update_time, last_w)

                                tindex = self.tag_id[predict[j]]
                                last_w = self.w[findex][tindex]
                                self.w[findex][tindex] -= 1
                                self.update_v(findex, tindex, update_time, last_w)

                        for f in gold_bi_feature:
                            if f in self.features:
                                findex = self.features[f]
                                tindex = self.tag_id[sen.tags[j]]
                                last_w = self.w[findex][tindex]
                                self.w[findex][tindex] += 1
                                self.update_v(findex, tindex, update_time, last_w)

                        for f in predict_bi_feature:
                            if f in self.features:
                                findex = self.features[f]
                                tindex = self.tag_id[predict[j]]
                                last_w = self.w[findex][tindex]
                                self.w[findex][tindex] -= 1
                                self.update_v(findex, tindex, update_time, last_w)

            # 一次迭代完成
            current_update_times = update_time
            for i in range(len(self.v)):
                for j in range(len(self.v[i])):
                    last_w = self.w[i][j]
                    last_update_time = self.update_times[i][j]
                    if current_update_times != last_update_time:
                        self.update_times[i][j] = current_update_times
                        self.v[i][j] += (current_update_times - last_update_time - 1) * last_w + self.w[i][j]

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data, averaged)
            print("\t"+"train_data precision : %d / %d = %f" % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data,averaged)
            print("\t"+"dev_data precision : %d / %d = %f" % (dev_correct_num, dev_num,dev_precision))

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1

            endtime = datetime.datetime.now()
            print("\titeration executing time is " + str((endtime-startTime).seconds)+ " s")
            if train_correct_num == total_num:
                break
            if counter >= exitor:
                break
        print("iterator = %d max_dev_precision = %f " % (max_iterator, max_dev_precision))


    def save_model(self):
        featuremodel = open("../result/POS-tagging-GLM-feature", mode="w", encoding='utf-8')
        for f in self.features:
            featuremodel.write(str(f)+'\n')
        featuremodel.close()

        tagFile = open("../result/POS-tagging-GLM-tagList", mode="w", encoding='utf-8')
        for tag in self.tagsList:
            tagFile.write(str(tag)+"\n")
        tagFile.close()

        weightFile = open("../result/POS-tagging-GLM-weight", mode="w", encoding='utf-8')
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                weightFile.write(str(self.w[i][j])+"\t")
            weightFile.write("\n")
        weightFile.close()

        vFile = open("../result/POS-tagging-GLM-v", mode="w", encoding='utf-8')
        for i in range(len(self.v)):
            for j in range(len(self.v[i])):
                vFile.write(str(self.v[i][j])+"\t")
            vFile.write("\n")
        vFile.close()













if __name__ == "__main__":
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    averaged = config['averaged']
    iterator = config['iterator']
    shuffle = config['shuffle']
    exitor = config['exitor']

    startTime = datetime.datetime.now()
    model = global_linear_model(train_data_file, dev_data_file, test_data_file)
    model.create_feature_space()
    model.online_train(iterator, averaged, shuffle, exitor)
    model.save_model()
    endTime = datetime.datetime.now()
    print("executing time is " + str((endTime-startTime).seconds) + "s")