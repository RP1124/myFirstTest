
import numpy as np


class POS_tagging_predict():
    def __init__(self):
        self.w = []
        self.v = []
        self.tagsList = []
        self.features ={}
        self.bi_features = []
        self.tag_id = {}
        self.id_tag = {}
        self.BOS = "BOS"

    def read_model(self, averaged=False): # app/model/POS_tagging_GLM/result/POS-tagging-GLM-tagList
        featuremodel = open("../result/POS-tagging-GLM-feature", mode="r", encoding='utf-8')
        for s in featuremodel:
            s = s.replace("\n", "")
            if s not in self.features:
                self.features[s] = len(self.features)
        featuremodel.close()

        tagFile = open("../result/POS-tagging-GLM-tagList", mode="r", encoding='utf-8')
        for s in tagFile:
            s = s.replace("\n", "")
            if s not in self.tagsList:
                self.tagsList.append(s)
        tagFile.close()

        weightFile = open("../result/POS-tagging-GLM-weight", mode="r", encoding='utf-8')
        for line in weightFile:
            line = line.replace("\n", "")
            line = line.split("\t")
            line.pop()
            for i in range(len(line)):
                line[i] = float(line[i])
            self.w.append(line)
        weightFile.close()
        if averaged:
            vFile = open("../result/POS-tagging-GLM-v", mode="r", encoding='utf-8')
            for line in vFile:
                line = line.replace("\n", "")
                line = line.split("\t")
                line.pop()
                for i in range(len(line)):
                    line[i] = float(line[i])
                self.v.append(line)
            vFile.close()

        self.tag_id = {t: i for i, t in enumerate(self.tagsList)}
        self.id_tag = {i: t for i, t in enumerate(self.tagsList)}
        self.bi_features = [self.create_bi_feature(pre_tag) for pre_tag in self.tagsList]

    def create_bi_feature(self, pre_tag):
        return ['01:'+ pre_tag]

    def create_uni_feature(self, sen, pos):
        template = []
        cur_word = sen[pos]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if pos == 0:
            pre_word = '##'
            pre_word_last_char = "#"
        else:
            pre_word = sen[pos-1]
            pre_word_last_char = pre_word[-1]

        if pos == len(sen)-1:
            next_word = "$$"
            next_word_first_char = "$"
        else:
            next_word = sen[pos+1]
            next_word_first_char = next_word[0]

        template.append("02:" + cur_word)
        template.append("03:" + pre_word)
        template.append("04:" + next_word)
        template.append("05:" + cur_word + "*" + pre_word_last_char)
        template.append("06:" + cur_word + "*" + next_word_first_char)
        template.append("07:" + cur_word_first_char)
        template.append("08:" + cur_word_last_char)

        for i in range(1,len(cur_word)-1):
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
            if i > len(cur_word) - 1:
                break
            template.append("14:" + cur_word[0:i + 1])
            template.append("15:" + cur_word[-(i + 1)::])
        return template

    def score(self, feature, averaged=True):
        if averaged:
            scores = [self.v[self.features[f]] for f in feature if f in self.features]
        else:
            scores = [self.w[self.features[f]] for f in feature if f in self.features]
        return np.sum(scores, axis=0)

    def predict(self, sen, averaged):
        max_score = np.zeros((len(sen), len(self.tagsList)))
        paths = np.zeros((len(sen), len(self.tagsList)), dtype="int")

        feature = self.create_bi_feature(self.BOS)
        feature.extend(self.create_uni_feature(sen, 0))
        max_score[0] = self.score(feature, averaged)

        bi_scores = [
            self.score(f, averaged)
            for f in self.bi_features
        ]

        for i in range(1, len(sen)):
            uni_feature = self.create_uni_feature(sen, i)
            uni_scores = self.score(uni_feature, averaged)
            scores = [[max_score[i - 1][j] + bi_scores[j] + uni_scores]
                      for j in range(len(self.tagsList))]
            paths[i] = np.argmax(scores, axis=0)
            max_score[i] = np.max(scores, axis=0)
        prev = np.argmax(max_score[-1])

        predict = [prev]
        for i in range(len(sen) - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tagsList[i] for i in reversed(predict)]












'''
if __name__ == "__main__":
    # 测试用
    test = POS_tagging_predict()
    test.read_model()
    sen = ["戴相龙", "说", "中国", "经济", "发展", "为", "亚洲", "作出", "积极", "贡献"]
    res = test.predict(sen, False)
    print(res)
'''