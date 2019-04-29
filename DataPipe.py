# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/30 下午6:53
#   南京大学软件学院 Nanjing University Software Institute
#
#   这个文件用于生成训练所需要的数据，主要包括以下几个方法
#   1. 从文件中读取，生成原始文本与摘要文本的序列
#   2. 将文本序列表示成字典序列
#   3. 生成batch数据
#   4. 去除停用词

import jieba
import numpy as np
import random
import tensorflow as tf
import re
import json
import logging
import sys
import os
import LDA
import struct
from  Meta import  Meta
from sklearn.metrics.pairwise import cosine_similarity



class Reader:
    def __init__(self):
        pass


class WordVec:
    def __init__(self,**kwargs):
        self.vec_dic = {}
        self.word_list = []
        self.vec_list = []
        self.num = 0
        self.ReadNum = -1
        self.VecFile = 'word_vec.char'
        print('[INFO] Start load word vector')
        for k in kwargs:
            self.__setattr__(k,kwargs[k])
        self._read_vec()
    def dump_file(self):
        file = open('word_vec.char','w',encoding='utf-8')
        file.write(str(self.num)+' 300\n')
        for w in self.vec_dic:
            vec_f = [str(i) for i in self.vec_dic[w]]
            vec_str = ' '.join(vec_f)
            file.write(w+' '+vec_str+'\n')
        file.close()
    @staticmethod
    def ulw(word):
        pattern = [
            r'[,.\(\)（），。\-\+\*/\\_|]{2,}',
            r'\d+',
            r'[qwertyuiopasdfghjklzxcvbnm]+',
            r'[ｑｗｅｒｔｙｕｉｏｐａｓｄｆｇｈｊｋｌｚｘｃｖｂｎｍ]+',
            r'[QWERTYUIOPASDFGHJKLZXCVBNM]+',
            r'[ＱＷＥＲＴＹＵＩＯＰＡＳＤＦＧＨＪＫＬＺＸＣＶＢＮＭ]+',
            r'[ⓐ ⓑ ⓒ ⓓ ⓔ ⓕ ⓖ ⓗ ⓘ ⓙ ⓚ ⓛ ⓜ ⓝ ⓞ ⓟ ⓠ ⓡ ⓢ ⓣ ⓤ ⓥ ⓦ ⓧ ⓨ ⓩ]+',
            r'[Ⓐ Ⓑ Ⓒ Ⓓ Ⓔ Ⓕ Ⓖ Ⓗ Ⓘ Ⓙ Ⓚ Ⓛ Ⓜ Ⓝ Ⓞ Ⓟ Ⓠ Ⓡ Ⓢ Ⓣ Ⓤ Ⓥ Ⓦ Ⓧ Ⓨ Ⓩ ]+',
        ]
        ulwf = open('uslw.txt','a',encoding='utf-8')
        for p in pattern:
            mr = re.match(p,word)
            if mr is not None:
                ulwf.write(word+'\n')
                ulwf.close()
                return True
        return  False

    def clear_ulw(self):
        vec_file = open(self.VecFile,'r',encoding='utf-8')
        meg = next(vec_file).split(' ')
        num = int(meg[0])
        file = open(self.VecFile, 'w', encoding='utf-8')

        count = 0

        for l in vec_file:
            m = l.strip().split(' ')
            w = m[0]
            if WordVec.ulw(w):
                continue
            count+=1
            if count%10000 == 0:
                p = float(count)/num*100
                sys.stdout.write('\r[INFO] write cleared vec data, %d finished'%count)
            file.write(l)
            # vec_dic[w] = vec
        print('\n Final count : %d'%count)
    def _read_vec(self):
        path = os.path.abspath('.')
        print(path)
        # path = '/'.join(path.split('\\')[:-1])+'/sgns.merge.char'
        # path = 'F:/python/word_vec/sgns.merge.char'
        # path = 'D:\\赵斯蒙\\EVI-fact\\word_vec.char'
        path = self.VecFile
        vec_file = open(path,'r',encoding='utf-8')
        # meg = next(vec_file).split(' ')
        # num = int(meg[0])
        # self.num = num
        count = 0
        for l in vec_file:

            m = l.strip().split(' ')
            w = m[0]
            vec = m[1:]
            vec =[float(v) for v in m[1:]]
            # if WORD_VEC.ulw(w):
            #     continue
            count+=1
            if count%10000 == 0:
                sys.stdout.write('\r[INFO] Load vec data, %d finished'%count)
            if count == self.ReadNum:
                break
            self.vec_list.append(vec)
            self.word_list.append(w)
            self.vec_dic[w] = np.array(vec,dtype=np.float32)

        print('\n[INFO] Vec data loaded')
        self.num = count
    def get_min_word(self,word):
        vec = self.vec_dic[word]
        dis = cosine_similarity(self.vec_list,[vec])
        dis = np.reshape(dis,[-1])
        dis_pair = [(i,dis[i]) for i in range(len(dis))]
        dis_pair.sort(key= lambda x:x[1],reverse=True)
        for i in range(10):
            print(self.word_list[dis_pair[i][0]])

    def get_min_word_v(self, vec):
        dis = cosine_similarity(self.vec_list, [vec])
        dis = np.reshape(dis, [-1])
        i = np.argmax(dis)
        return self.word_list[i]
    def get_sentence(self,vlist,l):
        result = ''
        x = 0
        for vec in vlist:
            if x == l:
                break
            print('[INFO] Search for nearest word on index %d'%x)
            dis = cosine_similarity(self.vec_list, [vec])
            dis = np.reshape(dis, [-1])
            i = np.argmax(dis)
            x+= 1
            print(self.word_list[i])
            result += self.word_list[i]

        return result
    def sen2vec(self,sen):
        sen = jieba.lcut(sen)
        vec_out = []
        for w in sen:
            if w in self.vec_dic:
                vec_out.append(self.vec_dic[w])
        return vec_out
    def get_vec(self,word):
        try:
            return self.vec_dic[word]
        except KeyError:
            return np.zeros([len(self.vec_list[0])])


class DictFreqThreshhold:
    def __init__(self, **kwargs):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.N2WF = {}
        self.WF2ID ={}
        self.ID2WF = {}
        self.freq_threshold = 0
        self.wordvec = None
        self.ULSW = ['\n', '\t',' ','\n']
        self.DictName = 'DICT.txt'
        self.DictSize = 80000
        for k in kwargs:
            self.__setattr__(k,kwargs[k])

        self.read_dic()
        self.DictSize = min(len(self.N2GRAM),self.DictSize)
    def read_dic(self):
        try:
            dic_file = open(self.DictName, 'r', encoding='utf-8')
            wordFlagCount = 0
            wordCount = 0
            for line in dic_file:
                wordInfo = line.split(' ')
                if len(wordInfo)<1:
                    continue
                word = wordInfo[1]

                wordIndex = int(wordInfo[0].strip())
                wordFlag = wordInfo[2]
                self.GRAM2N[word] = wordIndex
                self.N2GRAM[wordIndex] = word
                self.N2WF[wordIndex] = wordFlag
                if wordFlag not in self.WF2ID:
                    self.WF2ID[wordFlag] = wordFlagCount
                    self.ID2WF[wordFlagCount] = wordFlag
                    wordFlagCount += 1
                wordCount += 1
                if(self.DictSize is not  None ) and wordCount >=self.DictSize:
                    break
        except FileNotFoundError:
            print('[INFO] 未发现对应的*_DIC.txt文件，需要先初始化，初始化完毕之后重新运行程序即可')
            return
        print('[INFO] 字典初始化完毕，共计单词%d个'%len(self.N2GRAM))
    def doc2bow(self,doc):
        if isinstance(doc,list):
            wordSet = doc
        else:
            wordSet = doc.split(' ')
        wordCount = {}
        for w in wordSet:
            if w in self.GRAM2N :
                if w not in wordCount:
                    wordCount[self.GRAM2N[w]] = 0
                wordCount[self.GRAM2N[w]] += 1
        res = sorted(wordCount.items(),key=lambda x:x[1])
        return res

    def get(self,id):
        if id in self.N2GRAM:
            return self.N2GRAM[id]
        else:
            return ""

    def get_id(self,word):
        if word in self.GRAM2N:
            return self.GRAM2N[word]
        else:
            return -1

    def get_id_flag(self,word):
        if word in self.GRAM2N:
            id = self.GRAM2N[word]
            return id,self.WF2ID[self.N2WF[id]]
        else:
            return -1,-1
    def get_sentence(self, indexArr,cutSize = None):

        res = ''
        for i in range(len(indexArr)):
            if cutSize != None:
                if indexArr[i] > 1:
                    res+=(self.N2GRAM[indexArr[i]])
                if len(res)>cutSize:
                    break
            else:
                if indexArr[i] != 1:
                    res+=(self.N2GRAM[indexArr[i]])
                else:
                    break

        return res

    def get_char_list(self, index_arr):
        res = []
        for i in range(len(index_arr)):
            if index_arr[i] != 1:
                res.append(self.N2GRAM[index_arr[i]])
            else:
                break

        return res

    def Nencoder(self, ec_str):
        grams = jieba.lcut(ec_str)
        ec_vecs = [2]

        for gram in grams:
            if gram in self.GRAM2N:
                ec_vecs.append(self.GRAM2N[gram])
            else:
                # 当词典中没有对应的词时，简单的把单词变成unk符号，抑或是进行进一步的分词？
                ec_vecs.append(0)
        ec_vecs.append(1)
        return np.array(ec_vecs, np.int32)

    def bowencoder(self, ohcode, V):
        res = np.zeros([V], np.int32)
        for c in ohcode:
            res[c] = 1
        return res

    def read_file(self,data_source):
        source = open(data_source, 'r', encoding='utf-8')
        dt = json.load(source)
        for i in dt:
            yield i
    @staticmethod
    def context(title, pos, C):
        res = np.zeros([C], np.int32)
        for i in range(C):
            if pos-i-1<0:
                res[C - i - 1] = 0
            else:
                res[C - i - 1] = title[pos - i - 1]
        return res
class DataPipe:

    def __init__(self,**kwargs):
        self.SourceFile = 'DP.txt'
        self.TaskName = 'DP'
        self.Name = 'DP_gen'
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000
        for k in kwargs:
            self.__setattr__(k,kwargs[k])
        self.Dict  = DictFreqThreshhold(DictName = self.DictName,DictSize = self.DictSize)
        self.WordVectorMap = WordVec(**kwargs)
        lda = LDA.LDA_Train(TaskName = self.TaskName,SourceFile = self.SourceFile,DictName = self.DictName)
        self.LdaMap = lda.getLda()
        self.DictSize = self.Dict.DictSize
    def get_key_word(self,line,num):
            line = line[:]
            random.shuffle(line)
            num = min(num,len(line))
            return line[:num]


    def pipe_data(self,**kwargs):

        try:
            contentLen = kwargs['ContextLength']
            vecSize = kwargs['VecSize']
            refSize = kwargs['KeyWordNum']
            topicNum = kwargs['TopicNum']
            flagNum = kwargs['FlagNum']
            metaFile = open('meta_tfrecord.json','w',encoding='utf-8')
            json.dump(kwargs,metaFile,ensure_ascii=False)
            metaFile.close()
        except KeyError as e:
            print(str(e))
            return
        sourceFile = open(self.SourceFile,'r',encoding='utf-8')
        for line in sourceFile:
            line  = line.strip()
            words = line.split(' ')
            preWord = [np.zeros([vecSize]) for _ in range(contentLen)]
            preTopic = [topicNum for _ in range(contentLen)]
            preFlag = [flagNum for _ in range(contentLen)]
            ref_word = self.get_key_word(words,refSize)
            ref_word = {k:self.WordVectorMap.get_vec(k) for k in ref_word}
            lineBatch = []
            for word in words:
                select = 0
                selectWord = ""
                currentWordId,flag = self.Dict.get_id_flag(word)
                if currentWordId <0:

                    # 生成单词不从自定义单词表中选择的情况


                    currentWordId = random.randint(0,self.DictSize-1)
                    topic = self.LdaMap[currentWordId]
                    flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    wordVec = self.WordVectorMap.get_vec(word)

                    # 生成单词自定义一个单词表并且从中进行选择的情况

                    # currentWordId = 0
                    # flag = 0
                    # if word in ref_word:
                    #     select = 1
                    #     selectWord = word
                    #     wordVec = ref_word[word]
                    # elif len(ref_word) < refSize :
                    #     if word in self.WordVectorMap.vec_dic:
                    #         ref_word[word] = self.WordVectorMap.get_vec(word)
                    #         topic = topicNum
                    #         flag = flagNum
                    #     else:
                    #         ref_word[word] = np.zeros([vecSize])
                    #
                    #     select = 1
                    #     selectWord = word
                    #     wordVec = ref_word[word]
                    # else:
                    #
                    #     continue
                else:
                    topic = self.LdaMap[currentWordId]
                    wordVec = self.WordVectorMap.get_vec(word)

                lineBatch.append((preWord,preTopic,preFlag,topic,flag,currentWordId,select,selectWord))
                # yield {
                #     'wordVector': np.array(preWord, dtype=np.float32),
                #     'topicSeq': np.array(preTopic, dtype=np.int64),
                #     'flagSeq': np.array(preFlag, dtype=np.int64),
                #     # 'keyWordVector': np.array(refVector, dtype=np.float32),
                #     'topicLabel': topic,
                #     'flagLabel': flag,
                #     'wordLabel': currentWordId,
                #     # 'selLabel': select,
                #     # 'selWordLabel': selectWord,
                # }


                preWord = preWord[1:]
                preWord.append(wordVec)
                preFlag = preFlag[1:]
                preFlag.append(flag)
                preTopic = preTopic[1:]
                preTopic.append(topic)

            refMap = {}
            refVector = []
            for i,k in enumerate(ref_word):
                refMap[k] = i
                refVector.append(ref_word[k])
            for i in range(len(refVector),refSize):
                refVector.append(np.zeros([vecSize]))
            for preWord,preTopic,preFlag,topic,flag,currentWordId,select,selectWord in lineBatch:

                # if select > 0:
                #     selectWord = refMap[selectWord]
                # else:
                #     selectWord = refSize
                yield {
                    'wordVector':np.array(preWord,dtype=np.float32),
                    'topicSeq':np.array(preTopic,dtype=np.int64),
                    'flagSeq':np.array(preFlag,dtype=np.int64),
                    'keyWordVector':np.array(refVector,dtype=np.float32),
                    'topicLabel':topic,
                    'flagLabel':flag,
                    'wordLabel':currentWordId,
                    # 'selLabel':select,
                    # 'selWordLabel':selectWord,
                }
        pass

    def write_TFRecord(self,meta,million):

        gen = self.pipe_data(**meta)
        CountK = 0
        CountM = 0
        KCount = 0
        writer = tf.python_io.TFRecordWriter(self.Name + '-%d.tfrecord'%CountM)
        for v in gen:

            example = self.get_feature(**v)
            writer.write(example.SerializeToString())
            KCount += 1
            if KCount %1000 == 0:
                CountK += 1
                KCount = 0

                print("[INFO] %d K Samples read to record "%CountK)
                if CountK%1000 == 0:
                    CountM+=1
                    print("[INFO] %d M Samples has been read, Writing to record " % CountM)
                    writer.close()
                    if CountM == million:
                        break
                    writer = tf.python_io.TFRecordWriter(self.Name + '-%d.tfrecord'%CountM)

    def read_TFRecord(self,BATCH_SIZE):

        metaFile = open('meta_tfrecord.json','r',encoding='utf-8')
        meta = json.load(metaFile)
        try:
            contentLen = meta['ContextLength']
            vecSize = meta['VecSize']
            refSize = meta['KeyWordNum']

        except KeyError as e:
            print(str(e))
            return
        files = os.listdir('.')
        recordList = []
        for f in files:
            if f.endswith('.tfrecord'):
                recordList.append(f)

        fileQueue = tf.train.string_input_producer(recordList)
        recordReader = tf.TFRecordReader()
        i,serializeExample = recordReader.read(fileQueue)
        features = tf.parse_single_example(serializeExample,features={
            'wordVector': tf.FixedLenFeature(shape=[contentLen*vecSize],dtype=tf.float32),
            'topicSeq': tf.FixedLenFeature(shape=[contentLen],dtype=tf.int64),
            'flagSeq': tf.FixedLenFeature(shape=[contentLen],dtype=tf.int64),
            'keyWordVector': tf.FixedLenFeature(shape=[refSize*vecSize],dtype=tf.float32),
            'topicLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            'flagLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            'wordLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            # 'selLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            # 'selWordLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
        })
        # batchData = tf.train.batch(features,batch_size=BATCH_SIZE,num_threads=4,capacity=1000)
        batchData = tf.train.shuffle_batch(features,batch_size=BATCH_SIZE,
                                           capacity=20000,num_threads=4,
                                           min_after_dequeue=10000)
        return batchData

    def get_feature(self,**kwargs):
        features = {}
        for k in kwargs:
            var = kwargs[k]
            if not np.isscalar(var):
                var = np.reshape(var,[-1])
            else:
                var = np.array([var])

            if var.dtype == np.float32 or var.dtype == np.float64:
                features[k] = tf.train.Feature(float_list = tf.train.FloatList(value = var))
            elif var.dtype == np.int32 or var.dtype == np.int64:
                features[k] = tf.train.Feature(int64_list = tf.train.Int64List(value = var))
            else:
                features[k] = tf.train.Feature(bytes_list = tf.train.BytesList(value = var))
        example = tf.train.Example(features=tf.train.Features(
            feature=features
        ))
        return example

class EvalProvider:
    def __init__(self,**kwargs):
        self.TaskName = None
        self.SourceFile = None
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000

        self.VecSize = 300
        self.RefSize = 20
        self.TopicNum = 30
        self.FlagNum = 60
        self.ContextLen = 10

        for k in kwargs:
            self.__setattr__(k,kwargs[k])
        self.Dict  = DictFreqThreshhold(DictName = self.DictName,DictSize = self.DictSize)
        self.DictSize = self.Dict.DictSize
        self.WordVectorMap = WordVec(**kwargs)
        lda = LDA.LDA_Train(TaskName = self.TaskName,SourceFile = self.SourceFile,DictName = self.DictName)
        self.LdaMap = lda.getLda()
        self.get_word_mat()

    def get_word_mat(self):
        wordNum = self.Dict.DictSize
        flagNum = self.FlagNum
        topicNum = self.TopicNum
        self.TopicWordMat = []
        self.FlagWordMat = []
        for i in range(wordNum):
            ftmp = np.zeros(flagNum)
            ftmp[self.Dict.WF2ID[self.Dict.N2WF[i]]] = 1
            ttmp = np.zeros(topicNum)
            topic = self.LdaMap[i]
            if topic == 30:
                ttmp = np.ones([topicNum],dtype=np.float32) * (1.0/30.0)
            else:
                ttmp[topic] = 1
            self.TopicWordMat.append(ttmp)
            self.FlagWordMat.append(ftmp)
        self.FlagWordMat =np.array(self.FlagWordMat)
        self.TopicWordMat = np.array(self.TopicWordMat)
    def _get_prob(self,PW,PF,PT):
        PW_T = np.matmul(self.TopicWordMat,np.transpose(PT))
        PW_F = np.matmul(self.FlagWordMat,np.transpose(PF))
        PW_T = np.squeeze(PW_T)
        PW_F = np.squeeze(PW_F)
        PW = np.squeeze(PW)
        disSize = min(PW.shape[0],PW_F.shape[0])
        PW = PW[:disSize]
        PW_WT = PW * PW_T
        PW_WF = PW * PW_F

        PW_WTF = PW_WT * PW_F
        return PW,PW_WT,PW_WF,PW_WTF
    def get_prob_result(self,PW,PF,PT):
        PW, PW_WT, PW_WF, PW_WTF = self._get_prob(PW,PF,PT)
        IdPW_WTF = np.argmax(PW_WTF)
        resWord = self.Dict.N2GRAM[IdPW_WTF]
        resFlag = self.Dict.N2WF[IdPW_WTF]
        resTopic = self.LdaMap[IdPW_WTF]
        return resWord,resFlag,resTopic

    def get_input_data(self,wordSeq=None,topicSeq=None, flagSeq=None, newWord = None):
        if topicSeq is None and flagSeq is None and wordSeq is None :
            wordSeq = [np.zeros(self.VecSize) for _ in range(self.ContextLen)]
            topicSeq = [self.TopicNum for _ in range(self.ContextLen)]
            flagSeq = [self.FlagNum for _ in range(self.ContextLen)]
        else:
            try:
                wordSeq = wordSeq[1:]
                topicSeq = topicSeq[1:]
                flagSeq = flagSeq[1:]
                newWordID = self.Dict.GRAM2N[newWord]
                nVector = self.WordVectorMap.get_vec(newWord)
                nFlag = self.Dict.WF2ID[self.Dict.N2WF[newWordID]]
                nTopic = self.LdaMap[newWordID]
                wordSeq.append(nVector)
                topicSeq.append(nTopic)
                flagSeq.append(nFlag)

            except KeyError as e:
                print(str(e))
        return wordSeq,topicSeq,flagSeq




    def get_next_context(self,preWordSeq,topicSeq,flagSeq,lastWord,lastSelected):
        preWordSeq = preWordSeq[1:]
        id,flag = self.Dict.get_id_flag(lastWord)
        lastWordVector = self.WordVectorMap.get_vec(id)
        preWordSeq.append(lastWordVector)
        topic = self.LdaMap.get(id)
        topicSeq = topicSeq[1:]
        flagSeq = flagSeq[1:]
        topicSeq.append(topic)
        flagSeq.append(flag)

    def get_init_context(self):
        preWordSeq = [np.zeros([self.VecSize]) for _ in range(self.ContextLen)]
        topicSeq = [self.TopicNum  for _ in range(self.ContextLen)]
        flagSeq = [self.FlagNum for _ in range(self.ContextLen)]
        return preWordSeq,topicSeq,flagSeq
    def read_data_for_eval(self,**kwargs):
        try:
            vecSize = self.VecSize
            refSize = self.RefSize
            topicNum = self.TopicNum
            flagNum = self.FlagNum
        except KeyError as e:
            print(str(e))
            return
        # try:
        sourceFile = open(self.SourceFile,'r',encoding='utf-8')
        for line in sourceFile:
            line  = line.strip()
            words = line.split(' ')
            ref_word = {}
            topicSeq = []
            flagSeq =[]
            for word in words:
                topic = topicNum
                currentWordId,flag = self.Dict.get_id_flag(word)
                if  currentWordId <0:
                    currentWordId = 0
                    flag = 0
                    if word in ref_word:
                        wordVec = ref_word[word]
                    elif len(ref_word) < refSize :
                        if word in self.WordVectorMap.vec_dic:
                            ref_word[word] = self.WordVectorMap.get_vec(word)
                            topic = topicNum
                            flag = flagNum
                        else:
                            ref_word[word] = np.random.rand(vecSize)
                        wordVec = ref_word[word]
                    else:
                        continue

                else:
                    topic = self.LdaMap[currentWordId]
                topicSeq.append(topic)
                flagSeq.append(flag)
            refMap = {}
            refVector = []
            for i,k in enumerate(ref_word):
                refMap[i] = k
                refVector.append(ref_word[k])
            for i in range(len(refVector),refSize):
                refVector.append(np.zeros([vecSize]))
            ref_word = ref_word.keys()
            yield words,flagSeq,topicSeq,refMap,refVector,ref_word
        # except Exception as e:
        #     print(str(e))



if __name__ == '__main__':

    def getmeta(**kwargs):
        return kwargs

    def unit_test():
        dp = DataPipe(TaskName='DP',ReadNum = 20000)
        input = dp.read_TFRecord(64)
        keyWordVector = input['keyWordVector']
        wordVector = input['wordVector']
        topicSeq = input['topicSeq']
        flagSeq = input['flagSeq']
        topicLabel = input['topicLabel']
        flagLabel = input['flagLabel']
        wordLabel = input['wordLabel']
        # selWordLabel = input['selWordLabel']
        # selLabel = input['selLabel']
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            fr1, fr2, fr3 = sess.run([topicLabel,keyWordVector, wordVector])
            print(fr1)
            print(fr2)
            print(fr3)
            coord.request_stop()
            coord.join(threads)
    def t1():
        dp = DataPipe(TaskName = 'DP',ReadNum = int(args[1]),DictName='DP_DICT.txt')


    # unit_test()
    args = sys.argv
    meta  = Meta().get_meta()
    dp = DataPipe(TaskName=meta['TaskName'], ReadNum=int(args[1]), DictName=meta['DictName'])
    # meta = getmeta(**meta)
    dp.write_TFRecord(meta,int(args[2]))
