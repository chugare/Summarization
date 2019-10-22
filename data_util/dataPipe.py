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
import sys
import os
from model import LDA
from meta.Meta import  Meta


class Reader:
    def __init__(self):
        pass



class DictFreqThreshhold:
    def __init__(self, **kwargs):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.N2WF = {}
        self.N2FREQ = {}
        self.WF2ID ={}
        self.ID2WF = {}
        self.freq_threshold = 0
        self.wordvec = None
        self.ULSW = ['\n', '\t',' ','']
        self.DictName = 'DP_comma_DICT.txt'
        self.DictSize = 80000
        for k in kwargs:
            self.__setattr__(k,kwargs[k])

        self.read_dic()
        self.DictSize = min(len(self.N2GRAM),self.DictSize)
        self.HuffmanEncoding()
        # print(max(self.N2HUFF, key=lambda k:len(self.N2HUFF[k])))
    def read_dic(self):
        try:
            dic_file = open(self.DictName, 'r', encoding='utf-8')
            wordFlagCount = 0
            wordCount = 0
            freqMode = True
            for line in dic_file:
                wordInfo = line.split(' ')
                if len(wordInfo)<1:
                    continue
                word = wordInfo[1]
                if word in self.ULSW:
                    continue
                wordIndex = int(wordInfo[0].strip())
                wordFlag = wordInfo[2]
                self.GRAM2N[word] = wordIndex
                self.N2GRAM[wordIndex] = word
                if len(wordInfo) > 3 and freqMode:

                    wordFreq = wordInfo[3]
                    try:
                        self.N2FREQ[wordIndex] = int(wordFreq.strip())
                    except Exception as e:
                        print(line)
                        print(wordInfo)
                else:
                    freqMode = False
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
            print(self.DictName)
            return
        print('[INFO] 字典初始化完毕，共计单词%d个'%len(self.N2GRAM))
    def dictReformat(self):
        size = len(self.N2GRAM)
        # newDic = []
        tmp_dic = open('tmp_dic.txt','w',encoding='utf-8')
        count = 0
        for k in self.N2GRAM:

            w = self.N2GRAM[k]
            if w in self.ULSW:
                continue
            f = self.N2WF[k]
            freq = self.N2FREQ[k]
            tmp_dic.write('%d %s %s %d\n'%(count,w,f,freq))
            count += 1
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
    def getHuffmanDict(self):
        maxHuffLen = len(self.N2HUFF[max(self.N2HUFF,key=lambda k:len(self.N2HUFF[k]))])
        print('Max Huff Len: %d'%maxHuffLen)
        try:
            meta_file = open('Huffman_Layer.json','r',encoding='utf-8')
            jsdata = json.load(meta_file)

            huffTable = jsdata[0]
            huffLabelTable = jsdata[1]
            huffLenTable = jsdata[2]
            print('[INFO] Huffman Layer Data has been read')
            return huffTable,huffLabelTable,huffLenTable
        except Exception:
            pass
        huffTable = []
        huffLabelTable = []
        huffLenTable = []
        for k in range(self.DictSize):
            # tmphuff = np.zeros(shape=[maxHuffLen],dtype=np.int32)
            tmphuff = [0]*maxHuffLen
            # tmplabel = np.zeros(shape=[maxHuffLen],dtype=np.int32)
            tmplabel = [0]*maxHuffLen
            if str(k) not in self.N2HUFF:
                huffTable.append(tmphuff)
                huffLabelTable.append(tmplabel)
                huffLenTable.append(0)
                continue
            huffman_str = self.N2HUFF[str(k)]
            tlen = len(huffman_str)
            coding = ""
            for i in range(len(huffman_str)):
                if i > 0:
                    coding = huffman_str[:i]
                tmphuff[i] = self.HUFF2LAYER[coding]
                tmplabel[i] = 0 if huffman_str[i] == '0' else 1

            huffTable.append(tmphuff)
            huffLabelTable.append(tmplabel)
            huffLenTable.append(tlen)
        meta_file = open('Huffman_Layer.json','w',encoding='utf-8')
        json.dump([huffTable,huffLabelTable,huffLenTable],meta_file)
        print('[INFO] Huffman Layer Data has been build')

        return huffTable,huffLabelTable,huffLenTable
    def read_word_from_Huffman(self,layersValues):

        encoding = ''
        try:
            while True:
                np = self.HUFF2LAYER[encoding]
                if layersValues[np] >0.5:
                    encoding += '1'
                else:
                    encoding += '0'
        except KeyError:
            if encoding not in self.HUFF2N:
                print('[ERROR] 在字典中没有找到对应的哈夫曼编码 “%s”'%encoding)
                return 0
            else:
                wordId = self.HUFF2N[encoding]
                return wordId
            pass

    def HuffmanEncoding(self,forceBuild = False):
        class HuffmanNode:
            def __init__(self,val = None,word = None):
                self.right = None
                self.left = None
                self.value = val
                self.word = word
                self.huffman = ''
        Nodes = [HuffmanNode(self.N2FREQ[k],k) for k in self.N2FREQ]
        if not forceBuild:
            try:
                meta_file = open('Huffman_dic.json','r',encoding='utf-8')
                self.N2HUFF,self.HUFF2N,self.HUFF2LAYER,self.LAYER2HUFF = json.load(meta_file)
                print('[INFO] Huffman dictionary has been readed')
                return
            except Exception:
                pass
        if len(Nodes) < 1:
            return

        while len(Nodes) > 1:
            Nodes.sort(key=lambda node: node.value, reverse=True)
            nv = Nodes[-1].value + Nodes[-2].value
            tmpNode = HuffmanNode(nv)
            tmpNode.left = Nodes[-2]
            tmpNode.right = Nodes[-1]
            Nodes.pop(-1)
            Nodes.pop(-1)
            Nodes.append(tmpNode)
        self.N2HUFF = {}
        self.HUFF2N = {}
        rootNode = Nodes[0]
        NodeQ = [rootNode]
        c = 0
        self.HUFF2LAYER = {}
        self.LAYER2HUFF = {}

        while len(NodeQ) > 0:
            tmpNode = NodeQ[0]

            NodeQ.pop(0)
            if c %1000==0:
                print('[INFO] Huffman Build %d'%c)
            if tmpNode.word is not None:
                self.N2HUFF[str(tmpNode.word)] = tmpNode.huffman
                self.HUFF2N[tmpNode.huffman] = tmpNode.word
                continue
            self.HUFF2LAYER[tmpNode.huffman] = c
            self.LAYER2HUFF[str(c)] = tmpNode.huffman
            if tmpNode.left is not None:
                tmpNode.left.huffman = tmpNode.huffman + '0'
                NodeQ.append(tmpNode.left)
            if tmpNode.right is not None:
                tmpNode.right.huffman = tmpNode.huffman + '1'
                NodeQ.append(tmpNode.right)
            c+= 1
        meta_file = open('Huffman_dic.json', 'w', encoding='utf-8')
        json.dump([self.N2HUFF,self.HUFF2N,self.HUFF2LAYER,self.LAYER2HUFF],meta_file)


        # for k in self.N2HUFF:
        #     print("%s %d %s"%(self.N2GRAM[k],self.N2FREQ[k],self.N2HUFF[k]))


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
        lda = LDA.LDA_Train(TaskName = self.TaskName, SourceFile = self.SourceFile, DictName = self.DictName)
        self.LdaMap = lda.getLda()
        self.DictSize = self.Dict.DictSize
        self.get_word_mat()

    def get_key_word(self,line,num):
            line = line[:]
            random.shuffle(line)
            num = min(num,len(line))
            return line[:num]


    def pipe_data(self,**kwargs):

        try:
            contentLen = kwargs['ContextLen']
            vecSize = kwargs['VecSize']
            keyWordNum = kwargs['KeyWordNum']
            topicNum = kwargs['TopicNum']
            flagNum = kwargs['FlagNum']
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
            ref_word = self.get_key_word(words,keyWordNum)
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
                    # elif len(ref_word) < keyWordNum :
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
            for i in range(len(refVector),keyWordNum):
                refVector.append(np.zeros([vecSize]))
            for preWord,preTopic,preFlag,topic,flag,currentWordId,select,selectWord in lineBatch:

                # if select > 0:
                #     selectWord = refMap[selectWord]
                # else:
                #     selectWord = keyWordNum
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

        metaFile = open('meta_tfrecord.json', 'w', encoding='utf-8')
        json.dump(meta, metaFile, ensure_ascii=False)
        metaFile.close()
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
            contentLen = meta['ContextLen']
            vecSize = meta['VecSize']
            keyWordNum = meta['KeyWordNum']

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
            'keyWordVector': tf.FixedLenFeature(shape=[keyWordNum*vecSize],dtype=tf.float32),
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
        IdPW_WTF = np.argmax(PW)
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

    def pipe_data_for_eval(self,**kwargs):
        try:
            vecSize = self.VecSize
            keyWordNum = self.KeyWordNum
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
            topicSeq = []
            ref_word = self.get_key_word(words,keyWordNum)
            ref_word = {k:self.WordVectorMap.get_vec(k) for k in ref_word}
            flagSeq =[]
            for word in words:
                topic = topicNum
                currentWordId,flag = self.Dict.get_id_flag(word)
                if  currentWordId <0:
                    currentWordId = random.randint(0, self.DictSize - 1)
                    topic = self.LdaMap[currentWordId]
                    flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    wordVec = self.WordVectorMap.get_vec(word)

                topic = self.LdaMap[currentWordId]
                topicSeq.append(topic)
                flagSeq.append(flag)
            refMap = {}
            refVector = []
            for i,k in enumerate(ref_word):
                refMap[i] = k
                refVector.append(ref_word[k])
            for i in range(len(refVector),keyWordNum):
                refVector.append(np.zeros([vecSize]))
            ref_word = ref_word.keys()
            yield words,flagSeq,topicSeq,refMap,refVector,ref_word
        # except Exception as e:
        #     print(str(e))




if __name__ == '__main__':

    def getmeta(**kwargs):
        return kwargs

    def unit_test():
        meta = Meta().get_meta()

        dp = DataPipe(**meta)
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

            res = sess.run([topicLabel,flagLabel,wordLabel,keyWordVector, topicSeq,flagSeq,wordVector])
            for i in res:
                print(i)
            coord.request_stop()
            coord.join(threads)
    def t1():
        dp = DataPipe(TaskName = 'DP',ReadNum = 10,DictName='DP_DICT.txt')
        meta = Meta(ReadNum = 10).get_meta()
        g = dp.pipe_data(**meta)
        for w in g:
            pass


    args = sys.argv
    try:
        TaskName = args[1]
    except IndexError:
        TaskName = 'DP_lite'
    try:
        DictName = args[2]
    except IndexError:
        DictName = TaskName+'_DICT.txt'

    try:
        SourceFile = args[3]
    except IndexError:
        SourceFile = TaskName+'.txt'

    try:
        ReadNum = int(args[4])
    except IndexError:
        ReadNum = 10

    meta  = Meta().get_meta()
    dc = DictFreqThreshhold(ReadNum = ReadNum,DictName = DictName,SourceFile = SourceFile)
    # dc.HuffmanEncoding()
    dc.getHuffmanDict()
    print(' ')
    # # unit_test()
    # dp = DataPipe(**meta)
    # # meta = getmeta(**meta)
    # dp.write_TFRecord(meta,int(args[2]))
