import tensorflow as tf
import random
from DataPipe import DictFreqThreshhold,WordVec
import LDA
import numpy as np

class Model:
    def __init__(self):
        self.RNNUnitNum = 400
        self.KeyWordNum = 5
        self.VecSize = 300
        self.ContextLen = 10
        self.HiddenUnit = 800
        self.ContextVec = 400
        self.WordNum = 80000
        self.BatchSize = 128
        self.L2NormValue = 0.02
        self.DropoutProb = 0.7
        self.GlobalNorm = 0.5
        self.LearningRate = 0.001
        self.MaxSentenceLength = 100
        pass
    def get_attention(self,q,k,name='Attention'):
        assert q is tf.Tensor
        assert k is tf.Tensor
        w = self.get_variable(name=name+'_weight',shape = [q.shape[-1],k.shape[-1]],dtype=tf.float32,
                              initializer=tf.glorot_uniform_initializer())
        q = tf.expand_dims(q,1)
        align = tf.tensordot(q,w,[-1,0])*k
        align = tf.reduce_sum(align,axis=-1)
        align = tf.nn.softmax(align)
        align = tf.expand_dims(align,-1)
        v = align*k
        v = tf.reduce_sum(v,1)
        return v
    def get_variable(self,name,shape,dtype,initializer=None):
        if initializer == None:
            initializer = tf.truncated_normal_initializer(stddev=0.2)
        var = tf.get_variable(name,shape,dtype,initializer)
        l2_norm = tf.reduce_mean(var**2) * self.L2NormValue
        tf.add_to_collection('l2norm',l2_norm)
        return var
    def build_model(self):

        SentenceVector = tf.placeholder(dtype=tf.float32,shape=[self.BatchSize,self.MaxSentenceLength,self.VecSize],name='Sequence_Vector')
        KeyWordVector = tf.placeholder(dtype=tf.float32,shape=[self.BatchSize,self.KeyWordNum,self.VecSize],name='Sequence_Vector')
        WordLabel = tf.placeholder(dtype=tf.int64,shape=[self.BatchSize,self.MaxSentenceLength],name="Word_Label")
        SentenceLength = tf.placeholder(dtype=tf.int64,shape=[self.BatchSize,self.MaxSentenceLength],name="Sentence_Length")

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.RNNUnitNum)

        initState = cell.zero_state(batch_size=self.BatchSize,dtype=tf.float32)

        def loopOpt(i,state,output):
            lastWord = SentenceVector[:,i]
            atten = self.get_attention(q=lastWord,k=KeyWordVector)
            cellInput = tf.concat([lastWord,atten],axis=-1)
            cell.call(state,cellInput)
            i = i + 1



class Main:
    def __init__(self):
        pass

class Data:
    def __init__(self,**kwargs):
        self.SourceFile = 'DP.txt'
        self.TaskName = 'DP'
        self.Name = 'DP_gen'
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000
        self.FlagNum = 60
        self.TopicNum = 30
        self.KeyWordNum = 5
        self.VecSize = 300
        self.MaxSentenceSize = 100
        for k in kwargs:
            self.__setattr__(k,kwargs[k])
        self.Dict  = DictFreqThreshhold(DictName = self.DictName,DictSize = self.DictSize)
        self.WordVectorMap = WordVec(**kwargs)
        lda = LDA.LDA_Train(TaskName = self.TaskName,SourceFile = self.SourceFile,DictName = self.DictName)
        self.LdaMap = lda.getLda()
        self.DictSize = self.Dict.DictSize
        self.get_word_mat()
    def get_key_word(self,line,num):
            line = line[:]
            random.shuffle(line)
            num = min(num,len(line))
            return line[:num]
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
    def pipe_data(self):
        sourceFile = open(self.SourceFile, 'r', encoding='utf-8')
        for line in sourceFile:
            line = line.strip()
            words = line.split(' ')
            wordVecList = []
            wordVecList.append(np.zeros([self.VecSize],np.float32))
            wordList = []
            ref_word = self.get_key_word(words, self.KeyWordNum)
            ref_word = {k: self.WordVectorMap.get_vec(k) for k in ref_word}
            for word in words:
                currentWordId, flag = self.Dict.get_id_flag(word)
                if currentWordId < 0:
                    # 生成单词不从自定义单词表中选择的情况
                    currentWordId = random.randint(0, self.DictSize - 1)
                    topic = self.LdaMap[currentWordId]
                    flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    word = self.Dict.N2GRAM[currentWordId]
                wordVec = self.WordVectorMap.get_vec(word)
                wordVecList.append(wordVec)
                wordList.append(currentWordId)
            if len(wordList)>100:
                wordList = wordList[:100]
            else:
                I = 100-len(wordList)
                for i in range(I):
                    wordList.append(0)
                    wordVecList.append(np.zeros([self.VecSize],np.float32))

            wordVecList = wordVecList[:100]
