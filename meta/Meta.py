
class Meta:
    def __init__(self,**kwargs):
        self.KeyWordNum = 5
        self.VecSize = 300
        self.ContextLen = 10
        self.HiddenUnit = 800
        self.RNNUnitNum = 800

        self.KernelSize = 5
        self.KernelNum = 800
        self.TopicNum = 30
        self.FlagNum = 60
        self.TopicVec = 10
        self.FlagVec = 20
        self.ContextVec = 400
        self.WordNum = 80000
        self.BatchSize = 128
        self.L2NormValue = 0.02
        self.DropoutProb = 0.7
        self.GlobalNorm = 0.5
        self.LearningRate = 0.001
        self.HidderLayer = 3
        self.LRDecayRate = 0.8

        self.SourceFile = 'DP.txt'
        self.TaskName = 'DP'
        self.Name = 'DP_gen'
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000

        self.passes = 1
        self.numTopic = 30

        self.Epoch = 10
        self.EpochSize = 100000

        self.ReadNum = 10
        self.LogInterval = 10

        self.EvalCaseNum = 40


        self.MaxSentenceLength = 100
        self.BeamSize = 5
        for k in kwargs:
            self.__setattr__(k,kwargs[k])

    @staticmethod
    def get_meta(meta_file_name):
        m_file = open('MODEL.meta','r')
        result = {}
        for line in m_file:
            kv = line.strip().split('\t')
            if len(kv)<2:
                continue
            result[kv[0]] = kv[1]

        int_keys = ['KeyWordNum', 'VecSize', 'ContextLen', 'HiddenUnit', 'RNNUnitNum', 'KernelSize', 'KernelNum', 'TopicNum',
         'FlagNum', 'TopicVec', 'FlagVec', 'ContextVec', 'WordNum', 'BatchSize',
         'DictSize', 'passes', 'numTopic', 'Epoch', 'EpochSize', 'HidderLayer','ReadNum', 'LogInterval', 'EvalCaseNum',
         'MaxSentenceLength', 'BeamSize']

        float_keys = ['L2NormValue', 'DropoutProb',
         'GlobalNorm', 'LearningRate',  'LRDecayRate']

        str_keys = [ 'SourceFile', 'TaskName', 'Name', 'DictName']

        for k in int_keys:
            if k in result:
                result[k] = int(result[k])
        for k in float_keys:
            if k in result:
                result[k] = float(result[k])
        return result

    @staticmethod
    def get_meta_comb(meta_file_names):
        result = {}
        for meta_file_name in meta_file_names:
            m_file = open(meta_file_name, 'r')
            for line in m_file:
                kv = line.strip().split('\t')
                if len(kv) < 2:
                    continue
                result[kv[0]] = kv[1]

        int_keys = ['KeyWordNum', 'VecSize', 'ContextLen', 'HiddenUnit', 'RNNUnitNum', 'KernelSize', 'KernelNum',
                    'TopicNum',
                    'FlagNum', 'TopicVec', 'FlagVec', 'ContextVec', 'WordNum', 'BatchSize',
                    'DictSize', 'passes', 'numTopic', 'Epoch', 'EpochSize', 'HidderLayer', 'ReadNum', 'LogInterval',
                    'EvalCaseNum',
                    'MaxSentenceLength', 'BeamSize']

        float_keys = ['L2NormValue', 'DropoutProb',
                      'GlobalNorm', 'LearningRate', 'LRDecayRate']

        str_keys = ['SourceFile', 'TaskName', 'Name', 'DictName']

        for k in int_keys:
            if k in result:
                result[k] = int(result[k])
        for k in float_keys:
            if k in result:
                result[k] = float(result[k])
        return result
    def dump(self):
        m = open('MODEL.meta','w')
        for k in self.__dict__:
            m.write("%s\t%s\n"%(k,self.__dict__[k]))

    def print(self):
        keys = []
        for k in self.__dict__:
            keys.append(k)
        print(keys)


print(r)