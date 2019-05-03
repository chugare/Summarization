
class Meta:
    def __init__(self,**kwargs):
        self.KeyWordNum = 5
        self.VecSize = 300
        self.ContextLen = 10
        self.HiddenUnit = 800
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

        self.ReadNum = 10000
        self.LogInterval = 10

        self.EvalCaseNum = 40
        for k in kwargs:
            self.__setattr__(k,kwargs[k])

    def get_meta(self):
        return self.__dict__