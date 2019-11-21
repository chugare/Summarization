
from data_util.dictionary import DictFreqThreshhold



class DataPipe:
    def __init__(self):
        data_file = ""
        self.dictionary = DictFreqThreshhold(DictName = "NEW_DICT.txt")