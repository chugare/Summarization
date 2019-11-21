import sys
sys.path.append("/home/user/zsm/Summarization/")
from data_util.data_source import NewsDatasetBuilder




if __name__ == '__main__':
    l = NewsDatasetBuilder(open("/home/data/news2016.json",'r',encoding='utf-8'), "NEWS")
    l.build_dataset()