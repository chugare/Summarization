from data_util.data_source import DatasetBuilder
from data_util.news_2016.db_util import MysqlWriter
import json

class NewsDatasetMysqlWriter(DatasetBuilder):
    def read(self):
        count = 0
        for line in self.source:
            obj = json.loads(line)
            obj['content'].replace("#",'')
            obj['desc'].replace("#",'')
            obj['title'].replace("#",'')
            yield count, obj['title']+'#'+obj['desc']+'#'+obj['content']
            count+=1
    def write_mysql(self):
        gen = self.read()
        max_length = 1000
        min_length = 50

        MR = MysqlWriter()
        count = 0
        i = 0
        for _, sentence in gen:
            i += 1
            if i % 1000 == 0:
                print("[INFO] Now reading Line : %d ; write %d line" % (i,count))
            if len(sentence)>min_length and len(sentence)<max_length:
                count +=1
                MR.write(self.cut_with_comma(sentence))
        MR.close()


def