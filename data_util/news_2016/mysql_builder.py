from data_util.data_source import DatasetBuilder
from data_util.news_2016.db_util import MysqlWriter
import json
import re
from baseline.TextRank import textRank_s

class NewsDatasetMysqlWriter(DatasetBuilder):
    def read(self):
        count = 0
        for line in self.source:
            obj = json.loads(line)
            for k in obj:
                obj[k] = re.sub('[\'\"%$#@()&]','',obj[k])

            yield count, obj
            count+=1
    def write_mysql(self,MAX = -1):
        gen = self.read()
        max_length = 1000
        min_length = 50

        MR = MysqlWriter(buffer_size=2)
        count = 0
        i = 0
        for _, obj in gen:
            i += 1
            if MAX>0 and i>=MAX:
                break

            if len(obj['content']) > max_length or len(obj['content']) < min_length:
                continue
            count += 1
            if i % 100 == 0:
                print("[INFO] Now reading Line : %d ,write line %d" % (i, count))

            MR.write((obj.get('title',''),obj.get('source',''),obj.get('content','')))
        MR.close()
