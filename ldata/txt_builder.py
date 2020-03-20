
import sys
sys.path.append('/home/user/zsm/Summarization')

from data_util.data_source import DatasetBuilder
from data_util.BatchWriter import BatchWriter,SegFileBatchWriter
import json,jieba
import random
from baseline.TextRank import textRank_s
from util.mysql_utils import connect_db

def lcsts_data():
    con = connect_db('lcsts')
    cur = con.cursor()


    try:
        cur.execute("select * from base LIMIT 300")

        res = cur.fetchall()
        con.commit()
        return res
    except Exception as e:
        print("[W] some data fetch failed,%s, process continue"%e.args[1])
    finally:
        cur.close()
        con.close()

class LCSTSDatasetFromMysql:
    def __init__(self):
        pass
    def build(self):
        BR = BatchWriter(open("short_lcsts.txt",'w',encoding='utf-8'))
        BRE = BatchWriter(open("shorte_lcsts.txt",'w',encoding='utf-8'))
        E_size = 1000
        E_c = 0
        i = 0

        res = lcsts_data()
        for line in res:
            ID,title,titlelen,content,contentlen = line
            title = jieba.lcut(title)
            content = jieba.lcut(content)
            # txtrk = textRank_s(' '.join(content))
            # title = txtrk
            source = 'LCSTS'
            if E_c < E_size:
                rs = random.randint(0,100)
                if rs <= 3:
                    BRE.write(' '.join(title)+'#'+source+'#'+' '.join(content))
                    E_c += 1
                else:
                    BR.write(' '.join(title)+'#'+source+'#'+' '.join(content))
            else:
                BR.write(' '.join(title)+'#'+source+'#'+' '.join(content))
            i += 1
            if i%100 == 0:
                print("Read from mysql %d finished"%i)
        BR.close()
        BRE.close()
if __name__ == '__main__':

    l = LCSTSDatasetFromMysql()
    l.build()

