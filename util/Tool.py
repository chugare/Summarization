import xml.etree.ElementTree as ET
import os
import sys
import re
import jieba.posseg as pseg
import math
import json
import time
from data_util.word2vector import WordVec
import setting
thershold = 5

def get_name(basename):
    t = time.localtime()
    v = time.strftime("%y%m%d%H%M%S",t)
    return basename+'_'+v
def XML2TXT_extract(root_dic,dis_file=None):
    fl = os.listdir(root_dic)
    if dis_file == None:
        dis_file = get_name('DOC_SEG')+".txt"
    data_file = open(dis_file,'w',encoding='utf-8')
    count = 0
    fl = [root_dic+'/'+f for f in fl]
    length_map = {}
    dic = {}
    dic_pos = {}
    for file in fl:
        if not file.endswith('.xml'):
            try:
                fl_a = os.listdir(file)
                fl_a = [file+'/'+f for f in fl_a]
                fl += fl_a
            except Exception:
                pass
        else:
            try:
                if (count+1) % 100 == 0:
                    print("[INFO] Now reading file : %d "%(count+1))
                stree = ET.ElementTree(file = file)
                qw = next(stree.iter('QW')).attrib['value']
                sens = re.split(r"[,、，。；：\n]",qw)
                patterns = [
                    r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
                    r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
                    # r"[\s]",
                    r"[a-zA-Z《》【】（）\s]+",
                ]
                res = []
                for sen in sens:
                    for p in patterns:
                        sen = re.sub(p,'',sen)
                    if len(sen)<3:
                        continue
                    cutres = pseg.lcut(sen)
                    for w in cutres:
                        wc = w.word
                        wf = w.flag
                        if wc not in dic:
                            dic[wc] = 0
                            dic_pos[wc] = {}
                        dic[wc] += 1
                        if wf not in dic_pos[wc]:
                            dic_pos[wc][wf] = 0
                        dic_pos[wc][wf]  += 1
                    lc = len(cutres)
                    if lc not in length_map:
                        length_map[lc]=0
                    length_map[lc] += 1
                    cutres = list(zip(*cutres))
                    sen = ' '.join(list(cutres[0]))
                    res.append(sen)
                data_file.write(' '.join(res))
                data_file.write('\n')
            except StopIteration:
                pass
            #
            # if count>10:
            #     break
            count += 1
    dic_file = open('DICT.txt','w',encoding='utf-8')
    # pos_file = open('POS.txt','w',encoding='utf-8')
    count = 0
    for w in dic:
        if dic[w]>thershold:
            word_type = max(dic_pos[w],key = lambda x:dic_pos[w][x])
            dic_file.write("%d %s %s\n"%(count,w,word_type))
            count+=1
    dic_file.close()
    # pos_file.close()
    print("[INFO] 信息提取完毕，总共提取文书%d篇 句子长度统计如下"%count)
    # for kv in enumerate(length_map):
    ll = sorted(length_map.keys(),key = lambda x:x)
    for k in ll:
        print("k = %d : %d"%(k,length_map[k]))
    data_file.close()

class Tf_idf:
    def __init__(self,dic=None,doc_file=None):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.idf = {}
        self.FileName = doc_file
        try:
            _data_file = open(setting.DATA_PATH+'_tfidf.json','r',encoding='utf-8')
            _data_t = json.load(_data_file)
            self.GRAM2N = _data_t['G']
            self.N2GRAM = {int(k):_data_t['N'][k] for k in _data_t['N']}
            self.idf = _data_t['I']
        except Exception:
            if dic is None or doc_file is None:
                print('[ERROR] Require data file to initialize')
                return
            dic_file = open(dic,'r',encoding='utf-8')
            for line in dic_file:
                wd = line.split(' ')
                self.GRAM2N[wd[1]] = int(wd[0].strip())
                self.N2GRAM[int(wd[0].strip())] = wd[1]
                self.idf[wd[1]] = 0.0
            ga = self.read_doc_all()
            self.idf_calc(ga)
            _data_file = open(setting.DATA_PATH+'_tfidf.json','w',encoding='utf-8')
            obj = {
                'G':self.GRAM2N,
                'N':self.N2GRAM,
                'I':self.idf
            }
            json.dump(obj,_data_file,ensure_ascii=False)

    def idf_calc(self,doc_gen):
        # doc_data = json.load(self.doc_file)
        doc_num = 0.0
        print('[INFO] Start calc idf')
        for doc in doc_gen:
            tmp_idf = {}
            doc_num+=1
            for w in doc:
                if w not in tmp_idf:
                    tmp_idf[w] = 1

            for w in tmp_idf:
                if w in self.idf:
                    self.idf[w] += 1
            if int(doc_num) % 10000 == 0:
                print('[INFO] %d of doc read'%doc_num)
        print('[INFO] All docs have been read')
        for w in self.idf:
            self.idf[w] = math.log(doc_num/(self.idf[w]+1))
        print('[INFO] All idf value have been calculated')
    def tf_calc(self,sen):
        # tf = np.zeros(shape=[len(self.N2GRAM)])
        tf = {}
        l = len(sen)
        for word in sen:
            # wid = self.GRAM2N[word]
            if word not in tf:
                tf[word] = 0.0
            tf[word] = (tf[word]+1.0)
        tf_idf = {}
        for k in tf:
            try:
                idf = self.idf[k]
                tf_idf[k] = tf[k]*(idf**2)
            except KeyError:
                pass
        return  tf_idf
    def get_top_word(self,vec_l,k):
        vt_l = []
        for i in vec_l:
            vt_l.append((i,vec_l[i]))
        vt_l.sort(key=lambda x:x[1],reverse=True)
        res = []
        for i in range(k):
            try:
                w = vt_l[i][0]
                res.append(w)
            except IndexError:
                # print(vt_l)
                # print('')
                res.append(0)
        return  res
    def read_doc_all(self):
        file_all = open(self.FileName,'r',encoding='utf-8')
        for line in file_all:
            yield line.strip().split(' ')

    @staticmethod
    def read_doc_case(fname):
        file_all = open(fname,'r',encoding='utf-8')
        data_all = json.load(file_all)
def _ref():
    file = open('DP_lite.txt','r',encoding='utf-8')
    nfile = open('rDP_lite.txt','w',encoding='utf-8')
    for line in file:
        nl = re.sub(r'\s+',' ',line)
        # nl = line.replace('   ',' ')
        nfile.write(nl+'\n')

if __name__ == '__main__':
    arg = sys.argv
    mod = arg[1]
    fileName = arg[2]
    taskName = arg[3]
    thershold = int(arg[4])

    # _ref()
    try:
        readNum = int(arg[5])
    except IndexError:
        readNum = -1
    if mod == '-t':
        TXT2TXT_extract(fileName,taskName,testCase = readNum)
    elif mod == '-x':
        XML2TXT_extract(fileName)
    elif mod == '-tl':
        TXT2TXT_extract(fileName,"DP_lite",testCase = -1)

    elif mod == '-tf':
        tfidf = Tf_idf(dic='%s_DICT.txt'%taskName,doc_file='%s.txt'%taskName)
