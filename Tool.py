import xml.etree.ElementTree as ET
import os
import sys
import re
import jieba
import jieba.posseg as pseg
import math
import json
import random
import time

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
                if count == 100:
                    break
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
    pos_file = open('POS.txt','w',encoding='utf-8')
    count = 0
    for w in dic:
        if dic[w]>20:
            dic_file.write("%d %s\n"%(count,w))
            for wf in dic_pos[w]:
                pos_file.write("%s %d,"%(wf,dic_pos[w][wf]))
            pos_file.write('\n')
        count+=1

    dic_file.close()
    pos_file.close()

    print("[INFO] 信息提取完毕，总共提取文书%d篇 句子长度统计如下"%count)


    # for kv in enumerate(length_map):

    ll = sorted(length_map.keys(),key = lambda x:x)
    for k in ll:
        print("k = %d : %d"%(k,length_map[k]))


    data_file.close()

if __name__ == '__main__':
    arg = sys.argv
    XML2TXT_extract(arg[1])