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
thershold = 20

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
def TXT2TXT_extract(sourceFile,TaskName,dis_file = None,testCase = -1,
                    evalSize = 1000):
    sourceFile = open(sourceFile,'r',encoding='utf-8')
    if dis_file == None:
        dis_file = TaskName+".txt"
    data_file = open(dis_file,'w',encoding='utf-8')

    commentLine = ""
    countFile = 0
    length_map = {}
    dic = {}
    dic_pos = {}
    for line in sourceFile:
        line = line.strip()
        # if len(line) != 0:
        #     commentLine += line
        # else:
        #     if len(commentLine)!=0:
        #         try:
        commentLine = line
        countFile += 1
        if (countFile) % 100 == 0:
            print("[INFO] Now reading Line : %d "%(countFile))
        if countFile == testCase:
            break
        def cut_without_comma(commentLine):
            commentLine = commentLine.replace('\n',' ')
            sens = re.split(r"[,、，。；：\n]",commentLine)
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
                data_file.write(sen)
                res.append(sen)
                return ' '.join(res)

        def cut_with_comma(commentLine):
            patterns = [
                r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
                r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
                # r"[\s]",
                r"[a-zA-Z《》【】（）\s]+",
            ]
            for p in patterns:
                commentLine = re.sub(p, '', commentLine)
            
        data_file.write(cut_without_comma(commentLine))
        data_file.write('\n')
            #     except StopIteration:
            #         pass
            # commentLine = ""
    dic_file = open(TaskName+'_DICT.txt','w',encoding='utf-8')
    # pos_file = open('POS.txt','w',encoding='utf-8')
    count = 0
    for w in dic:
        if dic[w]>thershold:
            word_type = max(dic_pos[w],key = lambda x:dic_pos[w][x])
            dic_file.write("%d %s %s\n"%(count,w,word_type))
            count+=1
    dic_file.close()
    # pos_file.close()
    print("[INFO] 点评文本读取完毕 共计%d 文本 句子长度统计如下"%count)
    # for kv in enumerate(length_map):
    ll = sorted(length_map.keys(),key = lambda x:x)
    for k in ll:
        print("k = %d : %d"%(k,length_map[k]))
def SeperateSet():

if __name__ == '__main__':
    arg = sys.argv
    mod = arg[1]
    fileName = arg[2]
    if mod == '-t':
        TXT2TXT_extract(fileName,"DP")
    elif mod == '-x':
        XML2TXT_extract(fileName)