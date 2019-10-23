import re
import jieba.posseg as pseg
from data_util.word2vector import WordVec

def TXT2TXT_extract(sourceFile, TaskName, dis_file=None,min_line = 50,
                    evalSize=1000,threshold = 20):
    sourceFile = open(sourceFile, 'r', encoding='utf-8')
    if dis_file == None:
        dis_file = TaskName + ".txt"
    data_file = open(dis_file, 'w', encoding='utf-8')
    eval_file = open('E_' + dis_file, 'w', encoding='utf-8')

    countFile = 0
    length_map = {}
    dic = {}
    dic_pos = {}

    def cut_without_comma(commentLine):
        commentLine = commentLine.replace('\n', '')
        sens = re.split(r"[,、，。；：\n]", commentLine)
        patterns = [
            r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
            r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
            # r"[\s]",
            r"[a-zA-Z《》【】（）\s]+",
        ]
        res = []
        for sen in sens:
            for p in patterns:
                sen = re.sub(p, '', sen)
            if len(sen) < 3:
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
                dic_pos[wc][wf] += 1
            lc = len(cutres)
            if lc not in length_map:
                length_map[lc] = 0
            length_map[lc] += 1
            cutres = list(zip(*cutres))
            sen = ' '.join(list(cutres[0]))
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

            cutres = pseg.lcut(commentLine)
            for w in cutres:
                wc = w.word
                wf = w.flag
                if wc not in dic:
                    dic[wc] = 0
                    dic_pos[wc] = {}
                dic[wc] += 1
                if wf not in dic_pos[wc]:
                    dic_pos[wc][wf] = 0
                dic_pos[wc][wf] += 1
            lc = len(cutres)
            if lc not in length_map:
                length_map[lc] = 0
            length_map[lc] += 1
            cutres = list(zip(*cutres))
            return ' '.join(list(cutres[0]))

    ecount = 0

    def cut_with_comma_sen(commentLine):

        patterns = [
            r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
            r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
            # r"[\s]",
            r"[a-zA-Z《》【】（）\s]+",
        ]
        for p in patterns:
            commentLine = re.sub(p, '', commentLine)
        sens = re.split(r"[,、，。；：\n]", commentLine)
        res = []
        for sen in sens:
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
                dic_pos[wc][wf] += 1
            lc = len(cutres)
            if lc not in length_map:
                length_map[lc] = 0
            length_map[lc] += 1
            cutres = list(zip(*cutres))
            res.append(' '.join(list(cutres[0])))
        return '\n'.join(res)


    for line in sourceFile:
        line = line.strip()
        # if len(line) != 0:
        #     commentLine += line
        # else:
        #     if len(commentLine)!=0:
        #         try:
        commentLine = line
        commentLine = commentLine.replace('\n', '')
        if len(commentLine) < min_line:
            continue
        countFile += 1
        if (countFile) % 1000 == 0:
            print("[INFO] Now reading Line : %d " % (countFile))
        if ecount < 1000:
            ecount += 1
            eval_file.write(cut_with_comma(commentLine))
            eval_file.write('\n')
        else:
            data_file.write(cut_with_comma(commentLine))
            data_file.write('\n')
            #     except StopIteration:
            #         pass
            # commentLine = ""
    dic_file = open(TaskName + '_DICT.txt', 'w', encoding='utf-8')
    # pos_file = open('POS.txt','w',encoding='utf-8')
    count = 0
    ULSW = ['\n', '\t', ' ', '']
    for i in ULSW:
        dic[i] = 0
    for w in dic:
        if dic[w] > threshold:
            word_type = max(dic_pos[w], key=lambda x: dic_pos[w][x])
            wordCount = dic[w]
            dic_file.write("%d %s %s %d\n" % (count, w, word_type, wordCount))
            count += 1
    dic_file.close()
    # pos_file.close()
    print("[INFO] 点评文本读取完毕 共计%d 文本 句子长度统计如下" % count)
    # for kv in enumerate(length_map):
    ll = sorted(length_map.keys(), key=lambda x: x)
    for k in ll:
        print("k = %d : %d" % (k, length_map[k]))
