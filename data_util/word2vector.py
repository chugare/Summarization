from util.redis_util import save_value,get_value,init
import numpy as np
import sys, os, re, jieba
import time
from util.redis_util import init,save_value,get_value
class WordVec:
    def __init__(self,**kwargs):
        self.vec_dic = {}
        self.word_list = []
        self.vec_list = []
        self.num = 0
        self.ReadNum = -1
        self.TaskName = ''
        self.SourceFile = ''
        self.VecFile = ""
        self.VecSize = 300
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def init_wv_redis(self):

        print('[INFO] Start load word vector')
        st = time.time()
        init()
        vec_file = open('/home/user/zsm/data/sgns.merge.char','r',encoding='utf-8')
        count = 0
        vec_dict = {}
        for l in vec_file:

            m = l.strip().split(' ')
            w = m[0]
            try:
                vec =[float(v) for v in m[1:]]
            except ValueError:
                vec =[float(v.replace(')','-')) for v in m[1:]]
            count+=1
            vec_dict[w] = vec
            if count%10000 == 0:
                save_value(vec_dict)
                nt = time.time()
                sys.stdout.write('\r[INFO] Load vec data, %d finished, time cost %.3f' % (count, nt-st))
                vec_dict = {}
            if count == self.ReadNum:
                break
        save_value(vec_dict)
        nt = time.time()
        print('\n[INFO] Vec data loaded, total time cost %.3f'%(nt-st))
        self.num = count


    def SimplifiedByText(self,Name,wordlist):
        file = open('%s.char'%Name,'w',encoding='utf-8')
        # file.write(str(self.num)+' 300\n')
        print('[INFO] Now building simplified word vector dictionary, totally word %d'%len(wordlist))
        count = 0
        for w in wordlist:
            if w in self.vec_dic:
                vec_f = [str(i) for i in self.vec_dic[w]]
                vec_str = ' '.join(vec_f)
                file.write(w+' '+vec_str+'\n')
                count += 1
        file.close()
        print('[INFO] Simplified word vector has been built totally word %d'%count)



    @staticmethod
    def ulw(word):
        pattern = [
            # r'[,.\(\)（），。\-\+\*/\\_|]{2,}',
            r'\d+',
            r'[qwertyuiopasdfghjklzxcvbnm]+',
            r'[ｑｗｅｒｔｙｕｉｏｐａｓｄｆｇｈｊｋｌｚｘｃｖｂｎｍ]+',
            r'[QWERTYUIOPASDFGHJKLZXCVBNM]+',
            r'[ＱＷＥＲＴＹＵＩＯＰＡＳＤＦＧＨＪＫＬＺＸＣＶＢＮＭ]+',
            r'[ⓐ ⓑ ⓒ ⓓ ⓔ ⓕ ⓖ ⓗ ⓘ ⓙ ⓚ ⓛ ⓜ ⓝ ⓞ ⓟ ⓠ ⓡ ⓢ ⓣ ⓤ ⓥ ⓦ ⓧ ⓨ ⓩ]+',
            r'[Ⓐ Ⓑ Ⓒ Ⓓ Ⓔ Ⓕ Ⓖ Ⓗ Ⓘ Ⓙ Ⓚ Ⓛ Ⓜ Ⓝ Ⓞ Ⓟ Ⓠ Ⓡ Ⓢ Ⓣ Ⓤ Ⓥ Ⓦ Ⓧ Ⓨ Ⓩ ]+',
        ]
        ulwf = open('uslw.txt','a',encoding='utf-8')
        for p in pattern:
            mr = re.match(p,word)
            if mr is not None:
                ulwf.write(word+'\n')
                ulwf.close()
                return True
        return  False

    def clear_ulw(self):
        vec_file = open(self.VecFile,'r',encoding='utf-8')
        meg = next(vec_file).split(' ')
        num = int(meg[0])
        file = open(self.VecFile, 'w', encoding='utf-8')

        count = 0

        for l in vec_file:
            m = l.strip().split(' ')
            w = m[0]
            if WordVec.ulw(w):
                continue
            count+=1
            if count%10000 == 0:
                p = float(count)/num*100
                sys.stdout.write('\r[INFO] write cleared vec data, %d finished'%count)
            file.write(l)
            # vec_dic[w] = vec
        print('\n Final count : %d'%count)


    def sen2vec(self,sen):
        sen = jieba.lcut(sen)
        tmp_dict = get_value(sen)
        vec_out = []
        for w in sen:
            if tmp_dict[w]:
                vec_out.append(tmp_dict[w])
            else:
                vec_out.append(np.zeros([self.VecSize],np.float32))
        return vec_out
    def get_vec(self,word):
        try:
            return get_value([word])
        except KeyError:
            return np.zeros([self.VecSize])