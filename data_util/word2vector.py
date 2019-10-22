

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
        print('[INFO] Start load word vector')
        for k in kwargs:
            self.__setattr__(k,kwargs[k])

        self.VecFile = self.SourceFile.replace('.txt', '.char')
        self.VecFile = self.VecFile.replace('E_','')
        try:
            tmp = open(self.VecFile,mode='r',encoding='utf-8')

            tmp.close()
        except Exception:
            print(self.VecFile)
            self.VecFile = 'word_vec.char'

        pass
        self._read_vec()
    def dump_file(self):
        file = open('word_vec.char','w',encoding='utf-8')
        # file.write(str(self.num)+' 300\n')
        for w in self.vec_dic:
            vec_f = [str(i) for i in self.vec_dic[w]]
            vec_str = ' '.join(vec_f)
            file.write(w+' '+vec_str+'\n')
        file.close()
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
    def _read_vec(self):
        path = os.path.abspath('.')
        print(path)
        # path = '/'.join(path.split('\\')[:-1])+'/sgns.merge.char'
        # path = 'F:/python/word_vec/sgns.merge.char'
        # path = 'D:\\赵斯蒙\\EVI-fact\\word_vec.char'
        path = self.VecFile
        vec_file = open(path,'r',encoding='utf-8')
        # meg = next(vec_file).split(' ')
        # num = int(meg[0])
        # self.num = num
        count = 0
        for l in vec_file:

            m = l.strip().split(' ')
            w = m[0]
            vec = m[1:]
            try:
                vec =[float(v) for v in m[1:]]
            except ValueError:
                vec =[float(v.replace(')','-')) for v in m[1:]]
            # if WORD_VEC.ulw(w):
            #     continue
            count+=1
            if count%10000 == 0:
                sys.stdout.write('\r[INFO] Load vec data, %d finished'%count)
            if count == self.ReadNum:
                break
            self.vec_list.append(vec)
            self.word_list.append(w)
            self.vec_dic[w] = np.array(vec,dtype=np.float32)

        print('\n[INFO] Vec data loaded')
        self.num = count
    def get_min_word(self,word):
        vec = self.vec_dic[word]
        dis = cosine_similarity(self.vec_list,[vec])
        dis = np.reshape(dis,[-1])
        dis_pair = [(i,dis[i]) for i in range(len(dis))]
        dis_pair.sort(key= lambda x:x[1],reverse=True)
        for i in range(10):
            print(self.word_list[dis_pair[i][0]])

    def get_min_word_v(self, vec):
        dis = cosine_similarity(self.vec_list, [vec])
        dis = np.reshape(dis, [-1])
        i = np.argmax(dis)
        return self.word_list[i]
    def get_sentence(self,vlist,l):
        result = ''
        x = 0
        for vec in vlist:
            if x == l:
                break
            print('[INFO] Search for nearest word on index %d'%x)
            dis = cosine_similarity(self.vec_list, [vec])
            dis = np.reshape(dis, [-1])
            i = np.argmax(dis)
            x+= 1
            print(self.word_list[i])
            result += self.word_list[i]

        return result
    def sen2vec(self,sen):
        sen = jieba.lcut(sen)
        vec_out = []
        for w in sen:
            if w in self.vec_dic:
                vec_out.append(self.vec_dic[w])
        return vec_out
    def get_vec(self,word):
        try:
            return self.vec_dic[word]
        except KeyError:
            return np.zeros([len(self.vec_list[0])])