from data_util.data_source import DatasetBuilder,SegFileBatchWriter,BatchWriter
import json



class NewsDatasetBuilder(DatasetBuilder):

    def read(self):
        count = 0
        for line in self.source:
            obj = json.loads(line)
            obj['content'].replace("#",'')
            obj['desc'].replace("#",'')
            obj['title'].replace("#",'')
            yield count, obj['title']+'#'+obj['desc']+'#'+obj['content']
            count+=1
    def build_dataset(self):

        eval_file = open('E_' + self.dis_file, 'w', encoding='utf-8')
        gen = self.read()
        e_c = 0
        max_length = 1000
        min_length = 50

        BR1 = SegFileBatchWriter(fname=self.TaskName)
        BR2 = BatchWriter(eval_file)
        count = 0
        i = 0
        for _, sentence in gen:
            i += 1
            if i % 1000 == 0:
                print("[INFO] Now reading Line : %d ; write %d line" % (i,count))
            if len(sentence)>min_length and len(sentence)<max_length:
                count +=1
                if e_c < self.evalSize:
                    e_c += 1
                    BR2.write(self.cut_with_comma(sentence))
                else:
                    BR1.write(self.cut_with_comma(sentence))
        BR1.close()
        BR2.close()

        dic_file = open(self.TaskName + '_DICT.txt', 'w', encoding='utf-8')
        BRD = BatchWriter(dic_file)
        # pos_file = open('POS.txt','w',encoding='utf-8')
        count = 2
        ULSW = ['\n', '\t', ' ', '']
        for i in ULSW:
            self.dic[i] = 0
        self.dic = sorted(self.dic.items(),key=lambda i:i[1],reverse=True)
        # BRD.write("%d %s %s %d" % (0, "<PAD>", "x", 0))
        # BRD.write("%d %s %s %d" % (1, "<EOS>", "x", 0))

        for w , wordCount in self.dic:
            if wordCount > self.threshold:
                word_type = max(self.dic_pos[w], key=lambda x: self.dic_pos[w][x])
                BRD.write("%d %s %s %d" % (count, w, word_type, wordCount))
                count += 1
        BRD.close()
        # pos_file.close()
        print("[INFO] 读取完毕 共计 %d 单词 句子长度统计如下" % count)
        # for kv in enumerate(length_map):
        ll = sorted(self.length_map.keys(), key=lambda x: x)
        for k in ll:
            print("%d : %d" % (k, self.length_map[k]))