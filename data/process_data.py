import sys
sys.path.append("/home/user/zsm/Summarization/")
from data_util.data_source import DatasetBuilder


class LenghtGapDataset(DatasetBuilder):
    def __init__(self, sourceFile, TaskName, dis_file=None, min_line=50,
                 evalSize=1000, threshold=5):
        super(LenghtGapDataset, self).__init__(sourceFile, TaskName, dis_file, min_line,
                                               evalSize, threshold)

    def build_dataset(self):
        gen = self.read()
        BRS = []
        for i in range(7):
            fp = open("%d_%d.txt" % (50 + i * 50, 100 + i * 50), 'w', encoding='utf-8')
            BRS.append(self.BatchWriter(fp))

        for i, sentence in gen:
            if i % 1000 == 0:
                print("[INFO] Now reading Line : %d " % (i))
            k = int(len(sentence) / 50)
            if 0 < k < 7:
                seg_sen = self.cut_with_comma(sentence)
                BRS[k].write(seg_sen)
        for i,BR in enumerate(BRS):
            BR.close()
            print("%d_%d.txt 数据量 %d" % (50 + i * 50, 100 + i * 50, BR.count))


        dic_file = open(self.TaskName + '_DICT.txt', 'w', encoding='utf-8')

        BRD = self.BatchWriter(dic_file)
        count = 0
        ULSW = ['\n', '\t', ' ', '']
        for i in ULSW:
            self.dic[i] = 0
        for w in self.dic:
            if self.dic[w] > self.threshold:
                word_type = max(self.dic_pos[w], key=lambda x: self.dic_pos[w][x])
                wordCount = self.dic[w]
                BRD.write("%d %s %s %d" % (count, w, word_type, wordCount))
                count += 1
        BRD.close()
        # pos_file.close()
        print("[INFO] 点评文本读取完毕 共计%d单词 句子长度统计如下" % count)
        # for kv in enumerate(length_map):
        ll = sorted(self.length_map.keys(), key=lambda x: x)
        for k in ll:
            print("k = %d : %d" % (k, self.length_map[k]))

if __name__ == '__main__':
    l = LenghtGapDataset("/home/user/zsm/data/rating_2.txt","DP","")
    l.build_dataset()