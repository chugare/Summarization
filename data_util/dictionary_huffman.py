import json

import data_util.dictionary


class dic_huffman(data_util.dictionary.DictFreqThreshhold):

    def __init__(self):
        super(self,dic_huffman).__init__()


        self.HuffmanEncoding()
        print(max(self.N2HUFF, key=lambda k:len(self.N2HUFF[k])))
    def getHuffmanDict(self):
        maxHuffLen = len(self.N2HUFF[max(self.N2HUFF, key=lambda k: len(self.N2HUFF[k]))])
        print('Max Huff Len: %d' % maxHuffLen)
        try:
            meta_file = open('Huffman_Layer.json', 'r', encoding='utf-8')
            jsdata = json.load(meta_file)

            huffTable = jsdata[0]
            huffLabelTable = jsdata[1]
            huffLenTable = jsdata[2]
            print('[INFO] Huffman Layer Data has been read')
            return huffTable, huffLabelTable, huffLenTable
        except Exception:
            pass
        huffTable = []
        huffLabelTable = []
        huffLenTable = []
        for k in range(self.DictSize):
            # tmphuff = np.zeros(shape=[maxHuffLen],dtype=np.int32)
            tmphuff = [0] * maxHuffLen
            # tmplabel = np.zeros(shape=[maxHuffLen],dtype=np.int32)
            tmplabel = [0] * maxHuffLen
            if str(k) not in self.N2HUFF:
                huffTable.append(tmphuff)
                huffLabelTable.append(tmplabel)
                huffLenTable.append(0)
                continue
            huffman_str = self.N2HUFF[str(k)]
            tlen = len(huffman_str)
            coding = ""
            for i in range(len(huffman_str)):
                if i > 0:
                    coding = huffman_str[:i]
                tmphuff[i] = self.HUFF2LAYER[coding]
                tmplabel[i] = 0 if huffman_str[i] == '0' else 1

            huffTable.append(tmphuff)
            huffLabelTable.append(tmplabel)
            huffLenTable.append(tlen)
        meta_file = open('Huffman_Layer.json', 'w', encoding='utf-8')
        json.dump([huffTable, huffLabelTable, huffLenTable], meta_file)
        print('[INFO] Huffman Layer Data has been build')

        return huffTable, huffLabelTable, huffLenTable

    def read_word_from_Huffman(self, layersValues):

        encoding = ''
        try:
            while True:
                np = self.HUFF2LAYER[encoding]
                if layersValues[np] > 0.5:
                    encoding += '1'
                else:
                    encoding += '0'
        except KeyError:
            if encoding not in self.HUFF2N:
                print('[ERROR] 在字典中没有找到对应的哈夫曼编码 “%s”' % encoding)
                return 0
            else:
                wordId = self.HUFF2N[encoding]
                return wordId
            pass

    def HuffmanEncoding(self, forceBuild=False):
        class HuffmanNode:
            def __init__(self, val=None, word=None):
                self.right = None
                self.left = None
                self.value = val
                self.word = word
                self.huffman = ''

        Nodes = [HuffmanNode(self.N2FREQ[k], k) for k in self.N2FREQ]
        if not forceBuild:
            try:
                meta_file = open('Huffman_dic.json', 'r', encoding='utf-8')
                self.N2HUFF, self.HUFF2N, self.HUFF2LAYER, self.LAYER2HUFF = json.load(meta_file)
                print('[INFO] Huffman dictionary has been readed')
                return
            except Exception:
                pass
        if len(Nodes) < 1:
            return

        while len(Nodes) > 1:
            Nodes.sort(key=lambda node: node.value, reverse=True)
            nv = Nodes[-1].value + Nodes[-2].value
            tmpNode = HuffmanNode(nv)
            tmpNode.left = Nodes[-2]
            tmpNode.right = Nodes[-1]
            Nodes.pop(-1)
            Nodes.pop(-1)
            Nodes.append(tmpNode)
        self.N2HUFF = {}
        self.HUFF2N = {}
        rootNode = Nodes[0]
        NodeQ = [rootNode]
        c = 0
        self.HUFF2LAYER = {}
        self.LAYER2HUFF = {}

        while len(NodeQ) > 0:
            tmpNode = NodeQ[0]

            NodeQ.pop(0)
            if c % 1000 == 0:
                print('[INFO] Huffman Build %d' % c)
            if tmpNode.word is not None:
                self.N2HUFF[str(tmpNode.word)] = tmpNode.huffman
                self.HUFF2N[tmpNode.huffman] = tmpNode.word
                continue
            self.HUFF2LAYER[tmpNode.huffman] = c
            self.LAYER2HUFF[str(c)] = tmpNode.huffman
            if tmpNode.left is not None:
                tmpNode.left.huffman = tmpNode.huffman + '0'
                NodeQ.append(tmpNode.left)
            if tmpNode.right is not None:
                tmpNode.right.huffman = tmpNode.huffman + '1'
                NodeQ.append(tmpNode.right)
            c += 1
        meta_file = open('Huffman_dic.json', 'w', encoding='utf-8')
        json.dump([self.N2HUFF, self.HUFF2N, self.HUFF2LAYER, self.LAYER2HUFF], meta_file)

        # for k in self.N2HUFF:
        #     print("%s %d %s"%(self.N2GRAM[k],self.N2FREQ[k],self.N2HUFF[k]))
