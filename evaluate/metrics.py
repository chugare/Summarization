from nltk.translate.bleu_score import sentence_bleu
import json
import rouge
import jieba
from util import Tool


def BLEU(fname):
    score_sum = 0
    dfile = open(fname,'r',encoding='utf-8')
    data = json.load(dfile)
    count = 0
    for d in data:
        gen = d['gen']
        gen = [i for i in gen]
        ref = d['ref']
        ref = [[i for i in ref]]
        score = sentence_bleu(ref,gen,weights=(1.0,))
        # print(score)
        count += 1
        score_sum += score
    return score_sum/count


def ROUGE(fname):
    score_sum = 0
    dfile = open(fname,'r',encoding='utf-8')
    data = json.load(dfile)
    count = 0
    r = rouge.Rouge()
    sum = None
    for d in data:
        gen = d['gen']
        gen = ' '.join([i for i in gen])
        # gen = [[i for i in gen]]
        ref = d['ref']
        ref = ' '.join([i for i in ref])

        # ref = [[i for i in ref]]



        score = r.get_scores([gen],[ref])[0]
        if sum is None:
            sum = score
        else:
            for t in score:
                for i in score[t]:
                    sum[t][i] += score[t][i]

        print(score)
        count += 1
    avg = {}
    for t in sum:
        avg[t] = {}
        for r in sum[t]:
            avg[t][r] = float(sum[t][r]) / count
        print("%s %f" % (t, avg[t]['r']))
    return score_sum/count


def WordNum(gen):
    words = jieba.lcut(gen)
    word_dic = {}
    for w in words:
            word_dic[w] = word_dic.get(w,0)+1
    return len(word_dic)


def IdfAverage(gen):
    tfidf = Tool.Tf_idf()
    words = jieba.lcut(gen)
    idfsum = 0
    c = 1
    for w in words:
        if w in tfidf.idf:
            idfsum += tfidf.idf[w]
            c += 1
    return idfsum/c


def generate_ALL(fname):
    score_sum = 0
    print(fname)
    dfile = open(fname, 'r', encoding='utf-8')
    data = json.load(dfile)
    count = 0
    r = rouge.Rouge()
    rougeSum = None
    BLEUSum = 0
    wordNumSum = 0
    wordIdfSum = 0.0
    for d in data:
        try:
            gen = d['gen']
            wordCount = WordNum(gen)
            wordIdf = IdfAverage(gen)
            wordNumSum += wordCount
            wordIdfSum += wordIdf
            gen = ' '.join([i for i in gen])
            ref = d['ref']
            ref = ' '.join([i for i in ref])
            score = r.get_scores([gen], [ref])[0]
            if rougeSum is None:
                rougeSum = score
            else:
                for t in score:
                    for i in score[t]:
                        rougeSum[t][i] += score[t][i]

            gen = d['gen']
            gen = [i for i in gen]
            ref = d['ref']
            ref = [[i for i in ref]]
            score = sentence_bleu(ref, gen, weights=(1.0,))
            BLEUSum += score
            count+=1
        except Exception as e:
            print("%s"%e.args[0])


    rougeAvg = {}
    result = []
    for t in rougeSum:
        rougeAvg[t] = {}
        for r in rougeSum[t]:
            rougeAvg[t][r] = float(rougeSum[t][r]) / count
        result.append(rougeAvg[t]['r'])
        # print("%f" % (rougeAvg[t]['r']))
    BLEUAvg = BLEUSum/ count
    wordIdfAvg = wordIdfSum/count
    wordNumAvg = wordNumSum/count
    result.extend([BLEUAvg,wordNumAvg,wordIdfAvg])
    res = [str(s) for s in result]
    print('\t'.join(res))

    reslog = open('reslog.txt','a',encoding='utf-8')
    reslog.write(fname+'\t'+'\t'.join(res)+'\n')
    # print('%f'%BLEUAvg)
    # print('%f'%wordNumAvg)
    # print('%f'%wordIdfAvg)
    # print('BLEU %f'%BLEUAvg)
    # print('WordNUM %f'%wordNumAvg)
    # print('WordIDF %f'%wordIdfAvg)


if __name__ == '__main__':

    # generate_ALL('../baseline/lead.json')
    # generate_ALL('../baseline/covage.json')
    # generate_ALL('../baseline/tfidf.json')
    # generate_ALL('../baseline/textrank.json')
    # generate_ALL('../Estimator_edition/tfm.json')
    # generate_ALL('../Estimator_edition/tfmNobeam.json')
    # generate_ALL('../Estimator_edition/s2s.json')
    # generate_ALL('../Estimator_edition/abs.json')
    # generate_ALL('../Estimator_edition/absNobeam.json')
    # generate_ALL('../Estimator_edition/absnrwbeam.json')
    # generate_ALL('../Estimator_edition/absBeamplus.json')
    # generate_ALL('../Estimator_edition/lstmbeam+.json')

    # generate_ALL('../Estimator_edition/gru1layernobeam.json')
    # generate_ALL('../Estimator_edition/gru1layerbeam.json')
    # generate_ALL('../Estimator_edition/lstm1layerbeamplus.json')




    # generate_ALL('../Estimator_edition/lstm1layernobeam.json')
    # generate_ALL('../Estimator_edition/lstm1layerbeam.json')
    # generate_ALL('../Estimator_edition/lstm1layerbeamplus.json')

    # generate_ALL('../Estimator_edition/lstmnr1layernobeam.json')
    # generate_ALL('../Estimator_edition/lstmnr1layerbeam.json')
    # generate_ALL('../Estimator_edition/lstmnr1layerbeamplus.json')


    # generate_ALL('../Estimator_edition/tfm1layerNobeam.json')
    # generate_ALL('../Estimator_edition/tfm1layerbeamplus.json')
    # generate_ALL('../Estimator_edition/tfm1layerbeam.json')

    # generate_ALL('../Estimator_edition/tfmnrw1layerbeam.json')
    # generate_ALL('../Estimator_edition/tfmnrw1layernpbeam.json')
    # generate_ALL('../Estimator_edition/tfmnrw1layerbeamplus.json')

    # 重复惩罚

    # generate_ALL('../Estimator_edition/absNrwNobeamSrp.json')
    # generate_ALL('../Estimator_edition/absRwNobeamSrp.json')
    # generate_ALL('../Estimator_edition/absNrwNobeamWrp.json')
    # generate_ALL('../Estimator_edition/absNrwBeamWrp.json')
    # generate_ALL('../Estimator_edition/absRwNobeamWrp.json')
    # generate_ALL('../Estimator_edition/absRwNobeamErp.json')
    # generate_ALL('../Estimator_edition/absNrweamErp.json')


    generate_ALL('../Estimator_edition/tfmlcsts.json')
    # generate_ALL('../Estimator_edition/tfmnrw1layerbeamplus.json')
    # generate_ALL('../Estimator_edition/tfmnrw1layerbeamplus.json')
    # generate_ALL('../Estimator_edition/tfmnrw1layerbeamplus.json')
