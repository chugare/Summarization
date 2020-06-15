import numpy as np
from interface.BeamSearch import *
from  interface.RepeatPunish import  *
from interface.ContextTune import CTcore
import tensorflow as tf
import json
import os,time,sys


# PLUS_RATIO= 3
PLUS_RATIO= 0

class NewsPredictor(Predictor):
    def __init__(self,estimator,topk = None):
        self.topk = topk
        self.estimator = estimator


    def predict(self,source,context,con_len,topk = None):
        def input_fn():
            return {'source':tf.constant(source),'context':tf.constant(context),'con_len':tf.constant(con_len)},tf.constant(0)
            # return tf.data.Dataset.from_generator(lambda :{'source':tf.constant(source),'context':tf.constant(context),'con_len':tf.constant(con_len)},tf.constant(0))

        pred = self.estimator.predict(input_fn,['target'],yield_single_examples=False)
        res_v = []
        res = next(pred)
        for i,v in enumerate(res['target']):
            vmap = v[con_len]
            sort_res = np.argsort(vmap)[-topk:]
            map_res = {}
            for k in sort_res:
                map_res[k] = vmap[k]
            res_v.append(map_res)
        return res_v
        # count = len(res)


        # print(count)




class NewsBeamsearcher(Beamsearcher):

    def get_context(self,max_step):
        pass
    def get_pred_map(self,pred):
        pass
    def get_input_fn(self):
        pass
    def set_rpcore(self,rp_core):
        self.rpcore = rp_core

    def set_ctcore(self,ct_core):
        self.ctcore = ct_core
    def report(self,fname):

        data = []
        fname = fname.split('.json')[0] if fname.endswith('.json') else fname
        count = int(fname.split('_')[1]) if '_' in fname else 0
        while os.path.exists(fname):
            fname = fname.split('_')[0]+'_%d'%count
            count += 1
        fname += '.json'
        for gen, ref, source in self.gen_result:
            data.append({
                'gen':self.tokenizer.get_sentence(gen),
                'ref':self.tokenizer.get_sentence(ref),
                'source':self.tokenizer.get_sentence(source)
            })
        with open(fname,'w',encoding='utf-8') as jsfile:
            json.dump(data,jsfile,ensure_ascii=False)

        from  evaluate.metrics import generate_ALL

        generate_ALL("../Estimator_edition/%s"%fname)



    def do_search_mt(self,max_step,estimator,rp_fun = 'n'):
        # buffer_lock = threading.RLock()
        # # result_lock = threading.RLock()
        # buffer_lock = threading.Condition(1)
        # result_lock = threading.BoundedSemaphore(self.topk)

        def fill_data(searcher):
            while len(searcher.gen_result) < searcher.max_count:

                if len(searcher.buffer) == 0 or searcher.gen_len >= max_step:
                    if searcher.gen_len >= max_step:
                        searcher.gen_result.append((searcher.buffer[0][0],searcher.title_input,searcher.source_input))

                        print('第{0}步生成的内容：'.format(len(searcher.gen_result)))
                        print(searcher.tokenizer.get_sentence(searcher.buffer[-1][0]))
                    source, title = next(searcher.dataset)
                    searcher.source_input = source[:]
                    searcher.title_input = title[:]


                    searcher.buffer = []
                    for i in range(searcher.topk):
                        searcher.buffer.append(([],0))
                    searcher.gen_len = 0
                    if not searcher.next_topk.empty():
                        searcher.next_topk.pop()
                    if not searcher.context.empty():
                        searcher.context.pop()
                    searcher.ctcore.init(source)
                else:

                    tmp_buffer = []
                    buffer = searcher.buffer
                    next_topk = searcher.next_topk.get()
                    if searcher.gen_len == 0:
                        candidate,score = buffer[0]

                        for i, n in enumerate(next_topk[0]):
                            ts = next_topk[0][n]
                            tc = candidate[:]
                            if n==1:
                                tc.append(searcher.source_input[0])
                            else:
                                tc.append(n)
                            tmp_buffer.append((tc,score+ts,i*PLUS_RATIO))
                            # nexttopk是升序的，最后面的权重最高

                    else:
                        for i, val in enumerate(buffer):
                            candidate,score = val

                            opt_w = 0

                            for n in next_topk[i]:
                                if searcher.topk!= 1 and n==1 and searcher.gen_len<10:
                                    continue
                                ts = next_topk[i][n]
                                tc = candidate[:]
                                # # 无自动截断版本
                                #
                                # tc.append(n)
                                # ts = score + ts

                                # 自动截断
                                if tc[-1] == 1:
                                    tc.append(1)
                                    ts = score
                                else:
                                    tc.append(n)
                                    ts = score + ts
                                score_opt = ts + opt_w
                                opt_w += PLUS_RATIO
                                # tc.append(searcher.title_input[searcher.gen_len])

                                tmp_buffer.append((tc,ts,score_opt))

                    # tmp_buffer = sorted(tmp_buffer,key=lambda x:x[1])
                    tmp_buffer = sorted(tmp_buffer,key=lambda x:x[2])
                    tmp_buffer = tmp_buffer[-searcher.topk:]
                    tmp_buffer.reverse()
                    res = []
                    for t in tmp_buffer:

                        s = searcher.tokenizer.get_sentence(t[0])
                        res.append(s)
                    print(' & '.join(res)+'\\\\')
                    searcher.buffer = [(i[0],i[1]) for i in tmp_buffer]
                    searcher.gen_len += 1
                #
                # if searcher.gen_len>0:
                #     print('第{0}步生成的内容：'.format(searcher.gen_len))


                 # 统计buffer中完成输出的数量，
                if len(searcher.buffer) > 0:
                    c = np.sum([1 if len(l[0])>0 and l[0][-1] == 1 else 0 for l in searcher.buffer])
                else:
                    c = 0
                # c = len(searcher.buffer)
                # 数量达到topk之后，意味着beamsearch中所有的输出都已经达到终点
                if c >= searcher.topk or searcher.gen_len>=max_step:
                     searcher.gen_len = max_step
                else:
                    context = searcher.get_context(max_step)
                    searcher.context.put(context)

        pred = estimator.predict(self.get_input_fn(),['target'],yield_single_examples=False)

        def get_next(searcher):

            while len(searcher.gen_result) < searcher.max_count:
                try:
                    res_v = []
                    res = next(pred)
                    res = searcher.get_pred_map(res)
                    for i,v in enumerate(res):
                        vmap = v


                        # 需要重复惩罚或者上下文调优的时候
                        #
                        # 重复惩罚
                        if rp_fun == 's':
                            vmap_w = doRP_simple(100000,0.3,searcher.buffer[i][0],vmap)
                        elif rp_fun == 'w':
                            vmap_w = doRP_window(100000,0.3,10,searcher.buffer[i][0],vmap)
                        elif rp_fun == 'e':
                            vmap_w = doRP_exp(100000,0.3,0.8,searcher.buffer[i][0],vmap)
                        else:
                            vmap_w = vmap
                        #
                        # 上下文调优
                        # vmap_w = searcher.ctcore.do(vmap_w,searcher.buffer[i][0])

                        sort_res = np.argsort(vmap_w)[-searcher.topk:]
                        map_res = {}
                        for k in sort_res:
                            map_res[k] = vmap[k]
                        res_v.append(map_res)


                    searcher.next_topk.put(res_v)
                except StopIteration:
                    return

        producer = threading.Thread(target=fill_data,args=(self,))
        consumer = threading.Thread(target=get_next,args=(self,))
        producer.start()
        consumer.start()

        producer.join()
        consumer.join()

