
import sys,threading,queue
sys.path.append('/home/user/zsm/Summarization')
import tensorflow as tf
import numpy as np
from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.RNNs2s import RNNS2Smodel,create_padding_mask

import time
from interface.NewsInterface import NewsBeamsearcher,NewsPredictor
from baseline.Tf_idf import Tf_idf

from  interface.RepeatPunish import  doRP_simple

MODEL_PATH = './RNNs2s'
DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
DATA_PATH = '/home/user/zsm/Summarization/news_data'

D_MODEL = 200
ENCODE_SIZE = 256
ENCODER_NUM = 3
DECODER_SIZE = 256
DECODER_NUM = 3
VOCAB_SIZE = 100000

R_TF = 1
R_IDF = 0.5


def build_input_fn(name,data_set,batch_size = 1,input_seq_len = 1000,output_seq_len = 100):

    def generator():
        tokenizer = tokenization(DICT_PATH,DictSize=100000)
        idf_core = Tf_idf(DICT_PATH,DATA_PATH+'/NEWS.txt')

        source_file = queue_reader(name,data_set)
        for line in source_file:
            try:
                example = line.split('#')
                title = example[0]
                desc = example[1]
                content = example[2]
                title = title.split(' ')
                content = content.split(' ')
                source_sequence = tokenizer.tokenize(content)
                source_sequence = tokenizer.padding(source_sequence,input_seq_len)
                title_sequence = tokenizer.tokenize(title)
                title_sequence = tokenizer.padding(title_sequence,output_seq_len)
                title_sequence.insert(0,2)
                # title_context.append([0]*context_le)
                source_len = len(content)

                re_weight_map = idf_core.reweight_calc(content,R_IDF,R_TF)
                re_w = [re_weight_map[ti] for ti in title_sequence]


                feature = {
                    'source':source_sequence,
                    'last_word': 0,
                    'source_len':source_len,
                    'title':title_sequence,
                    'reweight':re_w
                }
                #     yield feature,label
                yield feature,0
            except Exception:
                continue
    def input_fn():
        ds = tf.data.Dataset.from_generator(generator=generator,output_types=({'source':tf.int64,'source_len':tf.int64,'last_word':tf.int64,'title':tf.int64,'reweight':tf.float32},tf.int64),
                                            output_shapes=({'source':[input_seq_len],'source_len':[],'last_word':[],'title':[output_seq_len+1],'reweight':[output_seq_len+1]},[]))
        ds = ds.shuffle(8192).batch(batch_size).cache().repeat()
        return ds
    return input_fn





def build_model_fn(lr,d_model, input_vocab_size,encoder_size,encoder_layer_num,decoder_size,decoder_layer_num):

    learning_rate = lr
    #
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction='none')
    def loss_function(real, pred,reweight = None):

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        # loss_ = loss_object(real, pred)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        if reweight!=None:
            mask *= reweight
        loss_ *= mask
        return tf.reduce_mean(loss_)
    def accuracy_function(real,pred):
        train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(real, pred)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=train_accuracy.dtype)

        count = tf.reduce_sum(mask)+1
        train_accuracy = tf.reduce_sum(train_accuracy * mask) / count
        return train_accuracy


    def model_fn(features,labels,mode,params=None):

        # source = features['source']
        # context = features['context']

        global_step = tf.compat.v1.train.get_or_create_global_step()
        source = features['source']
        last_word = features['last_word']
        source_len = features['source_len']
        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()

                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)

                self.warmup_steps = warmup_steps

            def __call__(self, step):
                step = tf.cast(step,tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)

                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        training = mode == tf.estimator.ModeKeys.TRAIN

        RNNS2S_model = RNNS2Smodel(d_model=d_model,input_vocab_size=input_vocab_size,
                                   encoder_size = encoder_size,encoder_layer_num=encoder_layer_num,
                                   decoder_size=decoder_size,decoder_layer_num=decoder_layer_num)


        if mode == tf.estimator.ModeKeys.TRAIN:
            title = features['title']
            title_real = title[:,1:]
            title_input = title[:,:-1]
            mask = create_padding_mask(title_real)
            reweight = features['reweight'][:,1:]
            prediction, decoder_states,enc_vec = RNNS2S_model(source,source_len,title_input,mode,None,mask = mask)
            prediction = prediction - tf.expand_dims(tf.reduce_max(prediction,-1),-1)

            loss = loss_function(title_real,prediction,reweight)
            learning_rate = CustomSchedule(d_model)(global_step)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.98,
                                                         epsilon=1e-9)
            train_accuracy = accuracy_function(title_real, prediction)
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad,global_step)
            grad, var = zip(*grads)
            tf.clip_by_global_norm(grad, 0.5)
            grads = zip(grad, var)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            tf.summary.scalar('accuracy',train_accuracy)
        else:

            mask = tf.zeros(shape=tf.shape(last_word))
            states = features['state']
            states = tf.transpose(states,[1,0,2])
            states = tf.split(states,DECODER_NUM)
            last_word = tf.expand_dims(last_word,-1)
            dec_init_state = []

            for state in states:

                dec_init_state.append([tf.squeeze(i,0) for i in tf.split(state,2)])


            prediction,decoder_states,enc_vec = RNNS2S_model(source,source_len,last_word,mode,dec_init_state,mask = mask)

            prediction = prediction - tf.expand_dims(tf.reduce_max(prediction,-1),-1)

            loss = tf.constant(0)
            train_accuracy = tf.constant(0)
            train_op = tf.no_op()
            prediction = tf.nn.softmax(prediction,-1)
            prediction = tf.math.log(prediction)


        class TransformerRunHook(tf.estimator.SessionRunHook):
            def __init__(self):
                self.count = 0
                self.start_time  = time.time()
                self.ctime = time.time()
            def before_run(self, run_context):
                return tf.estimator.SessionRunArgs({'loss':loss,'accuracy':train_accuracy,'global_step':global_step,
                                                    'learning_rate':learning_rate,'PRED':prediction,'title':title_real,
                                                    'enc_vec':enc_vec
                                                    })

            def after_run(self, run_context, run_values):

                self.count += 1
                # a = np.mean(run_values.results['accuracy'])
                a = run_values.results['accuracy']
                def statis(pred):
                    c_map = {}
                    pred = np.reshape(pred,[-1])
                    for w in pred:
                        c_map.setdefault(w,0)
                        c_map[w] += 1
                    return c_map

                if run_values.results['global_step'] % 5 == 0:
                    ntime = time.time()
                    dtime = ntime - self.ctime
                    self.ctime = ntime
                    print("Batch {0} : loss - {1:.3f} :lr - {5:.2e} : accuracy - {2:.3f} : TCPB - {3:.2f} : TTC - {4:.2f} h".format(
                        run_values.results['global_step'],run_values.results['loss'],a,dtime,(ntime-self.start_time)/3600,run_values.results['learning_rate']))
                    # print(run_values.results['PRED'])

                pass


        return tf.estimator.EstimatorSpec(mode,{'target':prediction,'state':decoder_states,
                                                'enc_vec':enc_vec
                                                },loss,train_op,training_hooks=[TransformerRunHook()])

    return model_fn


class S2SBeamSearcher(NewsBeamsearcher):
    def __init__(self,dataset,tokenizer,topk,predictor,context_len,max_count):
        super(S2SBeamSearcher,self).__init__(dataset,tokenizer,topk,predictor,max_count)
        self.context_len = context_len



    def get_pred_map(self,pred):
        return pred['target'][:,0,:],pred['state']


    def get_input_fn(self):
        def data_generator():
            searcher = self
            while len(searcher.gen_result) < searcher.max_count:
                state = searcher.state.get()
                # state_t = np.transpose(state,[1,0,2])
                # buffer_lock.wait(2)
                for i in range(searcher.topk):
                    lastword = searcher.buffer[i][0][-1] if searcher.buffer[i][0] else 2
                    yield {'source':searcher.source_input,'source_len':searcher.source_len,'state':state[i],'last_word':lastword}, tf.constant(0)

        def input_fn():
            return tf.data.Dataset.from_generator(generator=data_generator,output_types=({'source':tf.int64,'source_len':tf.int64,'last_word':tf.int64,'state':tf.float32},tf.int64),
                                                  output_shapes=({'source':[1000],'source_len':[],'last_word':[],'state':[DECODER_NUM*2,DECODER_SIZE]},[])).batch(self.topk,drop_remainder=True)
        return input_fn
    def do_search_mt(self,max_step,estimator,rp_core = None):
        # buffer_lock = threading.RLock()
        # # result_lock = threading.RLock()
        # buffer_lock = threading.Condition(1)
        # result_lock = threading.BoundedSemaphore(self.topk)

        self.state = queue.Queue(1)

        def fill_data(searcher):
            while len(searcher.gen_result) < searcher.max_count:

                if len(searcher.buffer) == 0 or searcher.gen_len == max_step:
                    if searcher.gen_len == max_step:
                        searcher.gen_result.append((searcher.buffer[-1][0],searcher.title_input,searcher.source_input))

                        print('第{0}步生成的内容：'.format(len(searcher.gen_result)))
                        # print('源：{}'.format(searcher.tokenizer.get_sentence(searcher.source_input)))
                        print(searcher.tokenizer.get_sentence(searcher.buffer[-1][0]))
                    source,source_len,title = next(searcher.dataset)
                    searcher.source_input = source[:]
                    searcher.source_len = source_len
                    searcher.title_input = title[:]


                    searcher.buffer = []
                    for i in range(searcher.topk):
                        searcher.buffer.append(([],0))
                    searcher.gen_len = 0
                    searcher.next_topk.empty()
                    searcher.state.empty()
                    state = tf.zeros([searcher.topk,DECODER_NUM*2,DECODER_SIZE],np.float32)
                else:

                    tmp_buffer = []
                    buffer = searcher.buffer
                    next_topk , state = searcher.next_topk.get()
                    if searcher.gen_len == 0:
                        candidate,score = buffer[0]

                        for i, n in enumerate(next_topk[0]):
                            ts = next_topk[0][n]
                            tc = candidate[:]
                            if n==1:
                                tc.append(searcher.source_input[0])
                            else:
                                tc.append(n)
                            tmp_buffer.append((tc,score+ts))

                    else:
                        for i, val in enumerate(buffer):
                            candidate,score = val
                            for n in next_topk[i]:

                                if n==1 and searcher.gen_len<20:
                                    continue

                                ts = next_topk[i][n]
                                tc = candidate[:]
                                # #
                                if tc[-1] == 1:
                                    tc.append(1)
                                    ts = score
                                else:
                                    tc.append(n)
                                    ts = score + ts

                                # print(searcher.tokenizer.get(n))
                                # tc.append(searcher.title_input[searcher.gen_len])

                                tmp_buffer.append((tc,ts))

                    tmp_buffer = sorted(tmp_buffer,key=lambda x:x[1])
                    tmp_buffer = tmp_buffer[-searcher.topk:]
                    searcher.buffer = tmp_buffer
                    searcher.gen_len += 1
                #
                # if searcher.gen_len>0:
                #     print('第{0}步生成的内容：'.format(searcher.gen_len))

                if len(searcher.buffer) > 0:
                    c = np.sum([1 if len(l[0])>0 and l[0][-1] == 1 else 0 for l in searcher.buffer])
                else:
                    c = 0
                if c == searcher.topk:
                    # 用于判定是否是第一次的生成，因为后续的生成都是 c * c 的数量
                    searcher.gen_len = max_step
                else:
                    searcher.state.put(state)


        pred = estimator.predict(self.get_input_fn(),['target','state',
                                                      'enc_vec'
                                                      ],yield_single_examples=False)

        def get_next(searcher):

            while len(searcher.gen_result) < searcher.max_count:
                try:
                    res_v = []
                    res = next(pred)
                    res,state = searcher.get_pred_map(res)
                    for i,v in enumerate(res):
                        vmap = v
                        vmap_w = doRP_simple(100000,0.1,searcher.buffer[i][0],vmap)

                        sort_res = np.argsort(vmap_w)[-searcher.topk:]
                        map_res = {}
                        for k in sort_res:
                            map_res[k] = vmap[k]
                        res_v.append(map_res)


                    searcher.next_topk.put((res_v,state))
                except StopIteration:
                    return

        producer = threading.Thread(target=fill_data,args=(self,))
        consumer = threading.Thread(target=get_next,args=(self,))
        producer.start()
        consumer.start()

        producer.join()
        consumer.join()




def train():
    model_fn = build_model_fn(lr = 0.01,d_model=D_MODEL,input_vocab_size=VOCAB_SIZE,encoder_size=ENCODE_SIZE,encoder_layer_num=ENCODER_NUM,
                              decoder_size=DECODER_SIZE,decoder_layer_num = DECODER_NUM)
    estimator = tf.estimator.Estimator(model_fn,model_dir=MODEL_PATH,)
    input_fn = build_input_fn("NEWSOFF", DATA_PATH,batch_size=16,input_seq_len=1000,output_seq_len=100)

    estimator.train(input_fn,max_steps=10000000)

def beamsearch(topk,cnt,name):
    tokenizer = tokenization(DICT_PATH,DictSize=100000)
    source_file = queue_reader("NEWSOFF", DATA_PATH )

    def _g():
        for source in source_file:
            value = source.split('#')
            source  = value[2].split(' ')
            source_len = len(source)
            print(''.join(source))

            title = value[0].split(' ')
            source = tokenizer.padding(tokenizer.tokenize(source),1000)
            title = tokenizer.padding(tokenizer.tokenize(title),100)
            yield  source,source_len,title
    g = _g()

    model_fn = build_model_fn(lr = 0.1,d_model=D_MODEL,input_vocab_size=VOCAB_SIZE,encoder_size=ENCODE_SIZE,encoder_layer_num=ENCODER_NUM,
                              decoder_size=DECODER_SIZE,decoder_layer_num = DECODER_NUM)
    estimator = tf.estimator.Estimator(model_fn, model_dir=MODEL_PATH, )
    predictor = NewsPredictor(estimator,topk)
    bs = S2SBeamSearcher(dataset=g,tokenizer = tokenizer,topk=topk,context_len=10,predictor=predictor,max_count=cnt)
    bs.do_search_mt(40,estimator)
    bs.report(name)

if __name__ == '__main__':

    # train()
    beamsearch(5,100,'s2s')