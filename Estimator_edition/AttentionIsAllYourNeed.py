
import sys
sys.path.append('/home/user/zsm/Summarization')
import tensorflow as tf
import tensorboard as tb
import numpy as np
from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.transformer import Transformer,create_look_ahead_mask,create_padding_mask
import time
from interface.ContextTune import CTcore
from baseline.Tf_idf import Tf_idf


from interface.NewsInterface import NewsBeamsearcher,NewsPredictor
MODEL_PATH = './transformer'
DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
DATA_PATH = '/home/user/zsm/Summarization/news_data'


NUM_LAYERS=5
D_MODEL=200
NUM_HEAD=8
DFF=512
VOCAB_SIZE=100000
INPUT_LENGTH=1000
OUTPUT_LENGTH=100

R_TF = 1
R_IDF = 0.5


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def build_input_fn(name,data_set,batch_size = 32):

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

                source_sequence = tokenizer.padding(source_sequence,1000)
                title_sequence = tokenizer.tokenize(title)
                title_sequence = tokenizer.padding(title_sequence,100)

                title_sequence.insert(0,2)

                # reweight 操作

                re_weight_map = idf_core.reweight_calc(content,R_IDF,R_TF)
                re_w = [re_weight_map[ti] for ti in title_sequence]
                feature = {
                    'source':source_sequence,
                    'context':title_sequence,
                    'reweight':re_w
                }
            #     yield feature,label
                yield feature,0
            except Exception:
                continue
    def input_fn():
        ds = tf.data.Dataset.from_generator(generator=generator,output_types=({'source':tf.int64,'context':tf.int64,
                                                                               'reweight':tf.float32
                                                                               },tf.int64),
                                            output_shapes=({'source':[INPUT_LENGTH],'context':[OUTPUT_LENGTH+1],
                                                            'reweight':[OUTPUT_LENGTH+1]
                                                            },[]))
        ds = ds.shuffle(8192).batch(batch_size).cache().repeat()
        return ds
    return input_fn





def build_model_fn(lr ,num_layers,d_model,num_head,dff,input_vocab_size,
                            target_vocab_size,
                            pe_input,pe_target):

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
        context = features['context']
        tar_inp = context[:, :-1]
        tar_real = context[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(source, tar_inp)

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
        transformer = Transformer(num_layers= num_layers,
                            d_model=d_model,
                            num_heads=num_head,
                            dff=dff,
                            input_vocab_size=input_vocab_size,
                            target_vocab_size=target_vocab_size,
                            pe_input=pe_input,pe_target=pe_target)
        training = mode == tf.estimator.ModeKeys.TRAIN

        prediction, atte_weight = transformer(source,tar_inp,training, enc_padding_mask,
              combined_mask, dec_padding_mask)
        if training:
            reweight = features['reweight'][:,1:]
        else:
            reweight = None
        loss = loss_function(tar_real,prediction,reweight)
        learning_rate = CustomSchedule(d_model)(global_step)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.98,
                                                     epsilon=1e-9)
        # new_global_step = global_step + 1

        # gradients = tape.gradient(loss,transformer.trainable_variables)
        # train_loss = tf.keras.metrics.Mean(name='train_loss')
        # train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        #     name='train_accuracy')

        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                                      epsilon=1e-9)
        train_accuracy = accuracy_function(tar_real, prediction)
        grads = optimizer.compute_gradients(loss)
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad,global_step)
        grad, var = zip(*grads)
        tf.clip_by_global_norm(grad, 0.5)
        grads = zip(grad, var)

        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        # train_op = tf.group(train_op, [global_step.assign(new_global_step)])

        tf.summary.scalar('ACC',train_accuracy)
        pred = tf.argmax(prediction,-1)
        tf.summary.histogram('PRED',pred)
        # = optimizer.minimize(lambda: loss,transformer.trainable_variables)
        # train_op = optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))
        # train_loss(loss)
        # train_accuracy(tar_real, prediction)
        class TransformerRunHook(tf.estimator.SessionRunHook):
            def __init__(self):
                self.count = 0
                self.start_time  = time.time()
                self.ctime = time.time()
            def before_run(self, run_context):
                return tf.estimator.SessionRunArgs({'loss':loss,'accuracy':train_accuracy,'global_step':global_step,'learning_rate':learning_rate,'PRED':pred})

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
                if self.count % 10  == 0:
                    ntime = time.time()
                    dtime = ntime - self.ctime
                    self.ctime = ntime
                    print("Batch {0} : loss - {1:.3f} :lr - {5:.2e} : accuracy - {2:.3f} : time_cost - {3:.2f} : all_time_cost - {4:.2f}".format(
                        run_values.results['global_step'],run_values.results['loss'],a,dtime,ntime-self.start_time,run_values.results['learning_rate']))
                    # print(statis(run_values.results['PRED']))
                pass
        # summaryHook = tf.estimator.SummarySaverHook(
        #     save_steps=10,
        #     output_dir='./transformer/summary',
        #     summary_writer=None,
        #     summary_op=tf.summary.
        # )
        prediction = prediction - tf.expand_dims(tf.reduce_max(prediction,-1),-1)
        prediction = tf.nn.softmax(prediction,-1)
        prediction = tf.math.log(prediction)

        return tf.estimator.EstimatorSpec(mode,{'target':prediction},loss,train_op,training_hooks=[TransformerRunHook()])

    return model_fn


class TFMBeam(NewsBeamsearcher):
    def get_context(self,max_step):
        context = []
        for seq in self.buffer:
            context.append(self.tokenizer.padding(seq[0][:],max_step))

        return context
    def get_pred_map(self,pred):
        return pred['target'][:,self.gen_len-1,:]
    def get_input_fn(self):
        def data_generator():
            searcher = self
            while len(searcher.gen_result) < searcher.max_count:
                context =  searcher.context.get()
                for c in context:
                    yield {'source':searcher.source_input,'context':c}, tf.constant(0)

        def input_fn():
            return tf.data.Dataset.from_generator(generator=data_generator,output_types=({'source':tf.int64,'context':tf.int64},tf.int64),output_shapes=({'source':[1000],'context':[100]},[])).batch(self.topk)
        return input_fn




    # def generator():
    #     tokenizer = tokenization("/root/zsm/Summarization/news_data/NEWS_DICT.txt",DictSize=100000)
    #     source_file = queue_reader("NEWS","/home/user/zsm/Summarization/news_data")
    #     for line in source_file:
    #         example = line.split('#')
    #         title = example[0]
    #         desc = example[1]
    #         content = example[2]
    #         title = title.replace(' ','')
    #         content = content.replace(' ','')
    #         source_sequence = tokenizer.tokenize(content)
    #         source_sequence = tokenizer.padding(source_sequence,1000)
    #         title_sequence = tokenizer.tokenize(title)
    #         title_sequence = tokenizer.padding(title_sequence,100)
    #         # for i,s in enumerate(title_sequence):
    #         #     label = s
    #         #     context = title_sequence[:i]
    #         # #
    #         # feature = {
    #         #     'source':source_sequence,
    #         #     'context':title_sequence
    #         # }
    #         #     yield feature,label
    #         yield source_sequence,title_sequence
    #
    # g = generator()
    # for s,t in g:
    #     print(s)
def train():
    model_fn = build_model_fn(lr =0.01,num_layers=NUM_LAYERS,d_model=D_MODEL,num_head=NUM_HEAD,dff=DFF,input_vocab_size=VOCAB_SIZE,
                              target_vocab_size=VOCAB_SIZE,
                              pe_input=INPUT_LENGTH,pe_target=OUTPUT_LENGTH)
    estimator = tf.estimator.Estimator(model_fn,model_dir=MODEL_PATH,)
    input_fn = build_input_fn("NEWSOFF",DATA_PATH)

    estimator.train(input_fn,max_steps=10000000)



def beamsearch(topk,cnt,name):
    tokenizer = tokenization(DICT_PATH,DictSize=100000)
    source_file = queue_reader("NEWSOFF", DATA_PATH)
    def _g():
        for source in source_file:

            value = source.split('#')
            source = value[2].split(' ')
            print(''.join(source))
            title = value[0].split(' ')
            source = tokenizer.padding(tokenizer.tokenize(source),1000)
            title = tokenizer.padding(tokenizer.tokenize(title),100)
            yield  source,title
    g = _g()

    model_fn = build_model_fn(lr =0.01,num_layers=NUM_LAYERS,d_model=D_MODEL,num_head=NUM_HEAD,dff=DFF,input_vocab_size=VOCAB_SIZE,
                              target_vocab_size=VOCAB_SIZE,
                              pe_input=INPUT_LENGTH,pe_target=OUTPUT_LENGTH)
    estimator = tf.estimator.Estimator(model_fn, model_dir=MODEL_PATH )

    predictor = NewsPredictor(estimator,topk)

    bs = TFMBeam(dataset=g,tokenizer = tokenizer,topk=topk,predictor=predictor,max_count=cnt)
    bs.set_ctcore(CTcore(VOCAB_SIZE,1.0))
    bs.do_search_mt(100,estimator=estimator)
    bs.report(name)

if __name__ == '__main__':

    # train()
    beamsearch(1,100,'tfmNobeam')
