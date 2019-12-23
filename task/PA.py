from old_session.Seq2seq_PA import Main,Meta

def run():
    # if not os.path.exists('PA_result'):
    #     os.mkdir('PA_result')
    # os.chdir('PA_result')
    # print(os.getcwd())
    meta = Meta().get_meta_comb([
        'LDA.meta',
        'MODEL.meta',
        'seq2seq.meta',
        'train.meta',
        '重复惩罚.meta'
    ])
    Main().run_train(**meta)
def eval():
    meta = Meta().get_meta_comb([
        'LDA.meta',
        'MODEL.meta',
        'seq2seq.meta',
        'train.meta',
        '重复惩罚.meta'
    ])
    Main().run_eval(**meta)