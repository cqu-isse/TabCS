import os
os.environ['CUDA_VISIBE_DEVICES'] = '1'
def get_config():   
    conf = {
        'workdir': './data/github/',
        'data_params':{
            #training data
            'train_methname':'train.methodname.pkl',
            'train_apiseq':'train.apiseq.pkl',
            'train_tokens':'train.tokens.pkl',
            'train_sbt':'train.sbt.pkl',
            'train_desc':'train.desc.pkl',
            #valid data
            'valid_methname':'test.methodname.pkl',
            'valid_apiseq':'test.apiseq.pkl',
            'valid_tokens':'test.tokens.pkl',
            'valid_sbt':'test.sbt.pkl',
            'valid_desc':'test.desc.pkl',
            #use data (computing code vectors)
            'use_codebase':'test_source.txt',#'use.rawcode.h5'
            #parameters
            'methname_len': 6,
            'apiseq_len':30,
            'tokens_len':40,
            'sbt_len':20,
            'desc_len': 30,
            'n_methodname_words': 9858,
            'n_desc_words': 12317, # len(vocabulary) + 1
            'n_tokens_words': 14246,
            'n_api_words': 19872,
            'n_sbt_words': 15415,
            #vocabulary info
            'vocab_methname':'vocab.methodname.pkl',
            'vocab_apiseq':'vocab.apiseq.pkl',
            'vocab_tokens':'vocab.tokens.pkl',
            'vocab_sbt':'vocab.sbt.pkl',
            'vocab_desc':'vocab.desc.pkl',
        },               
        'training_params': {           
            'batch_size': 128,
            'chunk_size':100000,
            'nb_epoch': 300,
            'validation_split': 0.025,
            # 'optimizer': 'adam',
            #'optimizer': Adam(clip_norm=0.1),
            'valid_every': 2,
            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'save_every': 1,
            'reload':0, #epoch that the model is reloaded from . If reload=0, then train from scratch
        },

        'model_params': {
            'model_name':'JointEmbeddingModel',
            'n_embed_dims': 100,
            'n_hidden': 400,#number of hidden dimension of code/desc representation
            # recurrent
            'n_lstm_dims': 200, # * 2
            'init_embed_weights_methname':None, #'methodname_index2vec.pkl',#'word2vec_100_methname.h5', 
            'init_embed_weights_tokens':None, #'tokens_index2vec.pkl',#'word2vec_100_tokens.h5',
            'init_embed_weights_sbt':None,#'ast_index2vec.pkl', 
            'init_embed_weights_desc':None, #'desc_index2vec.pkl',#'word2vec_100_desc.h5',
            'init_embed_weights_api':None,#'api_index2vec.pkl',           
            'margin': 0.05,
            'sim_measure':'cos',#similarity measure: gesd, cosine, aesd
        }        
    }
    return conf




