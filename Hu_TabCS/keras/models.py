from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.engine import Input
from keras.layers import Concatenate, Dot, Embedding, Dropout, Lambda, Activation, LSTM, Dense,Reshape,Conv1D,MaxPooling1D,Flatten,GlobalMaxPooling1D,dot,Bidirectional,SimpleRNN
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import logging
from coattention_layer import COAttentionLayer
from attention_layer import AttentionLayer
logger = logging.getLogger(__name__)


class JointEmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params',dict())
        self.methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='i_methname')
        self.apiseq= Input(shape=(self.data_params['apiseq_len'],),dtype='int32',name='i_apiseq')
        self.sbt= Input(shape=(self.data_params['sbt_len'],),dtype='int32',name='i_sbt')
        self.tokens=Input(shape=(self.data_params['tokens_len'],),dtype='int32',name='i_tokens')
        self.desc_good = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_good')
        self.desc_bad = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_bad')
        
        # initialize a bunch of variables that will be set later
        self._code_repr_model=None
        self._desc_repr_model=None        
        self._sim_model = None        
        self._training_model = None
        self._shared_model=None
        #self.prediction_model = None
        
        #create a model path to store model info
        if not os.path.exists(self.config['workdir']+'models/'+self.model_params['model_name']+'/'):
            os.makedirs(self.config['workdir']+'models/'+self.model_params['model_name']+'/')
    
    def build(self):
        '''
        1. Build Code Representation Model
        '''
        logger.debug('Building Code Representation Model')
        methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='methname')
        apiseq= Input(shape=(self.data_params['apiseq_len'],),dtype='int32',name='apiseq')
        tokens=Input(shape=(self.data_params['tokens_len'],),dtype='int32',name='tokens')
        sbt=Input(shape=(self.data_params['sbt_len'],),dtype='int32',name='sbt')

        ## method name representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_methname']) if self.model_params['init_embed_weights_methname'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_methodname_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers in the model must support masking, otherwise an exception will be raised.
                              name='embedding_methname')
        methname_embedding = embedding(methname)
        dropout = Dropout(0.25,name='dropout_methname_embed')
        methname_dropout = dropout(methname_embedding)
        methname_out = AttentionLayer(name = 'methname_attention_layer')(methname_dropout)

        ## API Sequence Representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_api']) if self.model_params['init_embed_weights_api'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_api_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              #weights=weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                         #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_apiseq')
        apiseq_embedding = embedding(apiseq)
        dropout = Dropout(0.25,name='dropout_apiseq_embed')
        apiseq_dropout = dropout(apiseq_embedding)
        api_out = AttentionLayer(name = 'API_attention_layer')(apiseq_dropout)




        ## Tokens Representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_tokens']) if self.model_params['init_embed_weights_tokens'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_tokens_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_tokens')
        tokens_embedding = embedding(tokens)
        dropout = Dropout(0.25,name='dropout_tokens_embed')
        tokens_dropout= dropout(tokens_embedding)
        tokens_out = AttentionLayer(name = 'Tokens_attention_layer')(tokens_dropout)



        ## Sbt Representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_sbt']) if self.model_params['init_embed_weights_sbt'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_sbt_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_sbt')
        sbt_embedding = embedding(sbt)
        dropout = Dropout(0.25,name='dropout_sbt_embed')
        sbt_dropout= dropout(sbt_embedding)
        sbt_out = AttentionLayer(name = 'AST_attention_layer')(sbt_dropout)


        # merge code#
        merged_code= Concatenate(name='code_merge',axis=1)([methname_out,api_out,tokens_out,sbt_out])   #(122,200)

        '''
        2. Build Desc Representation Model
        '''
        ## Desc Representation ##
        logger.debug('Building Desc Representation Model')
        desc = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='desc')
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_desc']) if self.model_params['init_embed_weights_desc'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_desc_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                      #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_desc')
        desc_embedding = embedding(desc)
        dropout = Dropout(0.25,name='dropout_desc_embed')
        desc_dropout = dropout(desc_embedding)
        merged_desc = AttentionLayer(name = 'desc_attention_layer')(desc_dropout)

        #AP networks#
        attention = COAttentionLayer(name='coattention_layer') #  (122,60)
        attention_out = attention([merged_code,merged_desc])

        # out_1 column wise
        gmp_1=GlobalMaxPooling1D(name='blobalmaxpool_colum')
        att_1=gmp_1(attention_out)
        activ1=Activation('softmax',name='AP_active_colum')
        att_1_next=activ1(att_1)
        dot1=Dot(axes=1,normalize=False,name='column_dot')
        desc_out = dot1([att_1_next, merged_desc])

        # out_2 row wise
        attention_trans_layer = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)),name='trans_coattention')
        attention_transposed = attention_trans_layer(attention_out)
        gmp_2=GlobalMaxPooling1D(name='blobalmaxpool_row')
        att_2=gmp_2(attention_transposed)
        activ2=Activation('softmax',name='AP_active_row')
        att_2_next=activ2(att_2)
        dot2=Dot(axes=1,normalize=False,name='row_dot')
        code_out = dot2([att_2_next, merged_code])



        self._code_repr_model=Model(inputs=[methname, apiseq,tokens,sbt,desc],outputs=[code_out],name='desc_repr_model')
        # self._desc_repr_model=desc_repr_model
        print('\nsummary of code representation model')
        self._code_repr_model.summary()
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_desc_repr_model.png'

        self._desc_repr_model=Model(inputs=[methname,apiseq,tokens,sbt,desc],outputs=[desc_out],name='code_repr_model')
        # self._code_repr_model=code_repr_model
        print('\nsummary of description representation model')
        self._desc_repr_model.summary()
  

        
        
        """
        3: calculate the cosine similarity between code and desc
        """     
        logger.debug('Building similarity model') 
        code_repr=self._code_repr_model([methname,apiseq,tokens,sbt,desc])
        desc_repr=self._desc_repr_model([methname,apiseq,tokens,sbt,desc])
        cos_sim=Dot(axes=1, normalize=True, name='cos_sim')([code_repr, desc_repr])
        
        sim_model = Model(inputs=[methname,apiseq,tokens,sbt,desc], outputs=[cos_sim],name='sim_model')   
        self._sim_model=sim_model  #for model evaluation  
        print ("\nsummary of similarity model")
        self._sim_model.summary() 
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_sim_model.png'
        #plot_model(self._sim_model, show_shapes=True, to_file=fname)
        
        
        '''
        4:Build training model
        '''
        good_sim = sim_model([self.methname,self.apiseq,self.tokens,self.sbt, self.desc_good])# similarity of good output
        bad_sim = sim_model([self.methname,self.apiseq,self.tokens,self.sbt, self.desc_bad])#similarity of bad output
        loss = Lambda(lambda x: K.maximum(1e-6, self.model_params['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0], name='loss')([good_sim, bad_sim])

        logger.debug('Building training model')
        self._training_model=Model(inputs=[self.methname,self.apiseq,self.tokens,self.sbt, self.desc_good,self.desc_bad],
                                   outputs=[loss],name='training_model')
        print ('\nsummary of training model')
        self._training_model.summary()      
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_training_model.png'
        #plot_model(self._training_model, show_shapes=True, to_file=fname)     

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        self._code_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
        self._desc_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
        self._training_model.compile(loss=lambda y_true, y_pred: y_pred+y_true-y_true, optimizer=optimizer, **kwargs)
        #+y_true-y_true is for avoiding an unused input warning, it can be simply +y_true since y_true is always 0 in the training set.
        self._sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self._training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1],dtype=np.float32)
        return self._training_model.fit(x, y, **kwargs)

    def repr_code(self, x, **kwargs):
        return self._code_repr_model.predict(x, **kwargs)
    
    def repr_desc(self, x, **kwargs):
        return self._desc_repr_model.predict(x, **kwargs)
    
    def predict(self, x, **kwargs):
        return self._sim_model.predict(x, **kwargs)

    def save(self, code_model_file, desc_model_file, **kwargs):
        assert self._code_repr_model is not None, 'Must compile the model before saving weights'
        self._code_repr_model.save_weights(code_model_file, **kwargs)
        assert self._desc_repr_model is not None, 'Must compile the model before saving weights'
        self._desc_repr_model.save_weights(desc_model_file, **kwargs)

    def load(self, code_model_file, desc_model_file, **kwargs):
        assert self._code_repr_model is not None, 'Must compile the model loading weights'
        self._code_repr_model.load_weights(code_model_file, **kwargs)
        assert self._desc_repr_model is not None, 'Must compile the model loading weights'
        self._desc_repr_model.load_weights(desc_model_file, **kwargs)

 
 
 
 