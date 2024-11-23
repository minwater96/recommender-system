# +
#
# Cornac version of ConvNCF (ver1.0)
#

from cornac.models import Recommender
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Embedding, Reshape, Dot
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from tqdm.auto import trange  # progress bar


# +
class ConvNCF(Recommender):    
    def __init__(
        self,
        name="ConvNCF",
        num_factors=64, # Embedding size
        num_channel=32, # Convolution channel size
        act_fn="relu",
        n_epochs=20,
        batch_size=512,
        num_neg=4,  # Number of negative instances to pair with a positive instance
        learning_rate=0.001,
        learner="adam",
        backend="tensorflow",
        early_stopping=None,  # Not yet implemented
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.num_factors = num_factors
        self.num_channel = num_channel
        self.act_fn = act_fn
        self.num_epochs = n_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = learning_rate
        self.learner = learner
        self.backend = backend
        self.early_stopping = early_stopping  # Not yet implemented
        self.seed = seed
        
    def fit(self, train_set, val_set=None):       
        Recommender.fit(self, train_set, val_set)
        self.num_users = train_set.num_users # number of items 
        self.num_items = train_set.num_items # number of items 

        ##### Build a Model #####

        # 입력층: 사용자 및 아이템 입력 정의
        user_input = Input(shape=(1, ))                         
        item_input = Input(shape=(1, ))   
        
        # 임베딩층
        embedding_user = Embedding(self.num_users, self.num_factors)(user_input)          
        embedding_item = Embedding(self.num_items, self.num_factors)(item_input)   
        
        # 임베딩 벡터의 외적(Outer Product) 계산
        embedding_user = Reshape((self.num_factors, 1))(embedding_user) # 임베딩 벡터를 (num_factors,1) 크기로 재배열
        embedding_item = Reshape((1, self.num_factors))(embedding_item) # 임베딩 벡터를 (1,num_factors) 크기로 재배열
        cnn_input = Dot(axes=[2,1])([embedding_user, embedding_item]) # 외적 계산
        cnn_input = Reshape((self.num_factors, self.num_factors, 1))(cnn_input) # KxKx1 이미지
        
        # 컨볼루션층 
        x = cnn_input
        K = self.num_factors
        while K > 1:  # 이미지 크기가 1x1이 될 때까지 컨볼루션 반복
            x = Conv2D(self.num_channel, (2, 2), strides=(2, 2), activation=self.act_fn)(x)
            K //= 2  # 이미지 크기 업데이트
        cnn_output = Flatten()(x)

        # 출력층
        prediction = Dense(1, activation='sigmoid')(cnn_output)

        # Full model
        model = Model(inputs=[user_input, item_input], outputs=prediction)
        
        ##### Set loss, Optimizer and Metrics
        if self.learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(learning_rate=self.lr), loss='binary_crossentropy')
        elif self.learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(learning_rate=self.lr), loss='binary_crossentropy')
        elif self.learner.lower() == "adam":
            model.compile(optimizer=Adam(learning_rate=self.lr), loss='binary_crossentropy')
        elif self.learner.lower() == "sgd":
            model.compile(optimizer=SGD(learning_rate=self.lr), loss='binary_crossentropy')
        else:
            model.compile(optimizer=self.learner, loss='binary_crossentropy')

        ##### Learning a Model #####
        loop = trange(self.num_epochs, disable=not self.verbose)  # progress bar 설정
        for _ in loop:  # 매 epoch 마다 아래 수행
            count = 0
            sum_loss = 0
            # Generate training samples
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                train_set.uir_iter(self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg)): # negative sampling
                # Training
                hist = model.fit([batch_users, batch_items], batch_ratings, batch_size=self.batch_size, epochs=1, verbose=0)
                count += len(batch_users)
                sum_loss += len(batch_users) * hist.history['loss'][0]
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))
        loop.close()    
        self.model = model 

        return self
    
    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            user_id = np.ones(self.num_items) * user_idx,  # 모든 아이템에 대해 예측하기 위해 user_id 반복
            item_id = np.arange(self.num_items)
        else:
            user_id = [user_idx]
            item_id = [item_idx]
        
        return self.model.predict([user_id, item_id], batch_size=self.batch_size*4, verbose=0).ravel()
    
#    def show(self, summary=True, direction="LR"):
#        if summary:
#            print(self.model.summary())
#        tf.keras.utils.plot_model(self.model, show_shapes=True, rankdir=direction)
