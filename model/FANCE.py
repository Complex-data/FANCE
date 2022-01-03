import pickle
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import *
from keras.utils import CustomObjectScope
from keras.engine.topology import Layer
from keras import initializers
from sklearn.utils.extmath import softmax


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from keras.utils import plot_model
import os
import datetime

K.set_learning_phase(1)


class Metrics(Callback):
    def __init__(self, platform, alter_params):
        self.log_file = open('./Log_FANCE_' + alter_params + '_' + platform + '.txt', 'a')

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []
        self.val_acc = []
        

    def on_epoch_end(self, epoch, logs={}):
        val_predict_onehot = (
            np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2], self.validation_data[3]]))).round()
        val_targ_onehot = self.validation_data[4]
        val_predict = np.argmax(val_predict_onehot, axis=1)
        val_targ = np.argmax(val_targ_onehot, axis=1)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_auc = roc_auc_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        self.val_acc.append(_val_acc)
        print "Epoch: %d - val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f auc: %f" % (
            epoch, _val_acc, _val_precision, _val_recall, _val_f1, _val_auc)
        self.log_file.write(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Epoch: %d - %f %f %f %f %f\n" % (epoch,
                                                                                                          _val_acc,
                                                                                                          _val_precision,
                                                                                                          _val_recall,
                                                                                                          _val_f1,
                                                                                                          _val_auc))
        return _val_acc,_val_precision,_val_recall,_val_f1,_val_auc


class LLayer(Layer):
    """
    Co-attention layer which accept content and comment states and computes co-attention between them and returns the
     weighted sum of the content and the comment states
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.latent_dim = 200
        self.k = 100
        super(LLayer, self).__init__(**kwargs)

    def build(self, input_shape, mask=None):
        self.Wl = K.variable(self.init((self.latent_dim, self.latent_dim)))

        self.Wc = K.variable(self.init((self.k, self.latent_dim)))
        self.Ws = K.variable(self.init((self.k, self.latent_dim)))

        self.whs = K.variable(self.init((1, self.k)))
        self.whc = K.variable(self.init((1, self.k)))
        self.trainable_weights = [self.Wl, self.Wc, self.Ws, self.whs, self.whc]

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        comment_rep = x[0]
        sentence_rep = x[1]
        sentence_rep_trans = K.permute_dimensions(sentence_rep, (0, 2, 1))
        comment_rep_trans = K.permute_dimensions(comment_rep, (0, 2, 1))
        L = K.tanh(tf.einsum('btd,dk,bkn->btn', comment_rep, self.Wl, sentence_rep_trans))
        L_trans = K.permute_dimensions(L, (0, 2, 1))

        Hs = K.tanh(tf.einsum('kd,bdn->bkn', self.Ws, sentence_rep_trans) + tf.einsum('kd,bdt,btn->bkn', self.Wc,
                                                                                      comment_rep_trans, L))
        Hc = K.tanh(tf.einsum('kd,bdt->bkt', self.Wc, comment_rep_trans) + tf.einsum('kd,bdn,bnt->bkt', self.Ws,
                                                                                     sentence_rep_trans, L_trans))
        As = K.softmax(tf.einsum('yk,bkn->bn', self.whs, Hs))
        Ac = K.softmax(tf.einsum('yk,bkt->bt', self.whc, Hc))
        co_s = tf.einsum('bdn,bn->bd', sentence_rep_trans, As)
        co_c = tf.einsum('bdt,bt->bd', comment_rep_trans, Ac)
        co_sc = K.concatenate([co_s, co_c], axis=1)

        return co_sc

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.latent_dim + self.latent_dim)


class MyAdd(Layer):
    def __init__(self, **kwargs):
        super(MyAdd, self).__init__(**kwargs)
        self.init = initializers.get('normal')
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask[1]
    
    def call(self, x, mask=None):
        return x[0] + x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[1]
            
class AttLayer(Layer):
    """
    Attention layer used for the calcualting attention in word and sentence levels
    """

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = 100

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class FANCE():
    def __init__(self, platform, alter_params, MAX_SENTENCE_LENGTH = 50, MAX_COMS_LENGTH = 20, MAX_COMS_COUNT = 50, MAX_SENTENCE_COUNT = 50, USER_COMS = 10, lr = 0.0004):
        self.model = None
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.MAX_SENTENCE_COUNT = MAX_SENTENCE_COUNT
        self.MAX_COMS_COUNT = MAX_COMS_COUNT
        self.MAX_COMS_LENGTH = MAX_COMS_LENGTH
        self.MAX_COMS_DIM_LENGTH = 200
        self.USER_COMS = USER_COMS
        self.lr = lr
        self.VOCABULARY_SIZE = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.sentence_comment_co_model = None
        self.tokenizer = None
        self.class_count = 2
        self.metrics = Metrics(platform, alter_params)

        # Variables for calculating attention weights
        self.news_content_word_level_encoder = None
        self.comment_word_level_encoder = None
        self.news_content_sentence_level_encoder = None
        self.comment_sequence_encoder = None
        self.co_attention_model = None
        self.user_encoder1 = None
        self.user_encoder2 = None
        self.commentEncoder_final =None

    def _build_model(self, n_classes=2, embedding_dim=100, aff_dim=80):
        GLOVE_DIR = "."
        content_input = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_COMS_DIM_LENGTH), dtype='float32')
        l_lstm = Bidirectional(GRU(200, return_sequences=True), name='word_lstm')
        l_lstm_output = l_lstm(content_input)
        content_encoder = Dense(200,activation='tanh',name="content_dropout")(l_lstm_output)


        sentiment_input = Input(shape=(1, self.MAX_COMS_DIM_LENGTH,), dtype='float32')
        sentiment_re = Dropout(0.2, name='sentiment')(sentiment_input)
        content_encoder_final = Add(name='sentiment_content')([content_encoder,sentiment_re])
        

        user_att_input = Input(shape=(self.MAX_COMS_COUNT, self.MAX_COMS_DIM_LENGTH,), dtype='float32')
        user_re = Dropout(0.2, name='user_dropout')(user_att_input)
        

        all_comment_input = Input(shape=(self.MAX_COMS_COUNT, self.MAX_COMS_DIM_LENGTH), dtype='float32')
        all_commment_out = Dropout(0.2, name='comment_sequence_out')(all_comment_input)


        all_comment_encoder_final = Add(name='hua')([all_commment_out,user_re])
        commentEncoder_final = Model([all_comment_input, user_att_input], all_comment_encoder_final)
        self.commentEncoder_final = commentEncoder_final


        # Co-attention
        L = LLayer(name="co-attention")([all_comment_encoder_final, content_encoder_final])
        L_Model = Model([all_comment_input, user_att_input, content_input,sentiment_input], L)
        self.co_attention_model = L_Model

        preds = Dense(2, activation='softmax')(L)
        model = Model(inputs=[all_comment_input, user_att_input, content_input,sentiment_input], outputs=preds)
        model.summary()
        optimize = RMSprop(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimize)

        return model


    def _encode_texts(self, texts):
        encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_COMS_DIM_LENGTH), dtype='float32')
        for i, text in enumerate(texts):
            encoded_text = text[:self.MAX_SENTENCE_COUNT]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts

    def _encode_sentiment(self, sentiment):
        encoded_texts = np.zeros((len(sentiment), 1, self.MAX_COMS_DIM_LENGTH), dtype='float32')
        for i, text in enumerate(sentiment):
            encoded_text = text[:1]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts

    def _encode_comments(self, comments):
        encoded_texts = np.zeros((len(comments), self.MAX_COMS_COUNT, self.MAX_COMS_DIM_LENGTH), dtype='float32')
        for i, text in enumerate(comments):
            encoded_text = text[:self.MAX_COMS_COUNT]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts
    
    def _encode_comments_id(self, comments_id):
        encoded_texts = np.zeros((len(comments_id), self.MAX_COMS_COUNT, self.MAX_COMS_DIM_LENGTH), dtype='float32')
        for i, text in enumerate(comments_id):
            encoded_text = text[:self.MAX_COMS_COUNT]
            encoded_texts[i][:len(encoded_text)] = encoded_text
        return encoded_texts

    def train(self, train_x, train_x_senti, train_y, train_c, train_cid, val_cid, val_c, val_x, val_x_senti, val_y,
              batch_size=20, epochs=10):
        self.model = self._build_model(
            n_classes=train_y.shape[-1],
            embedding_dim=100)

        encoded_train_x = self._encode_texts(train_x)
        encoded_val_x = self._encode_texts(val_x)
        encoded_train_x_senti =  self._encode_sentiment(train_x_senti)
        encoded_val_x_senti =  self._encode_sentiment(val_x_senti)
        encoded_train_c = self._encode_comments(train_c)
        encoded_val_c = self._encode_comments(val_c)
        encoded_train_cid = self._encode_comments_id(train_cid)
        encoded_val_cid = self._encode_comments_id(val_cid)


        callbacks = []

        callbacks.append(self.metrics)
        
        self.model.fit([encoded_train_c, encoded_train_cid, encoded_train_x,encoded_train_x_senti], y=train_y,
                       validation_data=([encoded_val_c, encoded_val_cid, encoded_val_x,encoded_val_x_senti], val_y),
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       callbacks=callbacks)

    def predict(self, x, c, cid, x_senti):
        encoded_x = self._encode_texts(x)
        encoded_c = self._encode_comments(c)
        encoded_cid = self._encode_comments_id(cid)
        encoded_x_senti = self._encode_sentiment(x_senti)
        return self.model.predict([encoded_c, encoded_cid, encoded_x, encoded_x_senti,])

    def process_atten_weight_com(self, encoded_text, sentence_co_attention):
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k][:50]
            for i in range(len(cur_text)):
                if sum(cur_text[i])==0:
                    continue
                tmp_no_pad_text_att.append(sentence_co_attention[k][i])

            no_pad_text_att.append(tmp_no_pad_text_att)

        return no_pad_text_att

    def activation_maps(self, news_article_sentence_list, news_article_comment_list, news_article_user_list, websafe=False):
        encoded_text = self._encode_texts(news_article_sentence_list)
        encoded_comment = self._encode_comments(news_article_comment_list)
        encoded_cid = self._encode_comments_id(news_article_user_list)
        content_word_level_attentions = []

        comment_level_encoder = Model(inputs=[self.comment_sequence_encoder.input, self.user_encoder2.input],
                                      outputs=self.commentEncoder_final.output)
        comment_level_weights = comment_level_encoder.predict([encoded_comment, encoded_cid])


        sentence_level_weights = encoded_text

        [Wl, Wc, Ws, whs, whc] = self.co_attention_model.get_weights()

        sentence_rep = sentence_level_weights
        comment_rep = comment_level_weights
        sentence_rep_trans = np.transpose(sentence_rep, axes=(0, 2, 1))
        comment_rep_trans = np.transpose(comment_rep, axes=(0, 2, 1))

        L = np.tanh(np.einsum('btd,dk,bkn->btn', comment_rep, Wl, sentence_rep_trans))
        
        L_trans = np.transpose(L, axes=(0, 2, 1))

        Hs = np.tanh(np.einsum('kd,bdn->bkn', Ws, sentence_rep_trans) + np.einsum('kd,bdt,btn->bkn', Wc,
                                                                                  comment_rep_trans, L))
        Hc = np.tanh(np.einsum('kd,bdt->bkt', Wc, comment_rep_trans) + np.einsum('kd,bdn,bnt->bkt', Ws,
                                                                                 sentence_rep_trans, L_trans))
        sent_unnorm_attn = np.einsum('yk,bkn->bn', whs, Hs)
        comment_unnorm_attn = np.einsum('yk,bkt->bt', whc, Hc)
        sentence_co_attention = (np.exp(sent_unnorm_attn) / np.sum(np.exp(sent_unnorm_attn), axis=1)[:, np.newaxis])
        comment_co_attention = (
                np.exp(comment_unnorm_attn) / np.sum(np.exp(comment_unnorm_attn), axis=1)[:, np.newaxis])
        if websafe:
            sentence_co_attention = sentence_co_attention.astype(float)
            comment_co_attention = comment_co_attention.astype(float)
            comment_word_level_attentions = np.array(comment_word_level_attentions).astype(float)
            content_word_level_attentions = np.array(content_word_level_attentions).astype(float)

        
        res_comment_weight = self.process_atten_weight_com(encoded_comment,comment_co_attention)

        res_sentence_weight = self.process_atten_weight_com(encoded_text, sentence_co_attention)

        return res_sentence_weight, res_comment_weight
