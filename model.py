import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
from tokenizers import ByteLevelBPETokenizer
import flask

from flask import Flask, jsonify, request
app = Flask(__name__)

MAX_LEN = 128
tokenizer = ByteLevelBPETokenizer(\
    vocab='vocab-roberta-base.json',\
    merges='merges-roberta-base.txt',\
    lowercase=True,\
    add_prefix_space=True)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

config = RobertaConfig.from_pretrained('config-roberta-base.json')
bert_model = TFRobertaModel.from_pretrained('pretrained-roberta-base.h5',config=config)
x = bert_model(ids,attention_mask=att,token_type_ids=tok)

x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
x1 = tf.keras.layers.Conv1D(1,1)(x1)
x1 = tf.keras.layers.Flatten()(x1)
x1 = tf.keras.layers.Activation('softmax')(x1)

x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
x2 = tf.keras.layers.Conv1D(1,1)(x2)
x2 = tf.keras.layers.Flatten()(x2)
x2 = tf.keras.layers.Activation('softmax')(x2)

model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

model.load_weights('roberta.h5')

def computeTestData(data,sentiment):	
    '''Prepare data as required for ROBERTA model'''    
    
    input_ids = np.ones((1,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((1,MAX_LEN),dtype='int32')
    token_type_ids = np.zeros((1,MAX_LEN),dtype='int32')
    
    
    text=" "+" ".join(data.split())
    enc=tokenizer.encode(text)
    s_tok = sentiment_id[sentiment]

    input_ids[0,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask[0,:len(enc.ids)+5] = 1
        
    return input_ids,attention_mask,token_type_ids

def find_selected_text(data,start,end):
	'''Finds the selected text for the given tweet'''  

	#Finding the start and end index
	start_idx=np.argmax(start)
	end_idx=np.argmax(end)


	#If start is greater than end index, predicted_text=text
	if (start_idx>end_idx):
	  predicted_text=data

	else:
	  text1 = " "+" ".join(data.split())
	  tokens=tokenizer.encode(text1)
	  predicted_text=tokenizer.decode(tokens.ids[start_idx-1:end_idx]) 

	return predicted_text

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	lst=request.form.to_dict()
	data=lst['tweet']
	sentiment=lst['sentiment']
	test1,test2,test3=computeTestData(data,sentiment)
	start,end=model.predict([test1,test2,test3])
	selected_text=find_selected_text(data,start,end)

	return flask.render_template('index.html',selected_text="Part of tweet that lead to {} sentiment is \"{}\" "\
									.format(sentiment,selected_text))


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080)