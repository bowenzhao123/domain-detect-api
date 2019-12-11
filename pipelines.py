

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
from collections import defaultdict
import pickle
import os
import json
import boto3
import sklearn

#sys.path.append('..')
# import config

from config import ModelConfig

session = boto3.session.Session(profile_name='lost')
s3 = session.resource('s3')

model_conf = ModelConfig()

class BaseFeatureHandler:
    def __init__(self):
        self.providers = {1,2,3,4,5,6,15,16,19}
        self.field = ['script','img','href','text','struct']
        # self.field = ['img','href','text','struct']
        self.provdier_mappers = {
            1: 'Flower Shop Network',
            2: 'teleflora',
            3: 'ftd',
            4: 'bloomnet',
            5: 'websystems',
            6: 'media99',
            15: 'ufn',
            16: 'bloomnation',
            19: 'floranext'
        }

    def load_model(self,path):
        return pickle.load(open(path,'rb'))
        
    def createMatrix(self,text):
        """ clean the url, image source, and script source
        """
        # replace and lower, split by ', '
        text = str(text)
        text = re.sub(r'[^a-zA-Z0-9 .]','',text).lower()
        text = text.replace('. ',' ')
        return text
        # combain tag and class to create a new feature

    def combine_tag_attr(self,tags,attrs):
        """ combine and clean the tag and the class attribute.
        """
        return ' '.join([str(x)+'_'+str(y) for x,y in zip(str(tags).split('. '),str(attrs).split('. '))])

    def createContentMatrix(self,text):
        """ clean the text
        """
        # replace and lower, split by ', '
        text = str(text)
        text = re.sub(r'[^a-zA-Z.]','',text).lower()
        text = text.replace('.',' ')
        return text

    def load_tf_model(self,tf_model=dict()):
        if tf_model:
            return tf_model
        tf_model = dict()
        for feature in self.field:
            try:
                tf_model[feature] = self.load_model(model_conf.tf_model_path[feature])
            except:
                raise ValueError('No tf idf model found!')
        return tf_model

    def load_bow_model(self,bow_model=dict()):
        if bow_model:
            return bow_model
        bow_model = dict()
        for feature in self.field:
            try:
                bow_model[feature] = self.load_model(model_conf.bow_model_path[feature])
            except:
                raise ValueError('No tf idf model found!')
        return bow_model

    def load_nmf_model(self,nmf_model=dict()):
        if nmf_model:
            return nmf_model
        nmf_model = dict()
        for feature in self.field:
            try:
                nmf_model[feature] = self.load_model(model_conf.nmf_model_path[feature])
            except:
                raise ValueError('model not existss!')
        return nmf_model

    def load_ml_model(self,ml_model=None):
        if ml_model:
            return ml_model
        return self.load_model(model_conf.ml_model_path)


class PipeLineHandler(BaseFeatureHandler):
    def __init__(self):
        BaseFeatureHandler.__init__(self)

    def parse_data(self, raw_data):
        struct = self.combine_tag_attr(raw_data['tag'], raw_data['class'])
        raw_data['struct'] = struct
        raw_data.pop('tag')
        raw_data.pop('class')
        parsed_raw_data = {
            'script': self.createMatrix(raw_data['script']),
            'img':self.createMatrix(raw_data['img']),
            'href':self.createMatrix(raw_data['href']),
            'text':self.createContentMatrix(raw_data['text']),
            'struct':raw_data['struct']
        }
        # parsed_raw_data = {k:self.feature_func_mapper[k](v) for k,v in raw_data.items()}
        pred_raw_data = pd.DataFrame({k: [v] for k, v in parsed_raw_data.items()})
        return pred_raw_data

    def load_pred_tf_idf(self,pred_raw_data,tf_model=dict()):
        pred_tf_data_list = dict()
        tf_model = self.load_tf_model(tf_model)
        for feature in self.field:
            pred_tf_data_list[feature] = tf_model[feature].transform(pred_raw_data[feature])
        return pred_tf_data_list
        
    def load_pred_bow(self,pred_raw_data,bow_model=dict()):
        pred_bow_data_list = dict()
        bow_model = self.load_bow_model(bow_model)
        for feature in self.field:
            pred_bow_data_list[feature] = bow_model[feature].transform(pred_raw_data[feature])
        return pred_bow_data_list
    
    def load_pred_trans_data(self,pred_raw_data,tf_model=dict(),nmf_model=dict()):
        mf_model = self.load_nmf_model(nmf_model)
        pred_tf_data_list = self.load_pred_tf_idf(pred_raw_data,tf_model)
        pred_trans_data = []
        for feature in self.field:
            temp = mf_model[feature].transform(pred_tf_data_list[feature])
            pred_trans_data.append(temp)
        pred_trans_data = np.concatenate(pred_trans_data,axis=1)
        return pred_trans_data

    ###############################################################################