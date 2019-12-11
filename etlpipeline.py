

import pandas as pd
import numpy as np
import re
import pickle
import boto3
import sklearn

#sys.path.append('..')
# import config

from config import ModelConfig


model_conf = ModelConfig()

from bs4 import BeautifulSoup
import time
import requests
import re
from collections import Counter
# import selenium
# from selenium import webdriver
import logging


class DataExtractor:
    def __init__(self):
        pass

    def parse_link(self,link):
        try:
            linkInfo = re.split('\W+',link.replace('https://','').replace('http://','').replace('www.','').replace('.com',''))
            linkInfo = ' '.join(list(filter(lambda x:x and len(x)>0, linkInfo)))
            return linkInfo
        except:
            return ''

    def parse_page_content(self,req):

        soup = BeautifulSoup(req,'html.parser')
        try:
            script_info = soup.find_all('script')
            script_info = list(filter(lambda x:x and x.get('src') and len(x.get('src'))>1, script_info))
            script = list(map(lambda x:x.get('src'),script_info))
            script = '. '.join(list(map(lambda x:self.parse_link(x),script)))
        except:
            script = ''
        
        # get all the img
        try:
            img_info = soup.find_all('img')
            imgs_info = list(filter(lambda x:x and x.get('src') and len(x.get('src'))>1, img_info))
            imgs = list(map(lambda x:x.get('src'),img_info))
            # imgs = list(filter(lambda x:x is not None and len(x)>0,imgs))
            imgs = '. '.join(list(map(lambda x:self.parse_link(x),imgs)))
        except:
            imgs = ''
        
        # get all the links
        try:
            url_info = soup.find_all('a')
            url_info = list(filter(lambda x:x and x.get('href') and len(x.get('href'))>1, url_info))
            urls = list(map(lambda x:x.get('href'),url_info))
            # urls = list(filter(lambda x:x is not None and len(x)>0,urls))
            urls = '. '.join(list(map(lambda x:self.parse_link(x),urls)))
        except:
            ulrs = ''
        
        for scripts in soup(["script", "style"]):
            scripts.extract()
        # get text
        content = soup.get_text().split('\n')
        content = list(map(lambda x:' '.join(re.sub(r'[^a-zA-Z ]','',x).split()).lower(),content))
        content = '. '.join(list(filter(lambda x:len(x)<50 and len(x)>1,content)))
        
        # get all elemnents' class names
        try:
            all_tag_elems = soup.find('body').find_all()
            all_tag_elems = list(filter(lambda x:x and x.name and x.get('class'), all_tag_elems))
            all_tags = list(map(lambda x:(x.name,' '.join(x.get('class'))),all_tag_elems))
            
            freqs = Counter(all_tags).items()
            all_tags = list(filter(lambda x:x[1]>1,freqs))
            
            tags = '. '.join(list(map(lambda x:x[0][0],all_tags)))
            class_values = '. '.join(list(map(lambda x:x[0][1],all_tags)))
        except:
            tags = class_values = ''
        return [script, imgs, urls, content, tags, class_values]


class DataTransfomer:
    def __init__(self):
        pass
        
    def create_matrix(self,text):
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

    def create_content_matrix(self,text):
        """ clean the text
        """
        # replace and lower, split by ', '
        text = str(text)
        text = re.sub(r'[^a-zA-Z.]','',text).lower()
        text = text.replace('.',' ')
        return text

    def parse_data(self, raw_data):
        struct = self.combine_tag_attr(raw_data['tag'], raw_data['class'])
        raw_data['struct'] = struct
        raw_data.pop('tag')
        raw_data.pop('class')
        parsed_raw_data = {
            'script': self.create_matrix(raw_data['script']),
            'img':self.create_matrix(raw_data['img']),
            'href':self.create_matrix(raw_data['href']),
            'text':self.create_content_matrix(raw_data['text']),
            'struct':raw_data['struct']
        }
        # parsed_raw_data = {k:self.feature_func_mapper[k](v) for k,v in raw_data.items()}
        pred_raw_data = pd.DataFrame({k: [v] for k, v in parsed_raw_data.items()})
        return pred_raw_data


class ModelLoader:
    def __init__(self):
        self.field = ['script', 'img', 'href', 'text', 'struct']
        #self.model_bucket = ModelConfig.bucket

    def load_model(self,path):
        return pickle.load(open(path,'rb'))

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


class FeatureLoader:
    def __init__(self):
        self.field = ['script','img','href','text','struct']
        self.providers = {1,2,3,4,5,6,15,16,19}
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
        self.model_loader = ModelLoader()

    def load_pred_tf_idf(self,pred_raw_data,tf_model=dict()):
        pred_tf_data_list = dict()
        tf_model = self.model_loader.load_tf_model(tf_model)
        for feature in self.field:
            pred_tf_data_list[feature] = tf_model[feature].transform(pred_raw_data[feature])
        return pred_tf_data_list
    
    """
    def load_pred_bow(self,pred_raw_data,bow_model=dict()):
        pred_bow_data_list = dict()
        bow_model = self.model_loader.load_bow_model(bow_model)
        for feature in self.field:
            pred_bow_data_list[feature] = bow_model[feature].transform(pred_raw_data[feature])
        return pred_bow_data_list
    """

    def load_pred_trans_data(self,pred_raw_data,tf_model=dict(),nmf_model=dict()):
        mf_model = self.model_loader.load_nmf_model(nmf_model)
        pred_tf_data_list = self.load_pred_tf_idf(pred_raw_data,tf_model)
        pred_trans_data = []
        for feature in self.field:
            temp = mf_model[feature].transform(pred_tf_data_list[feature])
            pred_trans_data.append(temp)
        pred_trans_data = np.concatenate(pred_trans_data,axis=1)
        return pred_trans_data


    def get_predict_result(self,pred_trans_data,ml_model=None):
        ml_model = self.model_loader.load_ml_model(ml_model)

        pred = ml_model.predict(pred_trans_data)[0]
        prob_matrix = ml_model.predict_proba(pred_trans_data)
        prob = [round(max(row),2) for row in prob_matrix]
        return pred, prob

    ###############################################################################

class SinglePredictPipelineConstructor:
    def __init__(self):
        self.data_extractor = DataExtractor()
        self.data_transfomer = DataTransfomer()
        self.data_loader = FeatureLoader()

    def predict(self,domain_name,body):
        raw_data = self.data_extractor.parse_page_content(body)

        raw_data = {'domain':domain_name, 'script':raw_data[0], 'img':raw_data[1],
                'href':raw_data[2], 'text':raw_data[3],'tag':raw_data[4],'class':raw_data[5]}

        pred_raw_data = self.data_transfomer.parse_data(raw_data)
        pred_trans_data = self.data_loader.load_pred_trans_data(pred_raw_data)
        pred, prob = self.data_loader.get_predict_result(pred_trans_data)
        print(pred, prob)
        data = dict()
        data['domain_name'] = domain_name
        data['provider'] = self.data_loader.provdier_mappers.get(int((pred)))
        data['confidence'] = float(prob[0])

        return data