from flask import *
# from db import db, RulePrediction
from setting import Config
import pandas as pd
import requests
import boto3


import config
import requests
import json

################
from pipelines import PipeLineHandler
from page_parser import getScreadPageContent
import pandas as pd
################

pipeline = PipeLineHandler()
import os

def create_app(conf):
    app = Flask(__name__)
    app.config.from_object(conf)
    #db.init_app(app)

    #db.create_all()
    return app

conf = Config()
app = create_app(conf)


pipeline = PipeLineHandler()
######
ml_model = pipeline.load_ml_model()
tf_model = pipeline.load_tf_model()
bow_model = pipeline.load_bow_model()
nmf_model = pipeline.load_nmf_model()
######


session = boto3.session.Session(profile_name='lost')
s3 = session.resource('s3')
bucket = 'lost-pagesource-staging'

@app.route('/', methods = ['POST'])
def detect():
    domain_name = request.json.get('domain')
    response = s3.Object(bucket, domain_name)
    source = response.get()['Body'].read()

    raw_data = getScreadPageContent(source)

    raw_data = {'domain':domain_name, 'script':raw_data[0], 'img':raw_data[1],
                'href':raw_data[2], 'text':raw_data[3],'tag':raw_data[4],'class':raw_data[5]}

    pred_raw_data = pipeline.parse_data(raw_data)
    pred_trans_data = pipeline.load_pred_trans_data(pred_raw_data,tf_model,nmf_model)

    pred = ml_model.predict(pred_trans_data)[0]
    print(pred)
    prob_matrix = ml_model.predict_proba(pred_trans_data)
    prob = round(max(prob_matrix[0]),2)
    data = dict()
    data['domain_name'] = domain_name
    data['provider'] = pipeline.provdier_mappers.get(int((pred)))
    data['confidence'] = float(prob)
    res = requests.post('http://0.0.0.0:5001/domains/', data=json.dumps(data), headers={'content-type': 'application/json'},verify=False)
    print(res)
    return {"message":"OK"}, 200

if __name__ == '__main__':
    app.run(port=8803, debug=True)
