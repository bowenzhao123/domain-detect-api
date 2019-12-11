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
from etlpipeline import SinglePredictPipelineConstructor
################

import os

def create_app(conf):
    app = Flask(__name__)
    app.config.from_object(conf)
    #db.init_app(app)

    #db.create_all()
    return app

conf = Config()
app = create_app(conf)


session = boto3.session.Session(profile_name='lost')
s3 = session.resource('s3')
bucket = 'lost-pagesource-staging'

@app.route('/', methods = ['POST'])
def detect():
    domain_name = request.json.get('domain')
    response = s3.Object(bucket, domain_name)
    source = response.get()['Body'].read()

    predictor = SinglePredictPipelineConstructor()
    data = predictor.predict(domain_name,source)

    res = requests.post('http://0.0.0.0:5001/domains/', data=json.dumps(data), headers={'content-type': 'application/json'},verify=False)
    print(res)
    return {"message":"OK"}, 200

if __name__ == '__main__':
    app.run(port=8804, debug=True)
