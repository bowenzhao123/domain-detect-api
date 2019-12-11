import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    DEBUGE = False
    TESTING = False
    SECRET_KEY = "12345"
    #SQLALCHEMY_DATABASE_URL = os.environment['DATABASE_URL']
