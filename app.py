import os
import sys
from json import load
from flask import Flask

#from flask_pymongo import PyMongo
import pymongo

STATIC_FOLDER = sys.path[0] + '/static/'
UPLOAD_FOLDER = sys.path[0] + '/static/uploads/'
TEMPLATES_FOLDER = sys.path[0] + '/templates/'

app = Flask(__name__, template_folder=TEMPLATES_FOLDER, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MONGO_URI'] = 'mongodb+srv://WeldDetector:B1cfD0idOsZQR0d1@weld.zhzshiw.mongodb.net/?retryWrites=true&w=majority'
app.secret_key = "SSKEY"

client = pymongo.MongoClient("mongodb+srv://WeldDetector:B1cfD0idOsZQR0d1@weld.zhzshiw.mongodb.net/?retryWrites=true&w=majority")
db = client.db