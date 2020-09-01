# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 01:35:48 2019

@author: SKT
"""

from flask import Flask, request, Response
from flask_cors import CORS
import base64
import json
import sys
from scipy import misc

import age_predict
import gender_predict
from mtcnn import get_single_face  

app = Flask(__name__)
CORS(app)

from config import IMAGE_PATH

def get_data(request):

    if request.data:
        data = request.data.decode('utf-8')
        data = json.loads(data)

    elif request.form:
        data = request.form

    return data

def decode_img(b64):
    if 'data:image/png;base64,' in b64:
        sub = len('data:image/png;base64,')
        b64 = ''.join(list(b64)[sub:])

    img = base64.b64decode(b64)
    
    return img

@app.route('/age_gender_pred', methods=['POST'])
def age_gender_pred():
    data = get_data(request)
    b64 = data['image']
    img = decode_img(b64)
    
    with open(IMAGE_PATH, 'wb') as f:
        f.write(img)
        
    face = get_single_face(IMAGE_PATH)
    print(face.shape)
    
    age_pred = age_predict.predict(face, path=False)
    print(age_pred)

    gender_pred, gender_prob = gender_predict.predict(face, path=False)
    print(gender_pred, gender_prob)

    result = {}
    result["age"] = {"pred" : age_pred}
    result["gender"] = {"pred" : gender_pred, "prob" : str(gender_prob)}
    result = json.dumps(result)
    
    return Response(response=result, status=200, mimetype="application/json")

@app.route('/test')
def test():
    return 'Age, Gender Prediction Works!'
    
if __name__ == '__main__':
   app.run()