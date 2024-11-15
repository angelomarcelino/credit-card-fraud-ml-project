import pandas as pd
import xgboost as xgb
import pickle
from sklearn.feature_extraction import DictVectorizer
from flask import Flask
from flask import request
from flask import jsonify

#constants
INPUT_FILE = '../model/xgboost_eta=0.3_depth=3_minchild=1_round=100.bin'
THRESHOLD = 0.51

#global varialbes
model=None
dv=None

#creating the flask app
app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict ():
    transaction = request.get_json()
    y_pred_proba = predict_proba(transaction, model, dv)
    fraud = y_pred_proba >= THRESHOLD
    result = {
        'fraud_probability': float(y_pred_proba),
        'fraud': bool(fraud)
    }
    return jsonify(result)

def predict_proba(transaction, model, dv):
    X = dv.transform([transaction])
    features = list(dv.get_feature_names_out())
    dMatrix = xgb.DMatrix(X, feature_names=features)
    y_pred_proba = model.predict(dMatrix)
    return y_pred_proba

def initialize():
    global model,dv
    #importing the model
    with open(INPUT_FILE, 'rb') as f_in:
        (model, dv) = pickle.load(f_in)
    app.config['model'] = model
    app.config['dv'] = dv
    print('Model and DictVectorizer loaded successfully. Ready to serve requests.')

initialize()
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)



