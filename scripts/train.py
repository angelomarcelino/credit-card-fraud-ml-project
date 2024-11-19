import glob
import zipfile
import os
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


#constants
DATASET_FOLDER_PATH ="../dataset/"
ETA=0.3
MAX_DEPTH=3
MIN_CHILD_WEIGHT=1
NUM_BOOST_ROUND =100

#functions
def prepare_data(df_train, y_train):
    dv = DictVectorizer(sparse=False)

    train_dicts = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    features = list(dv.get_feature_names_out())
    dMatrix_train = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    return dMatrix_train, dv

def train (df_train, y_train):
    dMatrix, dv = prepare_data(df_train, y_train)
    xgb_params ={
        'eta': ETA,
        'max_depth': MAX_DEPTH,
        'min_child_weight': MIN_CHILD_WEIGHT,
        'objective': 'binary:logistic',
        'seed': 1,
        'verbosity': 1
    }
    model_xgb=xgb.train(xgb_params, dMatrix, num_boost_round=NUM_BOOST_ROUND)
    return model_xgb, dv

def predict (df, dv, model):
    dicts = df.to_dict(orient='records')
    X=dv.transform(dicts)
    features = list(dv.get_feature_names_out())
    dMatrix = xgb.DMatrix(X, feature_names=features)
    y_pred_proba = model.predict(dMatrix)

    return y_pred_proba
    
#data preparation
print ('Reading data from csv files')
all_input_files = glob.glob(DATASET_FOLDER_PATH + "creditcard_part_*.csv")
df_list = [pd.read_csv(file) for file in all_input_files]
df = pd.concat(df_list, ignore_index=True)

print ('Preparing data')
df.columns = df.columns.str.lower()
df_full_train, df_test = train_test_split(df, test_size=0.20, random_state=1)

y_test = df_test["class"].values
y_full_train = df_full_train["class"].values

del df_test["class"]
del df_full_train["class"]

df_test.reset_index(drop=True, inplace=True)
df_full_train=df_full_train.reset_index(drop=True)

#training the model
print ('Training the final model')
model, dv = train(df_full_train,y_full_train)
y_test_pred_proba=predict(df_test, dv, model)
auc = round(roc_auc_score(y_test, y_test_pred_proba),3)
print (f'ROC AUC of the final model: {auc}')

#saving the model
output_file='../model/xgboost_eta=%s_depth=%s_minchild=%s_round=%s.bin'%(ETA, MAX_DEPTH, MIN_CHILD_WEIGHT, NUM_BOOST_ROUND)
with open(output_file, 'wb') as f_out:
    pickle.dump((model,dv), f_out)
