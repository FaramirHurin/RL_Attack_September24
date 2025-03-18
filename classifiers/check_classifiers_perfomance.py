import sklearn
import pandas as pd
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score


with open('Generator/0.5/DNN/classifier_DNN.pickle', 'rb') as file:
    dnn = pickle.load(file)

with open('Generator/0.5/RF/classifier_RF.pickle', 'rb') as file:
    rf = pickle.load(file)

test = pd.read_csv('Generator/0.5/test.csv')

label = test['label']
X = test.drop('label', axis=1)

predictions_dnn = dnn.predict(X)
predictions_rf = rf.predict(X)

print(f1_score(label, predictions_dnn),
      recall_score(label, predictions_dnn), precision_score(label, predictions_dnn))
print(f1_score(label, predictions_rf),
      recall_score(label, predictions_rf), precision_score(label, predictions_rf))

