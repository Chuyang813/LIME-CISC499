import numpy as np
import pandas as pd
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

'''
trainning model
'''
df = pd.read_csv("Wine.csv")

X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)

rfc = RandomForestClassifier(n_estimators=20,max_leaf_nodes=16,n_jobs=1)
rfc.fit(X_train, y_train)

rfc_res = rfc.predict(X_test)
print(accuracy_score(y_test,rfc_res))

'''
lime explain
'''
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train), 
    feature_names=X_train.columns, 
    class_names=['bad', 'good'], 
    mode='classification')

idx = 8
data_test = np.array(X_test.iloc[idx]).reshape(1, -1)
prediction = rfc.predict(data_test)[0]
y_true = np.array(y_test)[idx]
print("id in testset: ",idx," rfc res: ", prediction," truth: ",y_true)

exp = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=rfc.predict_proba
)

exp_list = exp.as_list()
print(exp_list)