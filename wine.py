import numpy as np
import pandas as pd
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


'''
trainning model
'''
df = pd.read_csv("wine.csv")

X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
"""print(X_train.shape)
print(X_test.shape)"""

rfc = RandomForestClassifier(n_estimators=20,max_leaf_nodes=16,n_jobs=1)
gnb = GaussianNB()
tre = tree.DecisionTreeClassifier()
rfc.fit(X_train, y_train)
gnb.fit(X_train, y_train)
tre.fit(X_train, y_train)


rfc_res = rfc.predict(X_test)
gnb_res = gnb.predict(X_test)
print(accuracy_score(y_test,rfc_res))
print(accuracy_score(y_test,gnb_res))

'''
lime explain
'''
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train), 
    feature_names=X_train.columns, 
    class_names=['bad', 'good'], 
    mode='classification')

idx = 8
exp_rfc = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=rfc.predict_proba,
    num_features=10
)
exp_gnb = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=gnb.predict_proba,
    num_features=10
)
exp_tre = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=tre.predict_proba,
    num_features=10
)

exp_list = exp_rfc.as_list()

features = [feature for feature, weight in exp_list]
weights = [weight for feature, weight in exp_list]

# Create a color list to differentiate positive and negative weights
colors = ['green' if weight > 0 else 'red' for weight in weights]

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(features))
plt.barh(y_pos, weights, color=colors, edgecolor='black')
plt.yticks(y_pos, features)
plt.xlabel('Feature Weights')
plt.title('Feature Contribution to the RFC Model Prediction')

# Invert the y-axis to have the highest weight on top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()

exp_list_gnb = exp_gnb.as_list()

features = [feature for feature, weight in exp_list_gnb]
weights = [weight for feature, weight in exp_list_gnb]

# Create a color list to differentiate positive and negative weights
colors = ['green' if weight > 0 else 'red' for weight in weights]

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(features))
plt.barh(y_pos, weights, color=colors, edgecolor='black')
plt.yticks(y_pos, features)
plt.xlabel('Feature Weights')
plt.title('Feature Contribution to the GNB Model Prediction')

# Invert the y-axis to have the highest weight on top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()

exp_list_tre = exp_tre.as_list()

features = [feature for feature, weight in exp_list_tre]
weights = [weight for feature, weight in exp_list_tre]

# Create a color list to differentiate positive and negative weights
colors = ['green' if weight > 0 else 'red' for weight in weights]

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(features))
plt.barh(y_pos, weights, color=colors, edgecolor='black')
plt.yticks(y_pos, features)
plt.xlabel('Feature Weights')
plt.title('Feature Contribution to the Tree Model Prediction')

# Invert the y-axis to have the highest weight on top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()
""" def show_chart(model_name,exp):
    exp_list = exp.as_list()

    features = [feature for feature, weight in exp_list]
    weights = [weight for feature, weight in exp_list]

    # Create a color list to differentiate positive and negative weights
    colors = ['green' if weight > 0 else 'red' for weight in weights]

    # Create the horizontal bar chart
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, weights, color=colors, edgecolor='black')
    plt.yticks(y_pos, features)
    plt.xlabel('Feature Weights')
    plt.title('Feature Contribution to the ',model_name,' Model Prediction')

    # Invert the y-axis to have the highest weight on top
    plt.gca().invert_yaxis()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
show_chart("RFC",exp_rfc) """