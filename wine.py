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

features = [('density <= 0.99', 0,0), 
            ('chlorides <= 0.04', 0,0), 
            ('volatile acidity <= 0.21', 0,0), 
            ('10.40 < alcohol <= 11.40', 0,0), 
            ('residual sugar <= 1.70', 0,0), 
            ('pH <= 3.09', 0,0), 
            ('free sulfur dioxide <= 23.00', 0,0), 
            ('total sulfur dioxide <= 109.00', 0,0), 
            ('fixed acidity > 7.30', 0,0), 
            ('0.47 < sulphates <= 0.55', 0,0), 
            ('0.27 < citric acid <= 0.32', 0,0)]


local = {}

# Process a single data row
exp_rfc = explainer.explain_instance(
    data_row=X_test.iloc[0],  # Using only the first row for example
    predict_fn=rfc.predict_proba,
    num_features=11)

print("1",exp_rfc.as_list())

# Update the cumulative weights and counts
for feature, weight in exp_rfc.as_list():
    if feature in local:
        local[feature]['total_weight'] += weight
        local[feature]['count'] += 1
    else:
        local[feature] = {'total_weight': weight, 'count': 1}

# Calculate the average weights and create the updated_features list
local_point = []
for feature, data in local.items():
    average_weight = data['total_weight'] / data['count']
    local_point.append((feature, average_weight))        
                
for i in range(200):
    exp_rfc = explainer.explain_instance(
        data_row=X_test.iloc[i], 
        predict_fn=rfc.predict_proba,
        num_features=11)
    for feature, weight in exp_rfc.as_list(): 
        for idx, (feature_in_features, weight_in_features, count) in enumerate(features):
            if feature == feature_in_features:
                features[idx] = (feature_in_features, weight_in_features + weight, count + 1)

# update the weight by average and add feature, count to new tuple
global_ = []
for feature, weight, count in features:
    updated_weight = weight / count if count > 0 else 0
    global_.append((feature, updated_weight))


""" 
exp_gnb = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=gnb.predict_proba,
    num_features=10
)
exp_tre = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=tre.predict_proba,
    num_features=10
) """

def plot_feature_contributions(exp_rfc, figsize=(10, 6)):
    """
    Plots the feature contributions to an RFC model prediction.

    Parameters:
    exp_rfc: The explanation list from the RFC model, expected to be in the format [(feature, weight), ...]
    figsize: A tuple indicating the figure size
    """

    # Convert explanation list to separate lists of features and weights
    exp_list = exp_rfc.as_list()
    features = [feature for feature, weight in exp_list]
    weights = [weight for feature, weight in exp_list]

    # Create a color list to differentiate positive and negative weights
    colors = ['green' if weight > 0 else 'red' for weight in weights]

    # Create the horizontal bar chart
    plt.figure(figsize=figsize)
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

def plot_feature_contributions_list(exp_list, figsize=(10, 6)):
    features = [feature for feature, weight in exp_list]
    weights = [weight for feature, weight in exp_list]

    # Create a color list to differentiate positive and negative weights
    colors = ['green' if weight > 0 else 'red' for weight in weights]

    # Create the horizontal bar chart
    plt.figure(figsize=figsize)
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
# Example usage:
plot_feature_contributions_list(local_point)
plot_feature_contributions_list(global_)