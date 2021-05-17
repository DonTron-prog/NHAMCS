import os
import unidecode
import pandas as pd
import seaborn as sns
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm


#import list of relevent features
file = open("features.txt")
features = file.read().splitlines()
file.close()

ed2011 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2011.csv')))
ed2012 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2012.csv')))
ed2013 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2013.csv')))
ed2014 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2014.csv')))
ed2015 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2015.csv')))
ed2016 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2016.csv')))
ed2017 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2017.csv')))
ed2018 = pd.DataFrame(pd.read_csv(
    os.path.join('data', 'nhamcs2018.csv')))

frames = [
    ed2011,
    ed2012,
    ed2013,
    ed2014,
    ed2015,
    ed2016,
    ed2017,
    ed2018,
]
dataset = pd.concat(frames, join='outer', ignore_index=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Dataset with required attributes
nhamcs = dataset[features]

print (nhamcs.head(5))
print(nhamcs.shape)

import rfv #reason for visit list
nhamcs = nhamcs[nhamcs["RFV1"].isin(rfv.rfv1)]
#nhamcs = nhamcs[nhamcs["RFV2"].isin(rfv.rfv1)]
#nhamcs = nhamcs[nhamcs["RFV3"].isin(rfv.rfv1)]

#create numpy arrayof the lable values for model
nhamcs['DIAG1'] = nhamcs['DIAG1'].map({'J101': 1,
                                       'J111': 1})
nhamcs['DIAG1'] = pd.to_numeric(nhamcs['DIAG1'], errors='coerce').fillna(0, downcast='infer')#docast infer makes it integer
labels = np.array(nhamcs['DIAG1'])
print(labels)

'''
labels = nhamcs['DIAG1']
enc_l = preprocessing.LabelEncoder()
enc_l.fit(labels)
labels = enc_l.transform(labels)#make a numby array of features for sklearn
#print(labels)
#labels = np.array(nhamcs['DIAG1'])
'''

#make model matrix - encode string variables as integers then into dummy variables, append to model matrix
def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

categorical_columns = ['RFV1', 'RFV2', 'RFV3']

Features = encode_string(nhamcs['SEX'])
for col in categorical_columns:
    temp = encode_string(nhamcs[col])
    Features = np.concatenate([Features, temp], axis = 1)

print(Features.shape)
print(nhamcs['AGE'].shape)
print(Features[:2, :])


#concatenate the numeric features to the numpy array
Features = np.concatenate([Features, np.array(nhamcs[['VMONTH', 'AGE']])], axis = 1)
print(Features.shape)
print(Features[:2,:])

#Randomly sample cases to create independent training and test data (bernoulli sampleing)
nr.seed(9977)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx,test_size = 300)
X_train = Features[indx[0], :]
Y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
Y_test = np.ravel(labels[indx[1]])

#rescale to insure a similar numeric range for only training data (Zscore)
scaler = preprocessing.StandardScaler().fit(X_train[:,34:])
X_train[:,34:] = scaler.transform(X_train[:,34:])
X_test[:,34:] = scaler.transform(X_test[:,34:])
print(X_train[:2,])

#compute the logistic regression model
#1 define model, 2)fit the linear model using the np arrays of features and the labels for the training data set

logistic_mod = linear_model.LogisticRegression() #define logistic regression model object
logistic_mod.fit(X_train, Y_train)

print(logistic_mod.intercept_)
print(logistic_mod.coef_)

#probabilities for each class
probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

#score and evaluate the classification model set threshold between 1(ture) and 0(false) at 0.5
def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)#threshold of 50%

#see if the predictions agree with the test lables
print(np.array(scores[:74]))
print(Y_test[:74])

#clssifier performance metrics
def print_matrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)#compute code metrics
    conf = sklm.confusion_matrix(labels, scores)#compute confusion_matrix
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

print_matrics(Y_test, scores)

def plot_auc(labels, probs):
    ##compute the false positive rate, true positive rate
    ##and threshold allong with the AUC
    fpr, tpr, treshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#plot_auc(Y_test, probabilities)
'''
#classify all as not having influenza
probs_positive = np.concatenate((np.ones((probabilities.shape[0], 1)),
                                 np.zeros((probabilities.shape[0], 1))),
                                 axis = 1)
scores_positive = score_model(probs_positive, 0.5)
print_matrics(Y_test, scores_positive)
'''
#the results are over biasing the other diagnositcs at the expence of influenza because of class imbalance,
#the results must be weighted towards influenza
logistic_mod = linear_model.LogisticRegression(class_weight={0:0.10, 1:0.90})#Weight classes in logistic regression model
logistic_mod.fit(X_train, Y_train)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

scores = score_model(probabilities, 0.5)
print_matrics(Y_test, scores)
plot_auc(Y_test, probabilities)

#find a batter threshold to idnetify bad credit cases
def test_threshold(probs, labels, threshold):
    scores = score_model(probs, threshold)
    print('')
    print('For threshold = ' + str(threshold))
    print_matrics(labels, scores)

thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
for t in thresholds:
    test_threshold(probabilities, Y_test, t)
