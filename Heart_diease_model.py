import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import VotingClassifier
the_input = pd.read_csv('test_case.csv')
print(the_input)
# input should be formated the same as training data
def predictor(the_input):
fbs = the_input.at[0,'fbs']
print(fbs)
if fbs == 0:
    df = pd.read_csv('cleaned_merged_heart_dataset.csv')
    # 0 are those who fasted and their lab data should be good
    df = df[df['fbs'] == 0]

else:
    df = pd.read_csv('cleaned_merged_heart_dataset.csv')
    # 0 are those who fasted and their lab data should be good
    df = df[df['fbs'] == 1]
# all of these are discrete
x = df[['thalachh', 'oldpeak', 'trestbps', 'chol', 'age']]
y = df['target']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=53)
log1 = LogisticRegression(max_iter = 200).fit(train_x, train_y)
log1_pred = log1.predict(test_x)
# accuracy = accuracy_score(test_y, lin_pred)
x1 = df['ca']
x2 = df['sex']
y1 = df['target']

# for ca
train_x1, test_x1, train_y1, test_y1 = train_test_split(x1.values.reshape(-1, 1), y1, test_size=0.2,
                                                        random_state=53)
real_classifier = LogisticRegression()
onevsone = OneVsOneClassifier(real_classifier).fit(train_x1, train_y1)
# data is discrete but not binary so this secondary classifier is needed to circumvent that
# for sex
train_x2, test_x2, train_y2, test_y2 = train_test_split(x2.values.reshape(-1, 1), y1, test_size=0.2,
                                                        random_state=53)
log = real_classifier.fit(train_x, train_y2)
# combines models through stacking decided on final logreg model to stack all

x = df[['thalachh', 'oldpeak', 'trestbps', 'chol', 'age', 'ca', 'sex']]
y = df['target']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=53)

ensemble1 = VotingClassifier(estimators=[('OnevsOne', onevsone), ('Logistic Regression', log), ('log1', log1)],
                             voting='hard').fit(train_x, train_y)
    ensemble1_predict = ensemble1.predict(the_input)
    return ensemble1_predict
predictor(the_input)