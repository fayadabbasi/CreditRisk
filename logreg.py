from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import linear_model
import scipy.stats as stat
import numpy as np   
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


class LogReg:

    def __init__(self):
        pass

    def logreg_action(self, xtr, xte, ytr, yte, tr=0.5, class_weight={0:10, 1:1}):
        '''
        perform logistic regression fit and prediction and output key variables
        OUTPUT: df_actual_predicted_probs, fpr, trp, thresholds, auroc, and score
        '''
        
        reg = LogisticRegression(class_weight=class_weight)
        rg = reg.fit(xtr, ytr)
        yhat_test = rg.predict(xte)
        yhat_test = pd.DataFrame(yhat_test)

        yhat_test_proba = rg.predict_proba(xte)
        yhat_test_proba = pd.DataFrame(yhat_test_proba)

        yte.reset_index(drop = True, inplace = True)
        df_actual_predicted_probs = pd.concat([yte, yhat_test_proba.iloc[:,1]], axis=1)
        df_actual_predicted_probs.columns = ['y_test','yhat_test_proba']
        df_actual_predicted_probs.index = xte.index
        
        score = rg.score(xtr, ytr)

        df_actual_predicted_probs['yhat_test_proba'] = np.where(df_actual_predicted_probs['yhat_test_proba'] > tr, 1, 0)
        
        fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba'])
        auroc = roc_auc_score(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba'])

        return df_actual_predicted_probs, fpr, tpr, thresholds, auroc, score

    


if __name__ == '__main__':
    
    # X_train_woe = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train_woe_tt.csv')
    # X_test_woe = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test_woe_tt.csv')
    # y_train = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_train_tt.csv', header=None)
    # y_test = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_test_tt.csv', header=None)

    X_train_woe = pd.read_csv('/home/ubuntu/X_train_woe_tt.csv')
    X_test_woe = pd.read_csv('/home/ubuntu/X_test_woe_tt.csv')
    y_train = pd.read_csv('/home/ubuntu/y_train_tt.csv', header=None)
    y_test = pd.read_csv('/home/ubuntu/y_test_tt.csv', header=None)


    X_train_woe.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
    X_test_woe.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)


    lr = LogReg()
    df_actual_predicted_probs, fpr, tpr, thresholds, auroc, score = lr.logreg_action(X_train_woe, X_test_woe, y_train.iloc[:,1], y_test.iloc[:,1], tr=0.5, class_weight={0:6, 1:1})

    print(df_actual_predicted_probs.shape)
    print(df_actual_predicted_probs.head(2))
    print("The logistic regression classifier score is: {:3%}".format(score))
    print(confusion_matrix(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba']))
    print(classification_report(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba']))
    print("The Area Under the Curve for the ROC is: {:3f}".format(auroc))