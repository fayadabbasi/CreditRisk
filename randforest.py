from sklearn.ensemble import RandomForestClassifier
import pandas as pd  
import numpy as np   
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

class RandFor:

    def __init__(self):
        pass

    def randfor_action(self, xtr, xte, ytr, yte, tr=0.5, class_weight={0:10, 1:1}, n_estimators=50, max_depth=None):    
        '''
        Perform Random Forest fit and prediction and output key variables for determining performance of model
        OUTPUT: df_actual_predicted_probs, fpr, tpr, thresholds, auroc, and score
        '''
        
        y_glb = ytr.iloc[:,0]

        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, random_state=42, n_jobs=-1).fit(xtr, y_glb)
        # r = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, random_state=42, n_jobs=-1)
        # rfit = r.fit(xtr, ytr)
        #print("The random forest classifier score is is: {:3%}".format(rf.score(input_train, y_glb)))

        yhat_test = rf.predict_proba(xte)
        yhat_test = pd.DataFrame(yhat_test) 

        df_actual_predicted_probs = pd.concat([yte, yhat_test.iloc[:,1]], axis=1)
        df_actual_predicted_probs.columns = ['y_test','yhat_test']
        df_actual_predicted_probs.index = xte.index
        #df_actual_predicted_probs.head(3)

        score = rf.score(xtr, y_glb)
        # score = rfit.score(xtr, ytr)

        df_actual_predicted_probs['yhat_test_proba'] = np.where(df_actual_predicted_probs['yhat_test'] > tr, 1, 0)
        # TODO: BE SURE TO FLIP BAD AND GOOD FROM 1 to 0

        #pd.crosstab(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_proba_rf'], rownames=['Actual'], colnames=['Predicted'])
        #print(confusion_matrix(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_proba_rf']))
        #print(classification_report(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_proba_rf']))

        fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test'])
        auroc = roc_auc_score(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test'])
        #print("The Area Under the Curve for the ROC is: {:3f}".format(auroc_rf))

        return df_actual_predicted_probs, fpr, tpr, thresholds, auroc, score


if __name__ == '__main__':
    
    # X_train_woe = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train_woe.csv')
    # X_test_woe = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test_woe.csv')
    # y_train = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_train.csv', header=None)
    # y_test = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_test.csv', header=None)
    
    X_train_woe = pd.read_csv('/home/ubuntu/X_train_woe.csv')
    X_test_woe = pd.read_csv('/home/ubuntu/X_test_woe.csv')
    y_train = pd.read_csv('/home/ubuntu/y_train.csv', header=None)
    y_test = pd.read_csv('/home/ubuntu/y_test.csv', header=None)

    ###### NEED TO CHECK FOR THE UNNAMED COLUMN IN POSITION 1 ###############
    ###### ALSO FOR THE RANDOM FOREST I NEED TO MAKE SURE I DO NOT DUMMIE EVERYTHING ########

    r = RandFor()
    df_actual_predicted_probs, fpr, tpr, thresholds, auroc, score = r.randfor_action(X_train_woe, X_test_woe, y_train, y_test, tr=0.5, class_weight={0:10, 1:1}, n_estimators=10, max_depth=2)
    print("The random forest classifier score is is: {:3%}".format(score))
    print(confusion_matrix(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba']))
    print(classification_report(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba']))
    print("The Area Under the Curve for the ROC is: {:3f}".format(auroc))