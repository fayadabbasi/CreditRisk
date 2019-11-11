from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd  
import numpy as np   
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

class RandFor:

    def __init__(self):
        pass

    def randfor_action(self, xtr, xte, ytr, yte, tr=0.5, class_weight={0:10, 1:1}, n_estimators=5, max_depth=2):    
        '''
        Perform Random Forest fit and prediction and output key variables for determining performance of model
        OUTPUT: df_actual_predicted_probs, fpr, tpr, thresholds, auroc, and score
        '''
        y_glb = ytr

        r = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, random_state=42, n_jobs=-1)
        r.fit(xtr, y_glb)

        yhat_test = r.predict_proba(xte)
        yhat_test = pd.DataFrame(yhat_test) 

        df_actual_predicted_probs = pd.concat([yte, yhat_test.iloc[:,1]], axis=1)
        df_actual_predicted_probs.columns = ['y_test','yhat_test']
        df_actual_predicted_probs.index = xte.index

        score = r.score(xtr, y_glb)

        features = pd.DataFrame(r.feature_importances_,index = xtr.columns, columns=['importance']).sort_values('importance',ascending=False)

        df_actual_predicted_probs['yhat_test_proba'] = np.where(df_actual_predicted_probs['yhat_test'] > tr, 1, 0)

        fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test'])
        auroc = roc_auc_score(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test'])

        return df_actual_predicted_probs, fpr, tpr, thresholds, auroc, score, features


if __name__ == '__main__':

    X_train_woe = pd.read_csv('/home/ubuntu/X_train_woe_tt_cat.csv')
    X_test_woe = pd.read_csv('/home/ubuntu/X_test_woe_tt_cat.csv')
    y_train = pd.read_csv('/home/ubuntu/y_train_tt.csv', header=None)
    y_test = pd.read_csv('/home/ubuntu/y_test_tt.csv', header=None)

    X_train_woe.drop(['Unnamed: 0.1', 'Unnamed: 0', 'loan_amnt_factor'], axis=1, inplace=True)
    X_test_woe.drop(['Unnamed: 0.1', 'Unnamed: 0', 'loan_amnt_factor'], axis=1, inplace=True)

    ###### NEED TO CHECK FOR THE UNNAMED COLUMN IN POSITION 1 ###############
    ###### ALSO FOR THE RANDOM FOREST I NEED TO MAKE SURE I DO NOT DUMMIE EVERYTHING ########
    
    #### feature importance search

    r = RandFor()
    
    df_actual_predicted_probs, fpr, tpr, thresholds, auroc, score, features = r.randfor_action(X_train_woe, X_test_woe, y_train.iloc[:,1], y_test.iloc[:,1], tr=0.17, class_weight={0:1, 1:20}, n_estimators=70, max_depth=None)
    
    print("The random forest classifier score is is: {:3%}".format(score))
    print(confusion_matrix(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba']))
    print(classification_report(df_actual_predicted_probs['y_test'], df_actual_predicted_probs['yhat_test_proba']))
    print("The Area Under the Curve for the ROC is: {:3f}".format(auroc))



