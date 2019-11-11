import numpy as np
import pandas as pd
import CreditRisk.preprocessing as p
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import scipy.stats as stat
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
import CreditRisk.ohe as ohe
import CreditRisk.woe as woe 

features_all = []


class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        #self.z_scores = z_scores
        self.p_values = p_values
        #self.sigma_estimates = sigma_estimates
        #self.F_ij = F_ij

class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """
    
    # nothing changes in __init__
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    
    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)
        
        # Calculate SSE (sum of squared errors)
        # and SE (standard error)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        # compute the t-statistic for each feature
        self.t = self.coef_ / se
        # find the p-value for each feature
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self

class LGD:
    def __init__(self, dummies):
        self.dummies = ['grade','home_ownership','verification_status','purpose','initial_list_status']

    def action(self, loan_data_preprocessed, tr=0.5):
        
        # will need to add back in recoveries and funded_amnt
        loan_data_defaults['recoveries'] = pd.to_numeric(loan_data_defaults['recoveries'])
        loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']
        
        loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
        loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])

        # will need to add back in tot_rec_prncp
        loan_data_defaults['tot_rec_prncp'] = pd.to_numeric(loan_data_defaults['total_rec_prncp'])

        loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'].astype('float')) - loan_data_defaults['total_rec_prncp'].astype('float') / loan_data_defaults['funded_amnt'].astype('float')

        loan_data_defaults['recovery_rate_0_1'] = np.where(loan_data_defaults['recovery_rate'] == 0, 0, 1)

        lgd_inputs_stage_1_train, lgd_inputs_stage_1_test, lgd_targets_stage_1_train, lgd_targets_stage_1_test = train_test_split(loan_data_defaults.drop(['recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['recovery_rate_0_1'], test_size = 0.2, random_state = 42)

        def loan_data_d(dataframe, dlist):    
            for items in dlist:
                loan_data_dummies = [pd.get_dummies(dataframe[items], prefix=items,prefix_sep=':')]
                loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
                dataframe = pd.concat([dataframe, loan_data_dummies], axis = 1)
            return dataframe
        
    
        lgd_inputs_stage_1_train = loan_data_d(lgd_inputs_stage_1_train, self.dummies)
        lgd_inputs_stage_1_test = loan_data_d(lgd_inputs_stage_1_test, self.dummies)

        lgd_inputs_stage_1_train = lgd_inputs_stage_1_train[features_all]
        lgd_inputs_stage_1_train = lgd_inputs_stage_1_train.drop(features_reference_cat, axis = 1)

        obj_conversion = ['dti','delinq_2yrs','inq_last_6mths','pub_rec','total_acc','acc_now_delinq']
        
        for items in obj_conversion:
            lgd_inputs_stage_1_train[items] = pd.to_numeric(lgd_inputs_stage_1_train[items])
        for items in obj_conversion:
            lgd_inputs_stage_1_test[items] = pd.to_numeric(lgd_inputs_stage_1_test[items])

        reg_lgd_st_1 = LogisticRegression_with_p_values()
        reg_lgd_st_1.fit(lgd_inputs_stage_1_train, lgd_targets_stage_1_train)
        feature_name = lgd_inputs_stage_1_train.columns.values
        
        lgd_inputs_stage_1_test = lgd_inputs_stage_1_test[features_all]

        lgd_inputs_stage_1_test = lgd_inputs_stage_1_test.drop(features_reference_cat, axis = 1)
        y_hat_test_lgd_stage_1 = reg_lgd_st_1.model.predict(lgd_inputs_stage_1_test)

        y_hat_test_proba_lgd_stage_1 = reg_lgd_st_1.model.predict_proba(lgd_inputs_stage_1_test)

        y_hat_test_proba_lgd_stage_1 = y_hat_test_proba_lgd_stage_1[: ][: , 1]

        lgd_targets_stage_1_test_temp = lgd_targets_stage_1_test
        lgd_targets_stage_1_test_temp.reset_index(drop = True, inplace = True)

        df_actual_predicted_probs = pd.concat([lgd_targets_stage_1_test_temp, pd.DataFrame(y_hat_test_proba_lgd_stage_1)], axis = 1)

        df_actual_predicted_probs.columns = ['lgd_targets_stage_1_test', 'y_hat_test_proba_lgd_stage_1']

        df_actual_predicted_probs.index = lgd_inputs_stage_1_test.index

        df_actual_predicted_probs['y_hat_test_lgd_stage_1'] = np.where(df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1'] > tr, 1, 0)

        fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1']
        auroc = roc_auc_score(df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1'])

        return summary_table, fpr, tpr, thresholds, auroc, df_actual_predicted_probs


    def action_2():
        lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]
        lgd_stage_2_data = loan_data_d(lgd_stage_2_data, self.dummies)
        lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop(['recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), lgd_stage_2_data['recovery_rate'], test_size = 0.2, random_state = 42)

        lgd_inputs_stage_2_train = lgd_inputs_stage_2_train[features_all]
        lgd_inputs_stage_2_train = lgd_inputs_stage_2_train.drop(features_reference_cat, axis = 1)

        for items in obj_conversion:
            lgd_inputs_stage_2_train[items] = pd.to_numeric(lgd_inputs_stage_2_train[items])

        for items in obj_conversion:
            lgd_inputs_stage_2_test[items] = pd.to_numeric(lgd_inputs_stage_2_test[items])

        reg_lgd_st_2 = LinearRegression()
        reg_lgd_st_2.fit(lgd_inputs_stage_2_train, lgd_targets_stage_2_train)

        feature_name = lgd_inputs_stage_2_train.columns.values

        summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
        summary_table['Coefficients'] = np.transpose(reg_lgd_st_2.coef_)
        summary_table.index = summary_table.index + 1
        summary_table.loc[0] = ['Intercept', reg_lgd_st_2.intercept_]
        summary_table = summary_table.sort_index()
        p_values = reg_lgd_st_2.p
        p_values = np.append(np.nan,np.array(p_values))
        summary_table['p_values'] = p_values.round(3)

        lgd_inputs_stage_2_test = lgd_inputs_stage_2_test[features_all]
        lgd_inputs_stage_2_test = lgd_inputs_stage_2_test.drop(features_reference_cat, axis = 1)

        y_hat_test_lgd_stage_2 = reg_lgd_st_2.predict(lgd_inputs_stage_2_test)

        lgd_targets_stage_2_test_temp = lgd_targets_stage_2_test

        lgd_targets_stage_2_test_temp = lgd_targets_stage_2_test_temp.reset_index(drop = True)

    def combined_action():
        y_hat_test_lgd_stage_2_all = reg_lgd_st_2.predict(lgd_inputs_stage_1_test)
        y_hat_test_lgd_stage_2_all
        y_hat_test_lgd = y_hat_test_lgd_stage_1 * y_hat_test_lgd_stage_2_all
        y_hat_test_lgd = np.where(y_hat_test_lgd < 0, 0, y_hat_test_lgd)
        y_hat_test_lgd = np.where(y_hat_test_lgd > 1, 1, y_hat_test_lgd)


if __name__ == '__main__':
    '''
    df_preprocessed_backup = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/df_preprocessed.csv',low_memory=False)
    df_preprocessed = df_preprocessed_backup.copy()
    
    ####### I can concatenate df_preprocessed with the X_train_woe_tt_cat or something like that #######



    pd.crosstab(df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_lgd_stage_1'], rownames = ['Actual'], colnames = ['Predicted'])
    print(auroc)
    # from the second action
    pd.concat([lgd_targets_stage_2_test_temp, pd.DataFrame(y_hat_test_lgd_stage_2)], axis = 1).corr()
    '''