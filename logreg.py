
class logreg:

    def __init__(self):
    '''
    I should make sure to drop the ref categories while creating the dummies
    '''
    #self.ref_categories = ['grade:G','home_ownership:RENT','purpose:wedding','emp_length_int:10.0','revol_util_factor:9','int_rate_factor:11','annual_inc_factor:7','loan_amnt_factor:8','term_int:60.0']
    self.X_train = loan_data_input_train.drop(ref_categories, axis=1)

    reg = LogisticRegression(class_weight={0:3, 1:1})
    reg.fit(X_train, y_train)

    def log_reg_summary_table(self, train_df,reg=reg):
        feature_name = train_df.columns.values
        summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
        summary_table['Coefficients'] = np.transpose(reg.coef_)
        summary_table.index = summary_table.index + 1
        summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
        summary_table = summary_table.sort_index()
        return summary_table

    print("Linear Regression training model score: {:.3%}".format(reg.score(input_train,loan_data_target_train)))

    input_test = loan_data_input_test
    input_test = input_test.drop(ref_categories, axis=1)
    yhat_test = reg.predict(input_test)
    yhat_test_proba = reg.predict_proba(input_test)
    yhat_test_proba = yhat_test_proba[: ][: ,1]
    loan_data_targets_test_temp = loan_data_target_test
    loan_data_targets_test_temp.reset_index(drop = True, inplace = True)
    df_actual_predicted_probs = pd.concat([loan_data_targets_test_temp, pd.DataFrame(yhat_test_proba)], axis=1)
    df_actual_predicted_probs.columns = ['loan_data_targets_test','yhat_test_proba']
    df_actual_predicted_probs.index = loan_data_input_test.index

    tr = 0.5
    df_actual_predicted_probs['yhat_test'] = np.where(df_actual_predicted_probs['yhat_test_proba'] > tr, 1, 0)
    pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['yhat_test'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['yhat_test']))
    print(classification_report(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['yhat_test']))

    fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['yhat_test_proba'])
    auroc = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['yhat_test_proba'])
    print("The Area Under the Curve for the ROC is: {:3f}".format(auroc))
