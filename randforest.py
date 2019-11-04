from sklearn.ensemble import RandomForestClassifier

y_glb = loan_data_target_train_2.iloc[:,0]

rf = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight={0:10, 1:1}, random_state=42, n_jobs=-1).fit(input_train, y_glb)
print("The random forest classifier score is is: {:3%}".format(rf.score(input_train, y_glb)))

yhat_test_rf = rf.predict_proba(input_test)
yhat_test_rf = pd.DataFrame(yhat_test_rf)

df_actual_predicted_probs_rf = pd.concat([loan_data_target_test, yhat_test_rf.iloc[:,1]], axis=1)
df_actual_predicted_probs_rf.columns = ['loan_data_targets_test','yhat_test_rf']
df_actual_predicted_probs_rf.index = loan_data_input_test.index
df_actual_predicted_probs_rf.head(3)

tr = 0.7
df_actual_predicted_probs_rf['yhat_test_proba_rf'] = np.where(df_actual_predicted_probs_rf['yhat_test_rf'] > tr, 1, 0)

pd.crosstab(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_proba_rf'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_proba_rf']))
print(classification_report(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_proba_rf']))

fpr_rf, tpr_rf, thresholds_rf = roc_curve(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_rf'])
auroc_rf = roc_auc_score(df_actual_predicted_probs_rf['loan_data_targets_test'], df_actual_predicted_probs_rf['yhat_test_rf'])
print("The Area Under the Curve for the ROC is: {:3f}".format(auroc_rf))

