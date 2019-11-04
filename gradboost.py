from sklearn.ensemble import GradientBoostingClassifier

gbl = GradientBoostingClassifier(loss='deviance', learning_rate=0.001, n_estimators=300, subsample=1, min_samples_leaf=1, max_depth=2, random_state=42).fit(input_train, y_glb)
print("Accuracy on Gradient Boost training model: {:.3%}".format(gbl.score(input_train, y_glb)))

yhat_test_gbl = gbl.predict_proba(input_test)
yhat_test_gbl = pd.DataFrame(yhat_test_gbl)

df_actual_predicted_probs_gbl = pd.concat([loan_data_target_test, yhat_test_gbl.iloc[:,1]], axis=1)
df_actual_predicted_probs_gbl.columns = ['loan_data_targets_test','yhat_test_gbl']
df_actual_predicted_probs_gbl.index = loan_data_input_test.index
df_actual_predicted_probs_gbl.head(3)

tr = 0.5
df_actual_predicted_probs_gbl['yhat_test_proba_gbl'] = np.where(df_actual_predicted_probs_gbl['yhat_test_gbl'] > tr, 1, 0)
print(confusion_matrix(df_actual_predicted_probs_gbl['loan_data_targets_test'], df_actual_predicted_probs_gbl['yhat_test_proba_gbl']))
print(classification_report(df_actual_predicted_probs_gbl['loan_data_targets_test'], df_actual_predicted_probs_gbl['yhat_test_proba_gbl']))
fpr_gbl, tpr_gbl, thresholds_gbl = roc_curve(df_actual_predicted_probs_gbl['loan_data_targets_test'], df_actual_predicted_probs_gbl['yhat_test_gbl'])

auroc_gbl = roc_auc_score(df_actual_predicted_probs_gbl['loan_data_targets_test'], df_actual_predicted_probs_gbl['yhat_test_gbl'])
print("The Area Under the Curve for the Gradient Boost Model is: {:3f}".format(auroc_gbl))
