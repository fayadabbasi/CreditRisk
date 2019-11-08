import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def action_ead(ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test):
    reg_ead = LinearRegression()
    reg_ead.fit(ead_inputs_train, ead_targets_train)

    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg_ead.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg_ead.intercept_]
    summary_table = summary_table.sort_index()
    p_values = reg_ead.p
    p_values = np.append(np.nan,np.array(p_values))
    summary_table['p_values'] = p_values
    
    ead_inputs_test = ead_inputs_test[features_all]

    ead_inputs_test = ead_inputs_test.drop(features_reference_cat, axis = 1)

    y_hat_test_ead = reg_ead.predict(ead_inputs_test)

    ead_targets_test_temp = ead_targets_test
    ead_targets_test_temp = ead_targets_test_temp.reset_index(drop = True)

    pd.concat([ead_targets_test_temp, pd.DataFrame(y_hat_test_ead)], axis = 1).corr()

    y_hat_test_ead = np.where(y_hat_test_ead < 0, 0, y_hat_test_ead)
    y_hat_test_ead = np.where(y_hat_test_ead > 1, 1, y_hat_test_ead)

def action_expected_loss():
    # need to make sure I have loan_data_preprocessed in a state that it can be handled - I am including the steps I needed to take in the Jupyter notebook but this should be imported correctly

    loan_data_preprocessed_lgd_ead = loan_data_d(loan_data_preprocessed, dummies)
    for items in obj_conversion:
        loan_data_preprocessed_lgd_ead[items] = pd.to_numeric(loan_data_preprocessed_lgd_ead[items])
    loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead[features_all]
    loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(features_reference_cat, axis = 1)

    # from here on, this is what I want action_expected_loss to do

    loan_data_preprocessed['recovery_rate_st_1'] = reg_lgd_st_1.model.predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed['recovery_rate_st_2'] = reg_lgd_st_2.predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed['recovery_rate'] = loan_data_preprocessed['recovery_rate_st_1'] * loan_data_preprocessed['recovery_rate_st_2']

    loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['recovery_rate'] < 0, 0, loan_data_preprocessed['recovery_rate'])
    loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['recovery_rate'] > 1, 1, loan_data_preprocessed['recovery_rate'])

    loan_data_preprocessed['LGD'] = 1 - loan_data_preprocessed['recovery_rate']
    
    loan_data_preprocessed['CCF'] = reg_ead.predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] < 0, 0, loan_data_preprocessed['CCF'])
    loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] > 1, 1, loan_data_preprocessed['CCF'])

    loan_data_preprocessed['EAD'] = loan_data_preprocessed['CCF'] * loan_data_preprocessed_lgd_ead['funded_amnt']

    '''
    this is importing the originally built files
    loan_data_inputs_train = pd.read_csv('loan_data_inputs_train.csv')

    loan_data_inputs_test = pd.read_csv('loan_data_inputs_test.csv')

    '''
    loan_data_inputs_pd = pd.concat([loan_data_inputs_train, loan_data_inputs_test], axis = 0)

    loan_data_inputs_pd = loan_data_inputs_pd.set_index('Unnamed: 0')

    

loan_data_inputs_pd_temp = loan_data_inputs_pd[features_all_pd]
loan_data_inputs_pd_temp = loan_data_inputs_pd_temp.drop(ref_categories_pd, axis = 1)

'''
reg_pd = pickle.load(open('pd_model.sav', 'rb'))
# Here I import the PD model, stored in the 'pd_model.sav' file.
'''

reg_pd.model.predict_proba(loan_data_inputs_pd_temp)[: ][: , 0]
loan_data_inputs_pd['PD'] = reg_pd.model.predict_proba(loan_data_inputs_pd_temp)[: ][: , 0]

loan_data_preprocessed_new = pd.concat([loan_data_preprocessed, loan_data_inputs_pd], axis = 1)

loan_data_preprocessed_new['EL'] = loan_data_preprocessed_new['PD'] * loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['EAD']

# NOW I CAN ANALYZE THE DATA

loan_data_preprocessed_new['EL'].sum()
# Total Expected Loss for all loans.

loan_data_preprocessed_new['funded_amnt'].sum()
# Total funded amount for all loans.

loan_data_preprocessed_new['EL'].sum() / loan_data_preprocessed_new['funded_amnt'].sum()
# Total Expected Loss as a proportion of total funded amount for all loans.





if __name__ == '__main__':
    '''
    IMPORT THE RIGHT FILES TO OPERATE ON
    '''
    
    #ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop(['recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['CCF'], test_size = 0.2, random_state = 42)
    
    '''
    ead_inputs_train = loan_data_d(ead_inputs_train, dummies)
    ead_inputs_test = loan_data_d(ead_inputs_test, dummies)
    for items in obj_conversion:
        ead_inputs_train[items] = pd.to_numeric(ead_inputs_train[items])
    for items in obj_conversion:
        ead_inputs_test[items] = pd.to_numeric(ead_inputs_test[items])
    # I DO THIS IN MY NOTEBOOK BUT THIS SHOULD ALREADY BE DONE


    '''


    pass