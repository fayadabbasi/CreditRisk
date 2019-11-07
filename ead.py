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

    features_all_pd = ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>=86']

ref_categories_pd = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

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