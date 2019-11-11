# By going through each category, I manually determined the coarse classification of the columns.
# While trying to determine an algorithmic way of doing this, I need to make sure I still do the same for the test set
# do not want to coarse classify something in the train set and find the test set is done differently
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt     

class WOE:
    def __init__(self):
        pass

    def woe_ordered_continuous(self, df, discrete_variable_name, good_bad_variable_df):
        '''
        Provides a dataframe to analyze the weight of evidence for selected inputs
        OUTPUT: dataframe
        '''
        df = pd.concat([df[discrete_variable_name], good_bad_variable_df],axis=1)
        df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
        df = df.iloc[:,[0,1,3]]
        df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
        df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
        df['n_good'] = df['prop_good'] * df['n_obs']
        df['n_bad'] = (1-df['prop_good']) * df['n_obs']
        df['prop_n_good'] = df['n_good']/ df['n_good'].sum()
        df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
        df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
        # df = df.sort_values(['WoE'])
        # df = df.reset_index(drop=True)
        df['diff_prop_good'] = df['prop_good'].diff().abs()
        df['diff_WoE'] = df['WoE'].diff().abs()
        df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
        df['IV'] = df['IV'].sum()
        return df

    def plot_by_woe(self, df_WoE, rotation_of_x_axis_label=0):
        x = np.array(df_WoE.iloc[:,0].apply(str))
        y = df_WoE['WoE']
        plt.figure(figsize=(18,6))
        plt.plot(x,y,marker='o', linestyle='--',color='k')
        plt.xlabel = (df_WoE.columns[0])
        plt.ylabel = ('Weight of Evidence')
        plt.title(str('Weight of Evidence by '+ df_WoE.columns[0]))
        plt.xticks(rotation=rotation_of_x_axis_label)
        # plot_by_woe(df_temp)

    def action_woe_cat(self, X_train):
        df_inputs_prepr = X_train

        if 'addr_state:OK' in X_train:
            df_inputs_prepr['addr_state:OK_AR_LA_MS'] = sum([df_inputs_prepr['addr_state:OK'],df_inputs_prepr['addr_state:AR'],df_inputs_prepr['addr_state:LA'],df_inputs_prepr['addr_state:MS']])
            df_inputs_prepr['addr_state:NV_NY'] = sum([df_inputs_prepr['addr_state:NV'],df_inputs_prepr['addr_state:NY']])
            df_inputs_prepr['addr_state:HI_FL_NM'] = sum([df_inputs_prepr['addr_state:HI'],df_inputs_prepr['addr_state:FL'],df_inputs_prepr['addr_state:NM']])
            df_inputs_prepr['addr_state:MD_MO_PA_NC_NJ_IN'] = sum([df_inputs_prepr['addr_state:MD'],df_inputs_prepr['addr_state:MO'],df_inputs_prepr['addr_state:PA'],df_inputs_prepr['addr_state:NC'],df_inputs_prepr['addr_state:NJ'],df_inputs_prepr['addr_state:IN']])
            df_inputs_prepr['addr_state:KY_CA'] = sum([df_inputs_prepr['addr_state:KY'],df_inputs_prepr['addr_state:CA']])
            df_inputs_prepr['addr_state:SD_NE_TN_MI_DE_VA'] = sum([df_inputs_prepr['addr_state:SD'],df_inputs_prepr['addr_state:NE'],df_inputs_prepr['addr_state:TN'],df_inputs_prepr['addr_state:MI'],df_inputs_prepr['addr_state:DE'],df_inputs_prepr['addr_state:VA']])
            df_inputs_prepr['addr_state:MN_AZ_TX_OH'] = sum([df_inputs_prepr['addr_state:MN'],df_inputs_prepr['addr_state:AZ'],df_inputs_prepr['addr_state:TX'],df_inputs_prepr['addr_state:OH']])
            df_inputs_prepr['addr_state:UT_GA_WI'] = sum([df_inputs_prepr['addr_state:UT'],df_inputs_prepr['addr_state:GA'],df_inputs_prepr['addr_state:WI']])
            df_inputs_prepr['addr_state:IL_CT_RI_MT'] = sum([df_inputs_prepr['addr_state:IL'],df_inputs_prepr['addr_state:CT'],df_inputs_prepr['addr_state:RI'],df_inputs_prepr['addr_state:MT']])
            df_inputs_prepr['addr_state:WY_KS_WA'] = sum([df_inputs_prepr['addr_state:WY'],df_inputs_prepr['addr_state:KS'],df_inputs_prepr['addr_state:WA']])
            df_inputs_prepr['addr_state:ND_CO'] = sum([df_inputs_prepr['addr_state:ND'],df_inputs_prepr['addr_state:CO']])
            df_inputs_prepr['addr_state:SC_OR_DC'] = sum([df_inputs_prepr['addr_state:SC'],df_inputs_prepr['addr_state:OR'],df_inputs_prepr['addr_state:DC']])
            df_inputs_prepr = df_inputs_prepr.drop(['addr_state:OK','addr_state:AR','addr_state:LA','addr_state:MS','addr_state:NV','addr_state:NY','addr_state:HI','addr_state:FL','addr_state:NM','addr_state:MD','addr_state:MO','addr_state:PA','addr_state:NC'], axis=1)
            df_inputs_prepr = df_inputs_prepr.drop(['addr_state:IN','addr_state:NJ','addr_state:KY','addr_state:CA','addr_state:SD','addr_state:NE','addr_state:TN','addr_state:MI','addr_state:DE','addr_state:VA','addr_state:MN','addr_state:AZ','addr_state:TX','addr_state:OH','addr_state:UT'], axis=1)
            df_inputs_prepr = df_inputs_prepr.drop(['addr_state:GA','addr_state:WI','addr_state:IL','addr_state:CT','addr_state:RI','addr_state:MT','addr_state:ND','addr_state:WY','addr_state:KS','addr_state:WA','addr_state:CO','addr_state:SC','addr_state:OR','addr_state:DC'], axis=1)
            # these are the individual states ['IA','AL','MA','NH','WV','ID','VT','ME']
        
        return df_inputs_prepr


    def action_woe(self, X_train):
        '''
        This manually assigned the appropriate range for Weight of Evidence for the selected one hot encoded categories

        OUTPUT: new train and test inputs and targets
        '''

        df_inputs_prepr = X_train
        if 'loan_amnt' in X_train:
            df_inputs_prepr['loan_amnt_factor'] = pd.cut(df_inputs_prepr['loan_amnt'],30)
            df_inputs_prepr['loan_amnt_factor:1'] = np.where((df_inputs_prepr['loan_amnt'].isin(range(1817))),1,0)
            df_inputs_prepr['loan_amnt_factor:2'] = np.where((df_inputs_prepr['loan_amnt'].isin(range(1817,7084))),1,0)
            df_inputs_prepr['loan_amnt_factor:3'] = np.where((df_inputs_prepr['loan_amnt'].isin(range(7084,11034))),1,0)
            df_inputs_prepr['loan_amnt_factor:4'] = np.where((df_inputs_prepr['loan_amnt'].isin(range(11034,int(df_inputs_prepr['loan_amnt'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['loan_amnt', 'loan_amnt_factor'], axis=1)

        if 'annual_inc' in X_train:
            df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'],100)
            df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc']<140000,:]
            df_inputs_prepr_temp['annual_inc_factor'] = pd.cut(df_inputs_prepr_temp['annual_inc'],20)
            df_inputs_prepr['annual_inc_factor:1'] = np.where((df_inputs_prepr['annual_inc'].isin(range(7000))),1,0)
            df_inputs_prepr['annual_inc_factor:2'] = np.where((df_inputs_prepr['annual_inc'].isin(range(7000,42000))),1,0)
            df_inputs_prepr['annual_inc_factor:3'] = np.where((df_inputs_prepr['annual_inc'].isin(range(42000,70000))),1,0)
            df_inputs_prepr['annual_inc_factor:4'] = np.where((df_inputs_prepr['annual_inc'].isin(range(70000,100000))),1,0)
            df_inputs_prepr['annual_inc_factor:5'] = np.where((df_inputs_prepr['annual_inc'].isin(range(100000,112000))),1,0)
            df_inputs_prepr['annual_inc_factor:6'] = np.where((df_inputs_prepr['annual_inc'].isin(range(112000,126000))),1,0)
            df_inputs_prepr['annual_inc_factor:7'] = np.where((df_inputs_prepr['annual_inc'].isin(range(126000,int(df_inputs_prepr['annual_inc'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['annual_inc','annual_inc_factor'], axis=1)
        
        if 'int_rate_int' in X_train:
            df_inputs_prepr['int_rate_factor_f'] = pd.cut(df_inputs_prepr['int_rate_int'],10)
            df_inputs_prepr['int_rate_factor:1'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(8))),1,0)
            df_inputs_prepr['int_rate_factor:2'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(8,11))),1,0)
            df_inputs_prepr['int_rate_factor:3'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(11,13))),1,0)
            df_inputs_prepr['int_rate_factor:4'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(13,15))),1,0)
            df_inputs_prepr['int_rate_factor:5'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(15,17))),1,0)
            df_inputs_prepr['int_rate_factor:6'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(17,19))),1,0)
            df_inputs_prepr['int_rate_factor:7'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(19,21))),1,0)
            df_inputs_prepr['int_rate_factor:8'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(21,23))),1,0)
            df_inputs_prepr['int_rate_factor:9'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(23,25))),1,0)
            df_inputs_prepr['int_rate_factor:10'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(25,27))),1,0)
            df_inputs_prepr['int_rate_factor:11'] = np.where((df_inputs_prepr['int_rate_int'].isin(range(27,int(df_inputs_prepr['int_rate_int'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['int_rate_int','int_rate_factor_f'], axis=1)

        if 'revol_util_int' in X_train:
            df_inputs_prepr['revol_util_factor_f'] = pd.cut(df_inputs_prepr['revol_util_int'],30)
            df_inputs_prepr['revol_util_factor:1'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(12))),1,0)
            df_inputs_prepr['revol_util_factor:2'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(12,24))),1,0)
            df_inputs_prepr['revol_util_factor:3'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(24,36))),1,0)
            df_inputs_prepr['revol_util_factor:4'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(36,48))),1,0)
            df_inputs_prepr['revol_util_factor:5'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(48,60))),1,0)
            df_inputs_prepr['revol_util_factor:6'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(60,72))),1,0)
            df_inputs_prepr['revol_util_factor:7'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(72,84))),1,0)
            df_inputs_prepr['revol_util_factor:8'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(84,96))),1,0)
            df_inputs_prepr['revol_util_factor:9'] = np.where((df_inputs_prepr['revol_util_int'].isin(range(96,int(df_inputs_prepr['revol_util_int'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['revol_util_int','revol_util_factor_f'], axis=1)

        if 'open_acc' in X_train:
            df_inputs_prepr['open_acc_t'] = pd.cut(df_inputs_prepr['open_acc'],30)
            df_inputs_prepr['open_acc:1'] = np.where((df_inputs_prepr['open_acc'].isin(range(3))),1,0)
            df_inputs_prepr['open_acc:2'] = np.where((df_inputs_prepr['open_acc'].isin(range(3,6))),1,0)
            df_inputs_prepr['open_acc:3'] = np.where((df_inputs_prepr['open_acc'].isin(range(6,10))),1,0)
            df_inputs_prepr['open_acc:4'] = np.where((df_inputs_prepr['open_acc'].isin(range(10,15))),1,0)
            df_inputs_prepr['open_acc:5'] = np.where((df_inputs_prepr['open_acc'].isin(range(15,20))),1,0)
            df_inputs_prepr['open_acc:6'] = np.where((df_inputs_prepr['open_acc'].isin(range(20,30))),1,0)
            df_inputs_prepr['open_acc:9'] = np.where((df_inputs_prepr['open_acc'].isin(range(30,int(df_inputs_prepr['open_acc'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['open_acc','open_acc_t'], axis=1)

        if 'zip_code_int' in X_train:
            df_inputs_prepr['zip_code_f'] = pd.cut(df_inputs_prepr['zip_code_int'],5)
            df_inputs_prepr['zip_code:1'] = np.where((df_inputs_prepr['zip_code_int'].isin(range(200))),1,0)
            df_inputs_prepr['zip_code:2'] = np.where((df_inputs_prepr['zip_code_int'].isin(range(200,400))),1,0)
            df_inputs_prepr['zip_code:3'] = np.where((df_inputs_prepr['zip_code_int'].isin(range(400,600))),1,0)
            df_inputs_prepr['zip_code:4'] = np.where((df_inputs_prepr['zip_code_int'].isin(range(600,800))),1,0)
            df_inputs_prepr['zip_code:5'] = np.where((df_inputs_prepr['zip_code_int'].isin(range(800,int(df_inputs_prepr['zip_code_int'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['zip_code_int','zip_code_f'], axis=1)
        
        if 'dti' in X_train:
            df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'],30)
            df_inputs_prepr['dti_factor:1'] = np.where((df_inputs_prepr['dti'].isin(range(8))),1,0)
            df_inputs_prepr['dti_factor:2'] = np.where((df_inputs_prepr['dti'].isin(range(8,12))),1,0)
            df_inputs_prepr['dti_factor:3'] = np.where((df_inputs_prepr['dti'].isin(range(12,14))),1,0)
            df_inputs_prepr['dti_factor:4'] = np.where((df_inputs_prepr['dti'].isin(range(14,16))),1,0)
            df_inputs_prepr['dti_factor:5'] = np.where((df_inputs_prepr['dti'].isin(range(16,20))),1,0)
            df_inputs_prepr['dti_factor:6'] = np.where((df_inputs_prepr['dti'].isin(range(20,25))),1,0)
            df_inputs_prepr['dti_factor:7'] = np.where((df_inputs_prepr['dti'].isin(range(25,int(df_inputs_prepr['dti'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['dti', 'dti_factor'], axis=1)

        if 'installment' in X_train:
            df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'],8)
            df_inputs_prepr['installment_factor:1'] = np.where((df_inputs_prepr['installment'].isin(range(180))),1,0)
            df_inputs_prepr['installment_factor:2'] = np.where((df_inputs_prepr['installment'].isin(range(180,355))),1,0)
            df_inputs_prepr['installment_factor:3'] = np.where((df_inputs_prepr['installment'].isin(range(355,530))),1,0)
            df_inputs_prepr['installment_factor:4'] = np.where((df_inputs_prepr['installment'].isin(range(530,706))),1,0)
            df_inputs_prepr['installment_factor:5'] = np.where((df_inputs_prepr['installment'].isin(range(706,880))),1,0)
            df_inputs_prepr['installment_factor:6'] = np.where((df_inputs_prepr['installment'].isin(range(880,1055))),1,0)
            df_inputs_prepr['installment_factor:7'] = np.where((df_inputs_prepr['installment'].isin(range(1055,int(df_inputs_prepr['installment'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['installment', 'installment_factor'], axis=1)

        if 'fico_range_low' in X_train:
            df_inputs_prepr['fico_range_low_factor'] = pd.cut(df_inputs_prepr['fico_range_low'],8)
            df_inputs_prepr['fico_range_low_factor:1'] = np.where((df_inputs_prepr['fico_range_low'].isin(range(680))),1,0)
            df_inputs_prepr['fico_range_low_factor:2'] = np.where((df_inputs_prepr['fico_range_low'].isin(range(680,700))),1,0)
            df_inputs_prepr['fico_range_low_factor:3'] = np.where((df_inputs_prepr['fico_range_low'].isin(range(700,730))),1,0)
            df_inputs_prepr['fico_range_low_factor:4'] = np.where((df_inputs_prepr['fico_range_low'].isin(range(730,760))),1,0)
            df_inputs_prepr['fico_range_low_factor:5'] = np.where((df_inputs_prepr['fico_range_low'].isin(range(760,int(df_inputs_prepr['fico_range_low'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['fico_range_low', 'fico_range_low_factor'], axis=1)

        if 'total_acc' in X_train:
            df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'],8)
            df_inputs_prepr['total_acc_factor:1'] = np.where((df_inputs_prepr['total_acc'].isin(range(18))),1,0)
            df_inputs_prepr['total_acc_factor:2'] = np.where((df_inputs_prepr['total_acc'].isin(range(18,25))),1,0)
            df_inputs_prepr['total_acc_factor:3'] = np.where((df_inputs_prepr['total_acc'].isin(range(25,30))),1,0)
            df_inputs_prepr['total_acc_factor:4'] = np.where((df_inputs_prepr['total_acc'].isin(range(30,38))),1,0)
            df_inputs_prepr['total_acc_factor:5'] = np.where((df_inputs_prepr['total_acc'].isin(range(38,int(df_inputs_prepr['total_acc'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['total_acc', 'total_acc_factor'], axis=1)

        if 'pub_rec' in X_train:
            df_inputs_prepr['pub_rec_factor'] = pd.cut(df_inputs_prepr['pub_rec'],8)
            df_inputs_prepr['pub_rec_factor:1'] = np.where((df_inputs_prepr['pub_rec'].isin(range(1))),1,0)
            df_inputs_prepr['pub_rec_factor:2'] = np.where((df_inputs_prepr['pub_rec'].isin(range(1,5))),1,0)
            df_inputs_prepr['pub_rec_factor:3'] = np.where((df_inputs_prepr['pub_rec'].isin(range(5,int(df_inputs_prepr['pub_rec'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['pub_rec', 'pub_rec_factor'], axis=1)


        if 'inq_last_6mths' in X_train:
            df_inputs_prepr['inq_last_6mths_factor'] = pd.cut(df_inputs_prepr['inq_last_6mths'],8)
            df_inputs_prepr['inq_last_6mths_factor:1'] = np.where((df_inputs_prepr['inq_last_6mths'].isin(range(1))),1,0)
            df_inputs_prepr['inq_last_6mths_factor:2'] = np.where((df_inputs_prepr['inq_last_6mths'].isin(range(1,4))),1,0)
            df_inputs_prepr['inq_last_6mths_factor:3'] = np.where((df_inputs_prepr['inq_last_6mths'].isin(range(4,int(df_inputs_prepr['inq_last_6mths'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['inq_last_6mths', 'inq_last_6mths_factor'], axis=1)

        if 'delinq_2yrs' in X_train:
            df_inputs_prepr['delinq_2yrs_factor'] = pd.cut(df_inputs_prepr['delinq_2yrs'],8)
            df_inputs_prepr['delinq_2yrs_factor:1'] = np.where((df_inputs_prepr['delinq_2yrs'].isin(range(1))),1,0)
            df_inputs_prepr['delinq_2yrs_factor:2'] = np.where((df_inputs_prepr['delinq_2yrs'].isin(range(1,4))),1,0)
            df_inputs_prepr['delinq_2yrs_factor:3'] = np.where((df_inputs_prepr['delinq_2yrs'].isin(range(4,int(df_inputs_prepr['delinq_2yrs'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['delinq_2yrs', 'delinq_2yrs_factor'], axis=1)

        if 'funded_amnt' in X_train:
            df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'],8)
            df_inputs_prepr['funded_amnt_factor:1'] = np.where((df_inputs_prepr['funded_amnt'].isin(range(8000))),1,0)
            df_inputs_prepr['funded_amnt_factor:2'] = np.where((df_inputs_prepr['funded_amnt'].isin(range(8000,12000))),1,0)
            df_inputs_prepr['funded_amnt_factor:3'] = np.where((df_inputs_prepr['funded_amnt'].isin(range(12000,20000))),1,0)
            df_inputs_prepr['funded_amnt_factor:4'] = np.where((df_inputs_prepr['funded_amnt'].isin(range(20000,int(df_inputs_prepr['funded_amnt'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['funded_amnt', 'funded_amnt_factor'], axis=1)


        if 'funded_amnt_inv' in X_train:
            df_inputs_prepr['funded_amnt_inv_factor'] = pd.cut(df_inputs_prepr['funded_amnt_inv'],8)
            df_inputs_prepr['funded_amnt_inv_factor:1'] = np.where((df_inputs_prepr['funded_amnt_inv'].isin(range(8000))),1,0)
            df_inputs_prepr['funded_amnt_inv_factor:2'] = np.where((df_inputs_prepr['funded_amnt_inv'].isin(range(8000,12000))),1,0)
            df_inputs_prepr['funded_amnt_inv_factor:3'] = np.where((df_inputs_prepr['funded_amnt_inv'].isin(range(12000,20000))),1,0)
            df_inputs_prepr['funded_amnt_inv_factor:4'] = np.where((df_inputs_prepr['funded_amnt_inv'].isin(range(20000,int(df_inputs_prepr['funded_amnt_inv'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['funded_amnt_inv', 'funded_amnt_inv_factor'], axis=1)

        if 'revol_bal' in X_train:
            df_inputs_prepr['revol_bal_factor'] = pd.cut(df_inputs_prepr['revol_bal'],8)
            df_inputs_prepr['revol_bal_factor:1'] = np.where((df_inputs_prepr['revol_bal'].isin(range(10))),1,0)
            df_inputs_prepr['revol_bal_factor:2'] = np.where((df_inputs_prepr['revol_bal'].isin(range(10,5000))),1,0)
            df_inputs_prepr['revol_bal_factor:3'] = np.where((df_inputs_prepr['revol_bal'].isin(range(5000,20000))),1,0)
            df_inputs_prepr['revol_bal_factor:4'] = np.where((df_inputs_prepr['revol_bal'].isin(range(20000,int(df_inputs_prepr['revol_bal'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['revol_bal', 'revol_bal_factor'], axis=1)

        if 'tax_liens' in X_train:
            df_inputs_prepr['tax_liens_factor'] = pd.cut(df_inputs_prepr['tax_liens'],8)
            df_inputs_prepr['tax_liens_factor:1'] = np.where((df_inputs_prepr['tax_liens'].isin(range(1))),1,0)
            df_inputs_prepr['tax_liens_factor:2'] = np.where((df_inputs_prepr['tax_liens'].isin(range(1,2))),1,0)
            df_inputs_prepr['tax_liens_factor:3'] = np.where((df_inputs_prepr['tax_liens'].isin(range(2,5))),1,0)
            df_inputs_prepr['tax_liens_factor:4'] = np.where((df_inputs_prepr['tax_liens'].isin(range(5,int(df_inputs_prepr['tax_liens'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['tax_liens', 'tax_liens_factor'], axis=1)

        if 'all_util' in X_train:
            df_inputs_prepr['all_util_factor'] = pd.cut(df_inputs_prepr['all_util'],8)
            df_inputs_prepr['all_util_factor:1'] = np.where((df_inputs_prepr['all_util'].isin(range(1))),1,0)
            df_inputs_prepr['all_util_factor:2'] = np.where((df_inputs_prepr['all_util'].isin(range(1,25))),1,0)
            df_inputs_prepr['all_util_factor:3'] = np.where((df_inputs_prepr['all_util'].isin(range(25,50))),1,0)
            df_inputs_prepr['all_util_factor:4'] = np.where((df_inputs_prepr['all_util'].isin(range(50,int(df_inputs_prepr['all_util'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['all_util', 'all_util_factor'], axis=1)

        if 'delinq_amnt' in X_train:
            df_inputs_prepr['delinq_amnt_factor'] = pd.cut(df_inputs_prepr['delinq_amnt'],8)
            df_inputs_prepr['delinq_amnt_factor:1'] = np.where((df_inputs_prepr['delinq_amnt'].isin(range(1))),1,0)
            df_inputs_prepr['delinq_amnt_factor:2'] = np.where((df_inputs_prepr['delinq_amnt'].isin(range(1,25))),1,0)
            df_inputs_prepr['delinq_amnt_factor:3'] = np.where((df_inputs_prepr['delinq_amnt'].isin(range(25,50))),1,0)
            df_inputs_prepr['delinq_amnt_factor:4'] = np.where((df_inputs_prepr['delinq_amnt'].isin(range(50,int(df_inputs_prepr['delinq_amnt'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['delinq_amnt', 'delinq_amnt_factor'], axis=1)

        if 'inq_fi' in X_train:
            df_inputs_prepr['inq_fi_factor'] = pd.cut(df_inputs_prepr['inq_fi'],8)
            df_inputs_prepr['inq_fi_factor:1'] = np.where((df_inputs_prepr['inq_fi'].isin(range(1))),1,0)
            df_inputs_prepr['inq_fi_factor:2'] = np.where((df_inputs_prepr['inq_fi'].isin(range(1,2))),1,0)
            df_inputs_prepr['inq_fi_factor:3'] = np.where((df_inputs_prepr['inq_fi'].isin(range(2,4))),1,0)
            df_inputs_prepr['inq_fi_factor:4'] = np.where((df_inputs_prepr['inq_fi'].isin(range(4,int(df_inputs_prepr['inq_fi'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['inq_fi', 'inq_fi_factor'], axis=1)
        
        if 'mths_since_last_delinq' in X_train:
            df_inputs_prepr['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr['mths_since_last_delinq'],8)
            df_inputs_prepr['mths_since_last_delinq_factor:1'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isin(range(1))),1,0)
            df_inputs_prepr['mths_since_last_delinq_factor:2'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isin(range(1,10))),1,0)
            df_inputs_prepr['mths_since_last_delinq_factor:3'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isin(range(10,50))),1,0)
            df_inputs_prepr['mths_since_last_delinq_factor:4'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isin(range(50,int(df_inputs_prepr['mths_since_last_delinq'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['mths_since_last_delinq', 'mths_since_last_delinq_factor'], axis=1)

        if 'num_accts_ever_120_pd' in X_train:
            df_inputs_prepr['num_accts_ever_120_pd_factor'] = pd.cut(df_inputs_prepr['num_accts_ever_120_pd'],8)
            df_inputs_prepr['num_accts_ever_120_pd_factor:1'] = np.where((df_inputs_prepr['num_accts_ever_120_pd'].isin(range(1))),1,0)
            df_inputs_prepr['num_accts_ever_120_pd_factor:2'] = np.where((df_inputs_prepr['num_accts_ever_120_pd'].isin(range(1,2))),1,0)
            df_inputs_prepr['num_accts_ever_120_pd_factor:3'] = np.where((df_inputs_prepr['num_accts_ever_120_pd'].isin(range(2,4))),1,0)
            df_inputs_prepr['num_accts_ever_120_pd_factor:4'] = np.where((df_inputs_prepr['num_accts_ever_120_pd'].isin(range(4,int(df_inputs_prepr['num_accts_ever_120_pd'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['num_accts_ever_120_pd', 'num_accts_ever_120_pd_factor'], axis=1)

        if 'num_actv_rev_tl' in X_train:
            df_inputs_prepr['num_actv_rev_tl_factor'] = pd.cut(df_inputs_prepr['num_actv_rev_tl'],8)
            df_inputs_prepr['num_actv_rev_tl_factor:1'] = np.where((df_inputs_prepr['num_actv_rev_tl'].isin(range(2))),1,0)
            df_inputs_prepr['num_actv_rev_tl_factor:2'] = np.where((df_inputs_prepr['num_actv_rev_tl'].isin(range(2,5))),1,0)
            df_inputs_prepr['num_actv_rev_tl_factor:3'] = np.where((df_inputs_prepr['num_actv_rev_tl'].isin(range(5,8))),1,0)
            df_inputs_prepr['num_actv_rev_tl_factor:4'] = np.where((df_inputs_prepr['num_actv_rev_tl'].isin(range(8,int(df_inputs_prepr['num_actv_rev_tl'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['num_actv_rev_tl', 'num_actv_rev_tl_factor'], axis=1)
        
        if 'num_rev_accts' in X_train:
            df_inputs_prepr['num_rev_accts_factor'] = pd.cut(df_inputs_prepr['num_rev_accts'],8)
            df_inputs_prepr['num_rev_accts_factor:1'] = np.where((df_inputs_prepr['num_rev_accts'].isin(range(5))),1,0)
            df_inputs_prepr['num_rev_accts_factor:2'] = np.where((df_inputs_prepr['num_rev_accts'].isin(range(5,10))),1,0)
            df_inputs_prepr['num_rev_accts_factor:3'] = np.where((df_inputs_prepr['num_rev_accts'].isin(range(10,20))),1,0)
            df_inputs_prepr['num_rev_accts_factor:4'] = np.where((df_inputs_prepr['num_rev_accts'].isin(range(20,int(df_inputs_prepr['num_rev_accts'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['num_rev_accts', 'num_rev_accts_factor'], axis=1)
        
        if 'total_rec_late_fee' in X_train:
            df_inputs_prepr['total_rec_late_fee_factor'] = pd.cut(df_inputs_prepr['total_rec_late_fee'],8)
            df_inputs_prepr['total_rec_late_fee_factor:1'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(1))),1,0)
            df_inputs_prepr['total_rec_late_fee_factor:2'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(1,15))),1,0)
            df_inputs_prepr['total_rec_late_fee_factor:3'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(15,30))),1,0)
            df_inputs_prepr['total_rec_late_fee_factor:4'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(30,int(df_inputs_prepr['total_rec_late_fee'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['total_rec_late_fee', 'total_rec_late_fee_factor'], axis=1)
        
        if 'loan_amnt' in X_train:
            df_inputs_prepr['total_rec_late_fee_factor'] = pd.cut(df_inputs_prepr['total_rec_late_fee'],8)
            df_inputs_prepr['total_rec_late_fee_factor:1'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(1))),1,0)
            df_inputs_prepr['total_rec_late_fee_factor:2'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(1,15))),1,0)
            df_inputs_prepr['total_rec_late_fee_factor:3'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(15,30))),1,0)
            df_inputs_prepr['total_rec_late_fee_factor:4'] = np.where((df_inputs_prepr['total_rec_late_fee'].isin(range(30,int(df_inputs_prepr['total_rec_late_fee'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['total_rec_late_fee', 'total_rec_late_fee_factor'], axis=1)

        if 'acc_now_delinq' in X_train:
            df_inputs_prepr['acc_now_delinq_factor'] = pd.cut(df_inputs_prepr['acc_now_delinq'],8)
            df_inputs_prepr['acc_now_delinq_factor:1'] = np.where((df_inputs_prepr['acc_now_delinq'].isin(range(1))),1,0)
            df_inputs_prepr['acc_now_delinq_factor:2'] = np.where((df_inputs_prepr['acc_now_delinq'].isin(range(1,int(df_inputs_prepr['acc_now_delinq'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['acc_now_delinq', 'acc_now_delinq_factor'], axis=1)
        
        if 'chargeoff_within_12_mths' in X_train:
            df_inputs_prepr['chargeoff_within_12_mths_factor'] = pd.cut(df_inputs_prepr['chargeoff_within_12_mths'],8)
            df_inputs_prepr['chargeoff_within_12_mths_factor:1'] = np.where((df_inputs_prepr['chargeoff_within_12_mths'].isin(range(1))),1,0)
            df_inputs_prepr['chargeoff_within_12_mths_factor:2'] = np.where((df_inputs_prepr['chargeoff_within_12_mths'].isin(range(1,int(df_inputs_prepr['chargeoff_within_12_mths'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['chargeoff_within_12_mths', 'chargeoff_within_12_mths_factor'], axis=1)
        
        if 'pub_rec_bankruptcies' in X_train:
            df_inputs_prepr['pub_rec_bankruptcies_factor'] = pd.cut(df_inputs_prepr['pub_rec_bankruptcies'],8)
            df_inputs_prepr['pub_rec_bankruptcies_factor:1'] = np.where((df_inputs_prepr['pub_rec_bankruptcies'].isin(range(1))),1,0)
            df_inputs_prepr['pub_rec_bankruptcies_factor:2'] = np.where((df_inputs_prepr['pub_rec_bankruptcies'].isin(range(1,int(df_inputs_prepr['pub_rec_bankruptcies'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['pub_rec_bankruptcies', 'pub_rec_bankruptcies_factor'], axis=1)

        if 'collections_12_mths_ex_med' in X_train:
            df_inputs_prepr['collections_12_mths_ex_med_factor'] = pd.cut(df_inputs_prepr['collections_12_mths_ex_med'],8)
            df_inputs_prepr['collections_12_mths_ex_med_factor:1'] = np.where((df_inputs_prepr['collections_12_mths_ex_med'].isin(range(1))),1,0)
            df_inputs_prepr['collections_12_mths_ex_med_factor:2'] = np.where((df_inputs_prepr['collections_12_mths_ex_med'].isin(range(1,int(df_inputs_prepr['collections_12_mths_ex_med'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['collections_12_mths_ex_med', 'collections_12_mths_ex_med_factor'], axis=1)


        return df_inputs_prepr


if __name__ == '__main__':
    
    X_train_ohe = pd.read_csv('/home/ubuntu/X_train_ohe_tt.csv')
    X_test_ohe = pd.read_csv('/home/ubuntu/X_test_ohe_tt.csv')

    woe = WOE()

    X_train_woe_tt_cat = woe.action_woe_cat(X_train_ohe)
    X_test_woe_tt_cat = woe.action_woe_cat(X_test_ohe)

    X_train_woe_tt = woe.action_woe(X_train_woe_tt_cat)
    X_test_woe_tt = woe.action_woe(X_test_woe_tt_cat)

    X_train_woe_tt_cat.to_csv('/home/ubuntu/X_train_woe_tt_cat.csv')
    X_test_woe_tt_cat.to_csv('/home/ubuntu/X_test_woe_tt_cat.csv')

    X_train_woe_tt.to_csv('/home/ubuntu/X_train_woe_tt.csv')
    X_test_woe_tt.to_csv('/home/ubuntu/X_test_woe_tt.csv')
    
    print('MISSION ACCOMPLISHED!!!')
