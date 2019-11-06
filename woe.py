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

    def action_woe(self, X_train, X_test):
        '''
        This manually assigned the appropriate range for Weight of Evidence for the selected one hot encoded categories

        OUTPUT: new train and test inputs and targets
        '''

        df_inputs_prepr = X_train
        if 'loan_amnt' in X_train:
            df_inputs_prepr['loan_amnt_factor'] = pd.cut(df_inputs_prepr['loan_amnt'],30)
            df_inputs_prepr['loan_amnt_factor:1'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(1817))),1,0)
            df_inputs_prepr['loan_amnt_factor:2'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(1817,7084))),1,0)
            df_inputs_prepr['loan_amnt_factor:3'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(7084,11034))),1,0)
            df_inputs_prepr['loan_amnt_factor:4'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(11034,28150))),1,0)
            df_inputs_prepr['loan_amnt_factor:5'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(28150,28160))),1,0)
            df_inputs_prepr['loan_amnt_factor:6'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(28160,32100))),1,0)
            df_inputs_prepr['loan_amnt_factor:7'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(32100,36050))),1,0)
            df_inputs_prepr['loan_amnt_factor:8'] = np.where((df_inputs_prepr['loan_amnt_factor'].isin(range(36050,int(df_inputs_prepr['loan_amnt'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['loan_amnt', 'loan_amnt_factor'], axis=1)
        else:
            pass 

        if 'annual_inc' in X_train:
            df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'],100)
            df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc']<140000,:]
            df_inputs_prepr_temp['annual_inc_factor'] = pd.cut(df_inputs_prepr_temp['annual_inc'],20)
            df_inputs_prepr['annual_inc_factor:1'] = np.where((df_inputs_prepr['annual_inc_factor'].isin(range(7000))),1,0)
            df_inputs_prepr['annual_inc_factor:2'] = np.where((df_inputs_prepr['annual_inc_factor'].isin(range(7000,42000))),1,0)
            df_inputs_prepr['annual_inc_factor:3'] = np.where((df_inputs_prepr['annual_inc_factor'].isin(range(42000,70000))),1,0)
            df_inputs_prepr['annual_inc_factor:4'] = np.where((df_inputs_prepr['annual_inc_factor'].isin(range(70000,100000))),1,0)
            df_inputs_prepr['annual_inc_factor:5'] = np.where((df_inputs_prepr['annual_inc_factor'].isin(range(100000,112000))),1,0)
            df_inputs_prepr['annual_inc_factor:6'] = np.where((df_inputs_prepr['annual_inc_factor'].isin(range(112000,126000))),1,0)
            df_inputs_prepr['annual_inc_factor:7'] = np.where((df_inputs_prepr['annual_inc_factor'].isin(range(126000,int(df_inputs_prepr['annual_inc'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['annual_inc','annual_inc_factor'], axis=1)
        else:
            pass
        if 'int_rate_int' in X_train:
            df_inputs_prepr['int_rate_factor_f'] = pd.cut(df_inputs_prepr['int_rate_int'],10)
            df_inputs_prepr['int_rate_factor:1'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(8))),1,0)
            df_inputs_prepr['int_rate_factor:2'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(8,11))),1,0)
            df_inputs_prepr['int_rate_factor:3'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(11,13))),1,0)
            df_inputs_prepr['int_rate_factor:4'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(13,15))),1,0)
            df_inputs_prepr['int_rate_factor:5'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(15,17))),1,0)
            df_inputs_prepr['int_rate_factor:6'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(17,19))),1,0)
            df_inputs_prepr['int_rate_factor:7'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(19,21))),1,0)
            df_inputs_prepr['int_rate_factor:8'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(21,23))),1,0)
            df_inputs_prepr['int_rate_factor:9'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(23,25))),1,0)
            df_inputs_prepr['int_rate_factor:10'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(25,27))),1,0)
            df_inputs_prepr['int_rate_factor:11'] = np.where((df_inputs_prepr['int_rate_factor_f'].isin(range(27,int(df_inputs_prepr['int_rate_int'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['int_rate_int','int_rate_factor_f'], axis=1)
        else: 
            pass

        if 'revol_util_int' in X_train:
            df_inputs_prepr['revol_util_factor_f'] = pd.cut(df_inputs_prepr['revol_util_int'],30)
            df_inputs_prepr['revol_util_factor:1'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(12))),1,0)
            df_inputs_prepr['revol_util_factor:2'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(12,24))),1,0)
            df_inputs_prepr['revol_util_factor:3'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(24,36))),1,0)
            df_inputs_prepr['revol_util_factor:4'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(36,48))),1,0)
            df_inputs_prepr['revol_util_factor:5'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(48,60))),1,0)
            df_inputs_prepr['revol_util_factor:6'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(60,72))),1,0)
            df_inputs_prepr['revol_util_factor:7'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(72,84))),1,0)
            df_inputs_prepr['revol_util_factor:8'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(84,96))),1,0)
            df_inputs_prepr['revol_util_factor:9'] = np.where((df_inputs_prepr['revol_util_factor_f'].isin(range(96,int(df_inputs_prepr['revol_util_int'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['revol_util_int','revol_util_factor_f'], axis=1)
        else:
            pass

        if 'open_acc' in X_train:
            df_inputs_prepr['open_acc_t'] = pd.cut(df_inputs_prepr['open_acc'],30)
            df_inputs_prepr['open_acc:1'] = np.where((df_inputs_prepr['open_acc_t'].isin(range(3))),1,0)
            df_inputs_prepr['open_acc:2'] = np.where((df_inputs_prepr['open_acc_t'].isin(range(3,6))),1,0)
            df_inputs_prepr['open_acc:3'] = np.where((df_inputs_prepr['open_acc_t'].isin(range(6,10))),1,0)
            df_inputs_prepr['open_acc:4'] = np.where((df_inputs_prepr['open_acc_t'].isin(range(10,15))),1,0)
            df_inputs_prepr['open_acc:5'] = np.where((df_inputs_prepr['open_acc_t'].isin(range(15,20))),1,0)
            df_inputs_prepr['open_acc:6'] = np.where((df_inputs_prepr['open_acc_t'].isin(range(20,30))),1,0)
            df_inputs_prepr['open_acc:9'] = np.where((df_inputs_prepr['open_acc_t'].isin(range(30,int(df_inputs_prepr['open_acc'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['open_acc','open_acc_t'], axis=1)
        else:
            pass

        if 'zip_code_int' in X_train:
            df_inputs_prepr['zip_code_f'] = pd.cut(df_inputs_prepr['zip_code_int'],5)
            df_inputs_prepr['zip_code:1'] = np.where((df_inputs_prepr['zip_code_f'].isin(range(200))),1,0)
            df_inputs_prepr['zip_code:2'] = np.where((df_inputs_prepr['zip_code_f'].isin(range(200,400))),1,0)
            df_inputs_prepr['zip_code:3'] = np.where((df_inputs_prepr['zip_code_f'].isin(range(400,600))),1,0)
            df_inputs_prepr['zip_code:4'] = np.where((df_inputs_prepr['zip_code_f'].isin(range(600,800))),1,0)
            df_inputs_prepr['zip_code:5'] = np.where((df_inputs_prepr['zip_code_f'].isin(range(800,int(df_inputs_prepr['zip_code_int'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['zip_code_int','zip_code_f'], axis=1)
        else:
            pass
        if 'tot_cur_bal' in X_train:
            df_inputs_prepr['tot_cur_bal_t'] = pd.cut(df_inputs_prepr['tot_cur_bal'],300)
            df_inputs_prepr['tot_cur_bal:1'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(30000))),1,0)
            df_inputs_prepr['tot_cur_bal:2'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(30000,70000))),1,0)
            df_inputs_prepr['tot_cur_bal:3'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(70000,120000))),1,0)
            df_inputs_prepr['tot_cur_bal:4'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(120000,180000))),1,0)
            df_inputs_prepr['tot_cur_bal:5'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(180000,250000))),1,0)
            df_inputs_prepr['tot_cur_bal:6'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(250000,350000))),1,0)
            df_inputs_prepr['tot_cur_bal:7'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(350000,500000))),1,0)
            df_inputs_prepr['tot_cur_bal:8'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(500000,750000))),1,0)
            df_inputs_prepr['tot_cur_bal:9'] = np.where((df_inputs_prepr['tot_cur_bal_t'].isin(range(750000,int(df_inputs_prepr['tot_cur_bal'].max())))),1,0)
            df_inputs_prepr = df_inputs_prepr.drop(['tot_cur_bal','tot_cur_bal_t'], axis=1)
        else:
            pass

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
            df_inputs_prepr = df_inputs_prepr.drop(['addr_state:GA','addr_state:WI','addr_state:IL','addr_state:CT','addr_state:RI','addr_state:MT','addr_state:WY','addr_state:KS','addr_state:WA','addr_state:ND','addr_state:CO','addr_state:SC','addr_state:OR','addr_state:DC'], axis=1)
            # these are the individual states ['IA','AL','MA','NH','WV','ID','VT','ME']
        else:
            pass

        if 'mths_since_issue_d' in X_train:
            df_inputs_prepr['mths_since_issue_d_factor:'+'<'+'38'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38)),1,0)
            df_inputs_prepr['mths_since_issue_d_factor:'+'38-59'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38,59)),1,0)
            df_inputs_prepr['mths_since_issue_d_factor:'+'59-63'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(59,63)),1,0)
            df_inputs_prepr['mths_since_issue_d_factor:'+'63-67'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(63,67)),1,0)
            df_inputs_prepr['mths_since_issue_d_factor:'+'67-70'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(67,70)),1,0)
            df_inputs_prepr['mths_since_issue_d_factor:'+'70-85'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(70,85)),1,0)
            df_inputs_prepr['mths_since_issue_d_factor:'+'85-95'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85,95)),1,0)
            df_inputs_prepr['mths_since_issue_d_factor:'+'95+'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(95,int(df_inputs_prepr['mths_since_issue_d'].max()))),1,0)
        else:
            pass 


        '''
        NOW DO THE SAME FOR THE TEST SET
        '''

        df_inputs_prepr_t = X_test
        if 'loan_amnt' in X_test:
            df_inputs_prepr_t['loan_amnt_factor'] = pd.cut(df_inputs_prepr_t['loan_amnt'],30)
            df_inputs_prepr_t['loan_amnt_factor:1'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(1817))),1,0)
            df_inputs_prepr_t['loan_amnt_factor:2'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(1817,7084))),1,0)
            df_inputs_prepr_t['loan_amnt_factor:3'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(7084,11034))),1,0)
            df_inputs_prepr_t['loan_amnt_factor:4'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(11034,28150))),1,0)
            df_inputs_prepr_t['loan_amnt_factor:5'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(28150,28160))),1,0)
            df_inputs_prepr_t['loan_amnt_factor:6'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(28160,32100))),1,0)
            df_inputs_prepr_t['loan_amnt_factor:7'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(32100,36050))),1,0)
            df_inputs_prepr_t['loan_amnt_factor:8'] = np.where((df_inputs_prepr_t['loan_amnt_factor'].isin(range(36050,int(df_inputs_prepr_t['loan_amnt'].max())))),1,0)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['loan_amnt', 'loan_amnt_factor'], axis=1)
        else:
            pass

        if 'annual_inc' in X_test:
            df_inputs_prepr_t['annual_inc_factor'] = pd.cut(df_inputs_prepr_t['annual_inc'],100)
            df_inputs_prepr_t['annual_inc_factor:1'] = np.where((df_inputs_prepr_t['annual_inc_factor'].isin(range(7000))),1,0)
            df_inputs_prepr_t['annual_inc_factor:2'] = np.where((df_inputs_prepr_t['annual_inc_factor'].isin(range(7000,42000))),1,0)
            df_inputs_prepr_t['annual_inc_factor:3'] = np.where((df_inputs_prepr_t['annual_inc_factor'].isin(range(42000,70000))),1,0)
            df_inputs_prepr_t['annual_inc_factor:4'] = np.where((df_inputs_prepr_t['annual_inc_factor'].isin(range(70000,100000))),1,0)
            df_inputs_prepr_t['annual_inc_factor:5'] = np.where((df_inputs_prepr_t['annual_inc_factor'].isin(range(100000,112000))),1,0)
            df_inputs_prepr_t['annual_inc_factor:6'] = np.where((df_inputs_prepr_t['annual_inc_factor'].isin(range(112000,126000))),1,0)
            df_inputs_prepr_t['annual_inc_factor:7'] = np.where((df_inputs_prepr_t['annual_inc_factor'].isin(range(126000,int(df_inputs_prepr_t['annual_inc'].max())))),1,0)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['annual_inc','annual_inc_factor'], axis=1)
        else:
            pass

        if 'int_rate_int' in X_test:
            df_inputs_prepr_t['int_rate_factor_f'] = pd.cut(df_inputs_prepr_t['int_rate_int'],10)
            df_inputs_prepr_t['int_rate_factor:1'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(8))),1,0)
            df_inputs_prepr_t['int_rate_factor:2'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(8,11))),1,0)
            df_inputs_prepr_t['int_rate_factor:3'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(11,13))),1,0)
            df_inputs_prepr_t['int_rate_factor:4'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(13,15))),1,0)
            df_inputs_prepr_t['int_rate_factor:5'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(15,17))),1,0)
            df_inputs_prepr_t['int_rate_factor:6'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(17,19))),1,0)
            df_inputs_prepr_t['int_rate_factor:7'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(19,21))),1,0)
            df_inputs_prepr_t['int_rate_factor:8'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(21,23))),1,0)
            df_inputs_prepr_t['int_rate_factor:9'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(23,25))),1,0)
            df_inputs_prepr_t['int_rate_factor:10'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(25,27))),1,0)
            df_inputs_prepr_t['int_rate_factor:11'] = np.where((df_inputs_prepr_t['int_rate_factor_f'].isin(range(27,int(df_inputs_prepr_t['int_rate_int'].max())))),1,0)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['int_rate_int','int_rate_factor_f'], axis=1)
        else:
            pass 

        if 'revol_util_int' in X_test:
            df_inputs_prepr_t['revol_util_factor_f'] = pd.cut(df_inputs_prepr_t['revol_util_int'],30)
            df_inputs_prepr_t['revol_util_factor:1'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(12))),1,0)
            df_inputs_prepr_t['revol_util_factor:2'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(12,24))),1,0)
            df_inputs_prepr_t['revol_util_factor:3'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(24,36))),1,0)
            df_inputs_prepr_t['revol_util_factor:4'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(36,48))),1,0)
            df_inputs_prepr_t['revol_util_factor:5'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(48,60))),1,0)
            df_inputs_prepr_t['revol_util_factor:6'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(60,72))),1,0)
            df_inputs_prepr_t['revol_util_factor:7'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(72,84))),1,0)
            df_inputs_prepr_t['revol_util_factor:8'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(84,96))),1,0)
            df_inputs_prepr_t['revol_util_factor:9'] = np.where((df_inputs_prepr_t['revol_util_factor_f'].isin(range(96,int(df_inputs_prepr_t['revol_util_int'].max())))),1,0)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['revol_util_int','revol_util_factor_f'], axis=1)
        else:
            pass 

        if 'open_acc' in X_test:
            df_inputs_prepr_t['open_acc_t'] = pd.cut(df_inputs_prepr_t['open_acc'],30)
            df_inputs_prepr_t['open_acc:1'] = np.where((df_inputs_prepr_t['open_acc_t'].isin(range(3))),1,0)
            df_inputs_prepr_t['open_acc:2'] = np.where((df_inputs_prepr_t['open_acc_t'].isin(range(3,6))),1,0)
            df_inputs_prepr_t['open_acc:3'] = np.where((df_inputs_prepr_t['open_acc_t'].isin(range(6,10))),1,0)
            df_inputs_prepr_t['open_acc:4'] = np.where((df_inputs_prepr_t['open_acc_t'].isin(range(10,15))),1,0)
            df_inputs_prepr_t['open_acc:5'] = np.where((df_inputs_prepr_t['open_acc_t'].isin(range(15,20))),1,0)
            df_inputs_prepr_t['open_acc:6'] = np.where((df_inputs_prepr_t['open_acc_t'].isin(range(20,30))),1,0)
            df_inputs_prepr_t['open_acc:9'] = np.where((df_inputs_prepr_t['open_acc_t'].isin(range(30,int(df_inputs_prepr_t['open_acc'].max())))),1,0)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['open_acc','open_acc_t'], axis=1)
        else:
            pass 

        if 'zip_code_int' in X_test:
            df_inputs_prepr_t['zip_code_f'] = pd.cut(df_inputs_prepr_t['zip_code_int'],5)
            df_inputs_prepr_t['zip_code:1'] = np.where((df_inputs_prepr_t['zip_code_f'].isin(range(200))),1,0)
            df_inputs_prepr_t['zip_code:2'] = np.where((df_inputs_prepr_t['zip_code_f'].isin(range(200,400))),1,0)
            df_inputs_prepr_t['zip_code:3'] = np.where((df_inputs_prepr_t['zip_code_f'].isin(range(400,600))),1,0)
            df_inputs_prepr_t['zip_code:4'] = np.where((df_inputs_prepr_t['zip_code_f'].isin(range(600,800))),1,0)
            df_inputs_prepr_t['zip_code:5'] = np.where((df_inputs_prepr_t['zip_code_f'].isin(range(800,int(df_inputs_prepr_t['zip_code_int'].max())))),1,0)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['zip_code_int','zip_code_f'], axis=1)
        else:
            pass 

        if 'tot_cur_bal' in X_test:
            df_inputs_prepr_t['tot_cur_bal_t'] = pd.cut(df_inputs_prepr_t['tot_cur_bal'],300)
            df_inputs_prepr_t['tot_cur_bal:1'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(30000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:2'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(30000,70000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:3'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(70000,120000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:4'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(120000,180000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:5'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(180000,250000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:6'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(250000,350000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:7'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(350000,500000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:8'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(500000,750000))),1,0)
            df_inputs_prepr_t['tot_cur_bal:9'] = np.where((df_inputs_prepr_t['tot_cur_bal_t'].isin(range(750000,int(df_inputs_prepr_t['tot_cur_bal'].max())))),1,0)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['tot_cur_bal','tot_cur_bal_t'], axis=1)
        else:
            pass  

        if 'addr_state:OK' in X_test:
            df_inputs_prepr_t['addr_state:OK_AR_LA_MS'] = sum([df_inputs_prepr_t['addr_state:OK'],df_inputs_prepr_t['addr_state:AR'],df_inputs_prepr_t['addr_state:LA'],df_inputs_prepr_t['addr_state:MS']])
            df_inputs_prepr_t['addr_state:NV_NY'] = sum([df_inputs_prepr_t['addr_state:NV'],df_inputs_prepr_t['addr_state:NY']])
            df_inputs_prepr_t['addr_state:HI_FL_NM'] = sum([df_inputs_prepr_t['addr_state:HI'],df_inputs_prepr_t['addr_state:FL'],df_inputs_prepr_t['addr_state:NM']])
            df_inputs_prepr_t['addr_state:MD_MO_PA_NC_NJ_IN'] = sum([df_inputs_prepr_t['addr_state:MD'],df_inputs_prepr_t['addr_state:MO'],df_inputs_prepr_t['addr_state:PA'],df_inputs_prepr_t['addr_state:NC'],df_inputs_prepr_t['addr_state:NJ'],df_inputs_prepr_t['addr_state:IN']])
            df_inputs_prepr_t['addr_state:KY_CA'] = sum([df_inputs_prepr_t['addr_state:KY'],df_inputs_prepr_t['addr_state:CA']])
            df_inputs_prepr_t['addr_state:SD_NE_TN_MI_DE_VA'] = sum([df_inputs_prepr_t['addr_state:SD'],df_inputs_prepr_t['addr_state:NE'],df_inputs_prepr_t['addr_state:TN'],df_inputs_prepr_t['addr_state:MI'],df_inputs_prepr_t['addr_state:DE'],df_inputs_prepr_t['addr_state:VA']])
            df_inputs_prepr_t['addr_state:MN_AZ_TX_OH'] = sum([df_inputs_prepr_t['addr_state:MN'],df_inputs_prepr_t['addr_state:AZ'],df_inputs_prepr_t['addr_state:TX'],df_inputs_prepr_t['addr_state:OH']])
            df_inputs_prepr_t['addr_state:UT_GA_WI'] = sum([df_inputs_prepr_t['addr_state:UT'],df_inputs_prepr_t['addr_state:GA'],df_inputs_prepr_t['addr_state:WI']])
            df_inputs_prepr_t['addr_state:IL_CT_RI_MT'] = sum([df_inputs_prepr_t['addr_state:IL'],df_inputs_prepr_t['addr_state:CT'],df_inputs_prepr_t['addr_state:RI'],df_inputs_prepr_t['addr_state:MT']])
            df_inputs_prepr_t['addr_state:WY_KS_WA'] = sum([df_inputs_prepr_t['addr_state:WY'],df_inputs_prepr_t['addr_state:KS'],df_inputs_prepr_t['addr_state:WA']])
            df_inputs_prepr_t['addr_state:ND_CO'] = sum([df_inputs_prepr_t['addr_state:ND'],df_inputs_prepr_t['addr_state:CO']])
            df_inputs_prepr_t['addr_state:SC_OR_DC'] = sum([df_inputs_prepr_t['addr_state:SC'],df_inputs_prepr_t['addr_state:OR'],df_inputs_prepr_t['addr_state:DC']])
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['addr_state:OK','addr_state:AR','addr_state:LA','addr_state:MS','addr_state:NV','addr_state:NY','addr_state:HI','addr_state:FL','addr_state:NM','addr_state:MD','addr_state:MO','addr_state:PA','addr_state:NC'], axis=1)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['addr_state:IN','addr_state:NJ','addr_state:KY','addr_state:CA','addr_state:SD','addr_state:NE','addr_state:TN','addr_state:MI','addr_state:DE','addr_state:VA','addr_state:MN','addr_state:AZ','addr_state:TX','addr_state:OH','addr_state:UT'], axis=1)
            df_inputs_prepr_t = df_inputs_prepr_t.drop(['addr_state:GA','addr_state:WI','addr_state:IL','addr_state:CT','addr_state:RI','addr_state:MT','addr_state:WY','addr_state:KS','addr_state:WA','addr_state:ND','addr_state:CO','addr_state:SC','addr_state:OR','addr_state:DC'], axis=1)
            # these are the individual states ['IA','AL','MA','NH','WV','ID','VT','ME']
        else:
            pass

        if 'mths_since_issue_d' in X_train:
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'<'+'38'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(38)),1,0)
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'38-59'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(38,59)),1,0)
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'59-63'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(59,63)),1,0)
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'63-67'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(63,67)),1,0)
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'67-70'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(67,70)),1,0)
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'70-85'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(70,85)),1,0)
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'85-95'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(85,95)),1,0)
            df_inputs_prepr_t['mths_since_issue_d_factor:'+'95+'] = np.where(df_inputs_prepr_t['mths_since_issue_d'].isin(range(95,int(df_inputs_prepr_t['mths_since_issue_d'].max()))),1,0)
        else:
            pass 
        return df_inputs_prepr, df_inputs_prepr_t

            # have addr_state and mths_since_issue_d_factor to add

if __name__ == '__main__':
    X_train_ohe = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train_ohe.csv')
    X_test_ohe = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test_ohe.csv')
    # y_train = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_train.csv')
    # y_test = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_test.csv')

    woe = WOE()

    X_train_woe, X_test_woe = woe.action_woe(X_train_ohe, X_test_ohe)

    # print(X_train_woe.shape)
    # print(X_test_woe.shape)
    X_train_woe.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train_woe.csv')
    X_test_woe.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test_woe.csv')
    print('MISSION ACCOMPLISHED!!!')
