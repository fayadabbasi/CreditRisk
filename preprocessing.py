import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

class Preprocessing:
    def __init__(self):
        '''
        Initialize the important lists used for the subsequent functions
        '''
        self.convert_to_numeric = ['recoveries','tot_rec_prncp','funded_amnt','funded_amnt_inv','installment','dti','delinq_2yrs','fico_range_low','inq_last_6mths','pub_rec','revol_bal','total_acc','acc_now_delinq','chargeoff_within_12_mths','tax_liens','loan_amnt','annual_inc','open_acc','pub_rec_bankruptcies','all_util','collections_12_mths_ex_med','delinq_amnt','inq_fi','mths_since_last_delinq','num_accts_ever_120_pd','num_actv_rev_tl','num_rev_accts','total_rec_late_fee']
        self.initial_list = ['recoveries','tot_rec_prncp','funded_amnt','funded_amnt_inv','installment','dti','delinq_2yrs','fico_range_low','inq_last_6mths','pub_rec','revol_bal','total_acc','acc_now_delinq','chargeoff_within_12_mths','tax_liens','emp_length','term','grade','sub_grade','home_ownership','verification_status','purpose','initial_list_status','addr_state','loan_amnt','annual_inc','int_rate','revol_util','open_acc','zip_code','loan_status','pub_rec_bankruptcies','all_util','collections_12_mths_ex_med','delinq_amnt','inq_fi','mths_since_last_delinq','num_accts_ever_120_pd','num_actv_rev_tl','num_rev_accts','total_rec_late_fee']
        
    
    def emp_length(self, dataframe,column='emp_length'):
        '''
        function to edit the emp_length so that it can be converted to a numeric value
        '''
        dataframe[column+'_int'] = dataframe[column]
        dataframe[column+'_int'].replace('< 1 year', 0, inplace=True)
        dataframe[column+'_int'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
        dataframe[column+'_int'] = pd.to_numeric(dataframe[column+'_int'])
        dataframe[column+'_int'].fillna(value=0, inplace=True)
        dataframe.drop([column], axis=1, inplace=True)

    def term_length(self, dataframe, column='term'):
        '''
        function to edit the term_length so that it can be converted to a numeric value
        '''        
        dataframe[column+'_int'] = dataframe[column]
        dataframe[column+'_int'] = dataframe[column].str.replace(' months','')
        dataframe[column+'_int'] = pd.to_numeric(dataframe[column+'_int'])
        dataframe.drop([column], axis=1, inplace=True)
    
    def perc_convert(self, dataframe, column):
        '''
        function to edit int_rate and revol_util so that they can be converted to numeric values
        '''
        dataframe[column+'_t'] = dataframe[column].fillna(str('0'))
        dataframe[column+'_int'] = dataframe[column+'_t'].map(lambda x: str(x).rstrip('%').strip())
        dataframe[column+'_int'] = pd.to_numeric(dataframe[column+'_int'])
        dataframe.drop([column, column+'_t'], axis=1, inplace=True)
    
    def zip_convert(self, dataframe, column='zip_code'):
        '''
        function to edit the zip_code so that it can be converted to a numeric value
        '''
        dataframe[column+'_int'] = dataframe[column].fillna(str('000'))
        dataframe[column+'_int'] = dataframe[column+'_int'].map(lambda x: str(x).rstrip('xx').strip())
        dataframe[column+'_int'] = pd.to_numeric(dataframe[column+'_int'])
        dataframe.drop([column], axis=1, inplace=True)
        
    def action(self, dataframe, tr=0.95):
        '''
        Preprocess editing pipeline to the dataframe
        Steps include:  1) create a dataframe with only the features I plan to use
                        2) drop rows inserted from the merged csv file
                        3) convert selected features to numeric
                        4) perform conversion of selected features - emp_length, term_length, int_rate, revol_util, zip_code
                        5) replace na values for annual_inc and revol_util with the mean
                        6) drop the loans where status is 'Current'
                        7) create a 'good_bad' column which will serve as the target or y value
                        8) drop the 51 rows where the feature objects have null values
                        9) for remaining numeric values, replace null values with '0'
        OUTPUT: dataframe, dataframe_current 
        dataframe is the dataframe to be used for modeling
        dataframe_current is the dataframe of all current loans from which the recommended list of loans will be built 
        '''
        dataframe = dataframe[self.initial_list]
        dataframe = dataframe[dataframe['funded_amnt']!='funded_amnt']

        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)

        '''
        replace the na values of annual_inc and revol_util_int with the mean amount
        '''

        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
        dataframe['revol_util_int'].fillna(dataframe['revol_util_int'].mean(), inplace=True)
        
        '''
        create a good_bad column which will have a 0 if the loan is bad and 1 if it is good
        this is going to be the target or y for my model
        '''
        
        # dataframe = dataframe[dataframe['loan_status']!='Current']
        # dataframe['good_bad'] = np.where(dataframe['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','Late (16-30 days)']),1,0)
        
        '''
        Drop all rows where the dtypes object has a null value - using 'grade' as the proxy but applies for all the dtypes object so far
        '''
        dataframe = dataframe[dataframe['grade'].notnull()]
        '''
        this creates a list of column names where the percent of missing values is greater than the threshold which is set to 95% as default; this is just to ensure I do not have additional fields that need to be pruned
        
        df_drop_val = [x for x in dataframe.columns if (100* dataframe[x].isnull().sum() / len(dataframe))>tr]
        dataframe.drop(df_drop_val, axis=1, inplace=True)
        '''
        for items in dataframe.columns:
            dataframe[items].fillna(0, inplace=True)
        
        dataframe_lgd = dataframe[['recoveries', 'funded_amnt','total_rec_prncp', 'loan_status']]
        dataframe_current = dataframe[dataframe['loan_status']=='Current']
        dataframe = dataframe[dataframe['loan_status']!='Current']
        dataframe.drop(['recoveries', 'total_rec_prncp'], axis=1, inplace=True)
        dataframe['good_bad'] = np.where(dataframe['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','Late (16-30 days)']),1,0)

        
        return dataframe, dataframe_current, dataframe_lgd

if __name__ == '__main__':
    
    df_backup = pd.read_csv('/home/ubuntu/merged.csv', skiprows=1, low_memory=False)
    df = df_backup.copy()

    prep = Preprocessing()
    df, current, lgd = prep.action(df)
    
    #df_preprocessed = df

    #TODO: send to postgres

    #X_train_tt, X_test_tt, y_train_tt, y_test_tt = tts(df.drop(['loan_status','good_bad'], axis=1), df['good_bad'], test_size=0.25, random_state=42)
    
    # df_preprocessed.to_csv('/home/ubuntu/df_preprocessed.csv')

    # X_train_tt.to_csv('/home/ubuntu/X_train_tt.csv')
    # y_train_tt.to_csv('/home/ubuntu/y_train_tt.csv')
    # X_test_tt.to_csv('/home/ubuntu/X_test_tt.csv')
    # y_test_tt.to_csv('/home/ubuntu/y_test_tt.csv')

    # current.to_csv('/home/ubuntu/current.csv')
    X_train_lgd, X_test_lgd, y_train_lgd, y_test_lgd = tts(lgd.drop(['loan_status'], axis=1), lgd['loan_status'], test_size=0.25, random_state=42)
    X_train_lgd.to_csv('/home/ubuntu/X_train_lgd.csv')
    y_train_lgd.to_csv('/home/ubuntu/y_train_lgd.csv')
    X_test_lgd.to_csv('/home/ubuntu/X_test_lgd.csv')
    y_test_lgd.to_csv('/home/ubuntu/y_test_lgd.csv')
    
    print('Mission Accomplished!!')
    