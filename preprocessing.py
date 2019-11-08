import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts


class Preprocessing:
    def __init__(self):
        '''
        Initialize the important lists used for the subsequent functions
        '''
        self.convert_to_numeric = ['funded_amnt','funded_amnt_inv','installment','dti','delinq_2yrs','fico_range_low','inq_last_6mths','pub_rec','revol_bal','total_acc','acc_now_delinq','chargeoff_within_12_mths','tax_liens','loan_amnt','annual_inc','open_acc','pub_rec_bankruptcies']
        self.initial_list = ['funded_amnt','funded_amnt_inv','installment','dti','delinq_2yrs','fico_range_low','inq_last_6mths','pub_rec','revol_bal','total_acc','acc_now_delinq','chargeoff_within_12_mths','tax_liens','emp_length','term','grade','sub_grade','home_ownership','verification_status','purpose','initial_list_status','addr_state','loan_amnt','annual_inc','int_rate','revol_util','open_acc','zip_code','loan_status','pub_rec_bankruptcies']

    
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
        This function actual does the preprocess editing to the dataframe
        OUTPUT: dataframe
        '''
        dataframe = dataframe[self.initial_list]
        dataframe = dataframe[dataframe.columns[0]!=self.initial_list[0]]

        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)

        # replace the na values of annual_inc and revol_util_int with the mean amount
        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
        dataframe['revol_util_int'].fillna(dataframe['revol_util_int'].mean(), inplace=True)
    
        '''
        create a good_bad column which will have a 0 if the loan is bad and 1 if it is good
        this is going to be the target or y for my model
        '''
        dataframe = dataframe[dataframe['loan_status']!='Current']
        dataframe['good_bad'] = np.where(dataframe['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','Late (16-30 days)']),0,1)
        
        '''
        this creates a list of column names where the percent of missing values is greater than the threshold which is set to 95% as default; this is just to ensure I do not have additional fields that need to be pruned
        '''
        df_drop_val = [x for x in dataframe.columns if (100* dataframe[x].isnull().sum() / len(dataframe))>tr]
        dataframe.drop(df_drop_val, axis=1, inplace=True)
        
        for items in dataframe.columns:
            dataframe = dataframe[dataframe[items].notnull()]
        
        return dataframe

    def action_current(self, dataframe, tr=0.95):
        '''
        This function actual does the preprocess editing to the dataframe
        OUTPUT: dataframe
        '''
        dataframe = dataframe[self.initial_list]
        dataframe = dataframe[dataframe.columns[0]!=self.initial_list[0]]

        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)
        
        # replace the na values of annual_inc and revol_util_int with the mean amount
        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
        dataframe['revol_util_int'].fillna(dataframe['revol_util_int'].mean(), inplace=True)
    
        '''
        this is my current list of loans - here is where I can apply the model to get back a list of loans to invest in
        '''
        dataframe = dataframe[dataframe['loan_status']=='Current']
        
        '''
        this creates a list of column names where the percent of missing values is greater than the threshold which is set to 95% as default; this is just to ensure I do not have additional fields that need to be pruned
        '''
        
        df_drop_val = [x for x in dataframe.columns if (100* dataframe[x].isnull().sum() / len(dataframe))>tr]
        dataframe.drop(df_drop_val, axis=1, inplace=True)

        for items in dataframe.columns:
            dataframe = dataframe[dataframe[items].notnull()]
        
        return dataframe


    def action_LGD(self, dataframe, tr=0.95):
        '''
        This function actual does the preprocess editing to the dataframe
        OUTPUT: dataframe
        '''
        dataframe = dataframe[self.initial_list]
        dataframe = dataframe[dataframe.columns[0]!=self.initial_list[0]]

        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)
        
        # replace the na values of annual_inc and revol_util_int with the mean amount
        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
        dataframe['revol_util_int'].fillna(dataframe['revol_util_int'].mean(), inplace=True)
    
        '''
        create a good_bad column which will have a 0 if the loan is bad and 1 if it is good
        this is going to be the target or y for my model
        '''
        #dataframe = dataframe[dataframe['loan_status']!='Current']
        #dataframe['good_bad'] = np.where(dataframe['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','Late (16-30 days)']),0,1)
        
        '''
        this creates a list of column names where the percent of missing values is greater than the threshold which is set to 95% as default; this is just to ensure I do not have additional fields that need to be pruned
        '''
        df_drop_val = [x for x in dataframe.columns if (100* dataframe[x].isnull().sum() / len(dataframe))>tr]
        dataframe.drop(df_drop_val, axis=1, inplace=True)
        
        for items in dataframe.columns:
            dataframe = dataframe[dataframe[items].notnull()]
        
        return dataframe

if __name__ == '__main__':
    
    #df_backup = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/merged.csv', skiprows=1, low_memory=False)
    df_backup = pd.read_csv('/home/ubuntu/merged.csv', skiprows=1, low_memory=False)
    df = df_backup.copy()

    prep = Preprocessing()
    df = prep.action(df)
    
    current = prep.action_current(df_backup)
    
    df_preprocessed = df

    #TODO: send to postgres

    X_train_tt, X_test_tt, y_train_tt, y_test_tt = tts(df.drop(['loan_status','good_bad'], axis=1), df['good_bad'], test_size=0.25, random_state=42)
    
    X_train_tt.to_csv('/home/ubuntu/X_train_tt.csv')
    y_train_tt.to_csv('/home/ubuntu/y_train_tt.csv')
    X_test_tt.to_csv('/home/ubuntu/X_test_tt.csv')
    y_test_tt.to_csv('/home/ubuntu/y_test_tt.csv')
    df_preprocessed.to_csv('/home/ubuntu/df_preprocessed.csv')
    current.to_csv('/home/ubuntu/current.csv')

    print('Mission Accomplished!!')
    