import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

class Preprocessing:
    def __init__(self):
        '''
        Initialize the important lists used for the subsequent functions -- this actually should be done outside of this model so I can refine the lists
        '''
        self.convert_to_numeric = ['loan_amnt','annual_inc','open_acc','tot_cur_bal','tot_coll_amt', 'funded_amnt','funded_amnt_inv','installment',]
        self.initial_list = ['loan_amnt','grade','emp_length','annual_inc','purpose','revol_util','home_ownership','term','int_rate','loan_status','open_acc','zip_code','tot_cur_bal','tot_coll_amt']
        self.missing_value_list = ['tot_cur_bal']
        self.misc_drop_list = ['issue_d','url','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
    
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
        
        dataframe = dataframe[dataframe.id!='id']

        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)
        # replace the na values of annual_inc with the mean annual_inc amount
        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
    
        '''
        create a good_bad column which will have a 0 if the loan is bad and 1 if it is good
        this is going to be the target or y for my model
        ********* THIS IS SOMETHING I COULD REVISE - SEE IF 16-30 DAYS IMPACTS RESULTS **********
        '''
        dataframe = dataframe[dataframe['loan_status']!='Current']
        dataframe['good_bad'] = np.where(dataframe['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','Late (16-30 days)']),0,1)
        
        '''
        this creates a list of column names where the percent of missing values is greater than the threshold which is set to 95% as default
        '''
        df_drop_val = [x for x in dataframe.columns if (100* dataframe[x].isnull().sum() / len(dataframe))>tr]
        dataframe.drop(df_drop_val, axis=1, inplace=True)
        dataframe.drop(self.misc_drop_list, axis=1, inplace=True)
        '''
        this drops rows that have no values - tested to be about 0.2% of all rows or about 4700
        '''
        for items in dataframe.columns:
            dataframe = dataframe[dataframe[items].notnull()]
        
        return dataframe

    def action_current(self, dataframe, tr=0.95):
        '''
        This function actual does the preprocess editing to the dataframe
        OUTPUT: dataframe
        '''
        
        dataframe = dataframe[dataframe.id!='id']

        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)
        # replace the na values of annual_inc with the mean annual_inc amount
        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
    
        '''
        create a good_bad column which will have a 0 if the loan is bad and 1 if it is good
        this is going to be the target or y for my model
        ********* THIS IS SOMETHING I COULD REVISE - SEE IF 16-30 DAYS IMPACTS RESULTS **********
        '''
        dataframe = dataframe[dataframe['loan_status']=='Current']
        
        '''
        this creates a list of column names where the percent of missing values is greater than the threshold which is set to 95% as default
        '''
        df_drop_val = [x for x in dataframe.columns if (100* dataframe[x].isnull().sum() / len(dataframe))>tr]
        dataframe.drop(df_drop_val, axis=1, inplace=True)
        dataframe.drop(self.misc_drop_list, axis=1, inplace=True)
        '''
        this drops rows that have no values - tested to be about 0.2% of all rows or about 4300
        '''
        for items in dataframe.columns:
            dataframe = dataframe[dataframe[items].notnull()]
        
        return dataframe


    def action_LGD(self, dataframe, tr=0.95):
        '''
        This function actual does the preprocess editing to the dataframe
        OUTPUT: dataframe
        '''
        
        dataframe = dataframe[dataframe.id!='id']

        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)
        # replace the na values of annual_inc with the mean annual_inc amount
        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
    
        '''
        create a good_bad column which will have a 0 if the loan is bad and 1 if it is good
        this is going to be the target or y for my model
        ********* THIS IS SOMETHING I COULD REVISE - SEE IF 16-30 DAYS IMPACTS RESULTS **********
        '''
        #dataframe = dataframe[dataframe['loan_status']!='Current']
        #dataframe['good_bad'] = np.where(dataframe['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','Late (16-30 days)']),0,1)
        
        '''
        this creates a list of column names where the percent of missing values is greater than the threshold which is set to 95% as default
        '''
        df_drop_val = [x for x in dataframe.columns if (100* dataframe[x].isnull().sum() / len(dataframe))>tr]
        dataframe.drop(df_drop_val, axis=1, inplace=True)
        dataframe.drop(self.misc_drop_list, axis=1, inplace=True)
        '''
        this drops rows that have no values - tested to be about 0.2% of all rows or about 4700
        '''
        for items in dataframe.columns:
            dataframe = dataframe[dataframe[items].notnull()]
        
        return dataframe

if __name__ == '__main__':
    
    df_backup = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/merged.csv', skiprows=1, low_memory=False)
    df = df_backup.copy()
    df = df.iloc[:100000,:]

    prep = Preprocessing()
    df = prep.action(df)
    #current = prep.action_current(df_backup)
    
    #TODO: send to postgres

    X_train_tt, X_test_tt, y_train_tt, y_test_tt = tts(df.drop(['loan_status','good_bad'], axis=1), df['good_bad'], test_size=0.25, random_state=42)
    X_train_tt.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train_tt.csv')
    y_train_tt.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_train_tt.csv')
    X_test_tt.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test_tt.csv')
    y_test_tt.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_test_tt.csv')
    #current.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/current.csv')
    
    print('Mission Accomplished!!')
    