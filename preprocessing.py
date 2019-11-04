import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

class Preprocessing:
    def __init__(self):
        '''
        Initialize the important lists used for the subsequent functions -- this actually should be done outside of this model so I can refine the lists
        '''
        self.convert_to_numeric = ['loan_amnt','annual_inc','open_acc','tot_cur_bal','tot_coll_amt']
        self.initial_list = ['loan_amnt','grade','emp_length','annual_inc','purpose','revol_util','home_ownership','term','int_rate','loan_status','open_acc','zip_code','tot_cur_bal','tot_coll_amt']
        self.missing_value_list = ['tot_cur_bal']
    
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
        
    def action(self, dataframe):
        for items in self.convert_to_numeric:
            dataframe[items] = pd.to_numeric(dataframe[items])
        
        self.emp_length(dataframe)
        self.term_length(dataframe)
        self.perc_convert(dataframe, column='int_rate')
        self.perc_convert(dataframe, column='revol_util')
        self.zip_convert(dataframe)

        '''
        replace the na values of annual_inc with the mean annual_inc amount
        '''

        dataframe['annual_inc'].fillna(dataframe['annual_inc'].mean(), inplace=True)
    
        '''
        create a good_bad column which will have a 0 if the loan is bad and 1 if it is good
        this is going to be the target or y for my model
        ********* THIS IS SOMETHING I SHOULD REVISE - NO CURRENT IN THIS LIST AND SEE IF 16-30 DAYS IMPACTS RESULTS **********
        '''
        dataframe['good_bad'] = np.where(dataframe['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','Late (16-30 days)']),0,1)
        
     
    def missing_values(self,dataframe,column):
        '''
        drops rows where there are missing values for these columns 
        '''        
        for items in column:
            dataframe = dataframe[dataframe[items].notnull()]
        return dataframe


class OneHotEncoding:
    def __init__(self):
        self.dummies_list = ['grade','home_ownership','purpose','emp_length_int','term_int']
        self.discrete_variable_name = discrete_variable_name
        self.good_bad_variable_df = good_bad_variable_df

    def loan_data_d(self, dataframe):
        for items in self.dummies_list:
            loan_data_dummies = [pd.get_dummies(dataframe[items], prefix=items,prefix_sep=':')]
            loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
            dataframe = pd.concat([dataframe, loan_data_dummies], axis = 1)
        return dataframe


    def woe_ordered_continuous(self, df, discrete_variable_name, good_bad_variable_df):
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

# class OneHotContinuous:
#     def __init__(self, discrete_variable_name, good_bad_variable_df):
#         self.discrete_variable_name = discrete_variable_name
#         self.good_bad_variable_df = good_bad_variable_df


#     def woe_ordered_continuous(self, df, discrete_variable_name, good_bad_variable_df):
#         df = pd.concat([df[discrete_variable_name], good_bad_variable_df],axis=1)
#         df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
#         df = df.iloc[:,[0,1,3]]
#         df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
#         df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
#         df['n_good'] = df['prop_good'] * df['n_obs']
#         df['n_bad'] = (1-df['prop_good']) * df['n_obs']
#         df['prop_n_good'] = df['n_good']/ df['n_good'].sum()
#         df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
#         df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
#         # df = df.sort_values(['WoE'])
#         # df = df.reset_index(drop=True)
#         df['diff_prop_good'] = df['prop_good'].diff().abs()
#         df['diff_WoE'] = df['WoE'].diff().abs()
#         df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
#         df['IV'] = df['IV'].sum()
#         return df


if __name__ == '__main__':
    '''df_backup = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/merged.csv', skiprows=1, low_memory=False)
    df = df_backup.copy()
    df = df[df.id!='id']'''
    
    #df = df.drop(['emp_length','term','int_rate','int_rate_t','revol_util','revol_util_t', 'zip_code'], axis=1)
    #df = missing_values(df,['loan_amnt','tot_cur_bal','tot_coll_amt'])
    # prep = Preprocessing()
    # prep.action(df)
    # df = prep.missing_values(df, ['tot_cur_bal'])
    #df = loan_data_d(df,lst)
    #df = df.drop(lst, axis=1)
    '''
    I should have an output that pickles the train test dataframes so I can access later
    But make sure the pickled dataframes do not have the one hot encoding
    '''
    #print(miss_values)
    #print(df.head(2))
    # X_train, X_test, y_train, y_test = tts(df.drop(['loan_status','good_bad'], axis=1), df['good_bad'], test_size=0.25, random_state=42)
