import pandas as pd
import numpy as np

class OHE:
    def __init__(self):
        '''
        initialize the list to be used for one hot encoding
        '''
        self.ohe_list = ['grade','sub_grade','home_ownership','verification_status','purpose','initial_list_status','addr_state']
        
    
    def loan_data_d(self, dataframe, dlist):    
        '''
        Convert the specified items into one hot encoded items and add to original dataframe
        OUTPUT: dataframe

        '''
        for items in dlist:
            loan_data_dummies = [pd.get_dummies(dataframe[items], drop_first=True, prefix=items,prefix_sep=':')]
            loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
            dataframe = pd.concat([dataframe, loan_data_dummies], axis = 1)
        return dataframe

    def action(self, train, test):
        '''
        Performs the one hot encoding for the dataset based on a list of inputs: 1) train - training dataframe; 2) test - test dataframe, 3) other - input list of fields to one hot encode
        OUTPUT: Modified Train dataframe, Modified Test dataframe
        '''
        dummies_other_train = self.loan_data_d(train, self.ohe_list)
        dummies_other_test = self.loan_data_d(test, self.ohe_list)

        dummies_other_train.drop(self.ohe_list, axis=1, inplace=True)
        dummies_other_test.drop(self.ohe_list, axis=1, inplace=True)

        return dummies_other_train, dummies_other_test

if __name__ == '__main__':
    
    X_train = pd.read_csv('/home/ubuntu/X_train_tt.csv')
    X_test = pd.read_csv('/home/ubuntu/X_test_tt.csv')
    
    X_train.drop(['Unnamed: 0'], axis=1, inplace=True)
    X_test.drop(['Unnamed: 0'], axis=1, inplace=True)

    ohe = OHE()

    X_train_ohe_tt, X_test_ohe_tt = ohe.action(X_train, X_test)
    #X_train_ohe_tt.drop(['Unnamed: 0'], axis=1, inplace=True)
    #X_test_ohe_tt.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    X_train_ohe_tt.to_csv('/home/ubuntu/X_train_ohe_tt.csv')
    X_test_ohe_tt.to_csv('/home/ubuntu/X_test_ohe_tt.csv')
    
    print('Mission Accomplished')
