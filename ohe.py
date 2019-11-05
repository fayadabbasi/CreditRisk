import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ohe:
    def __init__(self):
        pass
    

    # X_train.drop('addr_state', axis=1, inplace=True)
    # X_test.drop('addr_state', axis=1, inplace=True)

    # ohe_list = ['grade','sub_grade','home_ownership','verification_status','pymnt_plan','purpose','initial_list_status','application_type','hardship_flag','debt_settlement_flag']

    def loan_data_d(self, dataframe, dlist):    
        for items in dlist:
            loan_data_dummies = [pd.get_dummies(dataframe[items], drop_first=True, prefix=items,prefix_sep=':')]
            loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
            dataframe = pd.concat([dataframe, loan_data_dummies], axis = 1)
        return dataframe

    # dummies_list_other = ['grade','home_ownership','pymnt_plan','purpose','sub_grade','verification_status','initial_list_status','application_type','hardship_flag','debt_settlement_flag']

    def action(self, train, test, other):
        dummies_other_train = self.loan_data_d(train, other)
        dummies_other_test = self.loan_data_d(test, other)

        dummies_other_train.drop(dummies_list_other, axis=1, inplace=True)
        dummies_other_test.drop(dummies_list_other, axis=1, inplace=True)

if __name__ == '__main__':
    X_train = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train.csv')
    X_test = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test.csv')
    y_train = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_train.csv', header=None)
    y_test = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/y_test.csv', header=None)
    
    X_train.drop('addr_state', axis=1, inplace=True)
    X_test.drop('addr_state', axis=1, inplace=True)
    
    ohe_list = ['grade','sub_grade','home_ownership','verification_status','pymnt_plan','purpose','initial_list_status','application_type','hardship_flag','debt_settlement_flag']



    dummies_list_other = ['grade','home_ownership','pymnt_plan','purpose','sub_grade','verification_status','initial_list_status','application_type','hardship_flag','debt_settlement_flag']
