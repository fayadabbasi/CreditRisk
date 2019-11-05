import pandas as pd
import numpy as np

class OHE:
    def __init__(self):
        pass
    
    def loan_data_d(self, dataframe, dlist):    
        for items in dlist:
            loan_data_dummies = [pd.get_dummies(dataframe[items], drop_first=True, prefix=items,prefix_sep=':')]
            loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
            dataframe = pd.concat([dataframe, loan_data_dummies], axis = 1)
        return dataframe

    def action(self, train, test, other):
        dummies_other_train = self.loan_data_d(train, other)
        dummies_other_test = self.loan_data_d(test, other)

        dummies_other_train.drop(other, axis=1, inplace=True)
        dummies_other_test.drop(other, axis=1, inplace=True)

        return dummies_other_train, dummies_other_test

if __name__ == '__main__':
    # X_train = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train.csv')
    # X_test = pd.read_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test.csv')
    
    # ohe_list = ['grade','sub_grade','home_ownership','verification_status','pymnt_plan','purpose','initial_list_status','application_type','hardship_flag','debt_settlement_flag']

    # ohe = OHE()

    # X_train_ohe, X_test_ohe = ohe.action(X_train, X_test, ohe_list)

    # X_train_ohe.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_train_ohe.csv')
    # X_test_ohe.to_csv('/Users/fayadabbasi/Desktop/Python_Scripts/Galvanize/DSI/CreditRisk/X_test_ohe.csv')
