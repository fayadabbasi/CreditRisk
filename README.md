# CreditRisk

## Background

Lending Club is a peer to peer lending platform that emerged in 2007. The platform allows individuals to review loan applications from others and evaluate a variety of inputs and then based on one's conclusions, the individual can commit a designated dollar amount to the applicants. Lending Club provides an API to download information about all historical loans on the platform. 

For my project, I have looked at Lending Club data from 2007 through Q2 2019. In total, there are 2,384,781 entries and a total of 150 features. While many categories have incomplete data, most of the relevant categories appear to have pretty thorough details about the loans. 

## Objective

For my project, I am looking to select some features from the dataset and see if I can build a model to predict if a loan is considered a good loan or a bad loan. I have defined good loans as Fully Paid or Current while bad loans are defined as all other categories. 

For the dataset, good loans accounted for 86% of the total dataset while bad loans were 14%. Of the bad loans, 88% were charged off. 

I will one hot encode two different sets of data: my first set is based on 10 criteria that I feel could be good predictors and the second set is based on an additional 13 criteria. The criteria are listed below with a brief description. 

I begin by applying a logistic regression model. I will split my dataset into a train/test split (75%/25%) and then train the model on the one hot encoded data. From that, I will test against the test dataset and see what my area under curve is for my ROC curve. 

I will then do the same with a random forest, one test on a default class weighting and a second over-weighting the bad loans. I am using n_estimators at 100 and no max depth to allow the trees to grow fully. For the class weighted, I am using a 10:1 weighting for bad:good to better represent the bad loan category. 

Finally I will run a gradient boosted test. For the boosted model, I am using a learning rate of 0.001, n_estimators of 300, and max depth of 2. In the future, I can apply more tuning to this model, potentially including multiple models on an EC2 instance to better determine optimal hyper parameters. As it is, with the parameters I chose, given the large dataset, it takes roughly 20min to train the gradient boosted model. 

## Loan Characteristics

Below is a simple breakdown of the total loans in the dataset. 

|                                                     |                 |         | 
|-----------------------------------------------------|-----------------|---------| 
| Status                                              | Number of Loans | Percent | 
| Fully Paid                                          | 1,191,125       | 50%     | 
| Current                                             |   868,848       | 36%     | 
| Charged Off                                         |   287,174       | 12%     | 
| Late (31-120 days)                                  |    20,775       | 1%      | 
| In Grace Period                                     |     9,242       | 0%      | 
| Late (16-30 days)                                   |     4,826       | 0%      | 
| Does not meet the credit policy. Status:Fully Paid  |     1,988       | 0%      | 
| Does not meet the credit policy. Status:Charged Off |       761       | 0%      | 
| Default                                             |        42       | 0%      | 
| Total                                               | 2,384,781       | 100%    | 


## Critera

Loan critera used with the first ten being used on the first pass and the remaining three added on the second pass: 

|                |                                                                                                                    | 
|----------------|--------------------------------------------------------------------------------------------------------------------| 
| Category       | Description                                                                                                        | 
| loan amnt      | Listed amount of loan applied for by the borrower                                                                  | 
| grade          | Lending Club assigned loan grade                                                                                   | 
| emp length     | Employment length in years, between 0 and 10                                                                       | 
| annual inc     | Self reported annual income provided by the borrower                                                               | 
| purpose        | A category provided by the borrower for the loan request                                                           | 
| revol util     | Revolving line utilization rate or amount of credit the borrower is using vs total available revolving credit line | 
| home ownership | Home ownership status provided by the borrower                                                                     | 
| term           | Number of payments on the loan. Either 36 or 60 months                                                             | 
| int rate       | Interest rate on the loan                                                                                          | 
| loan status    | Current status of loan - this is used for the target                                                               | 
| open acc       | Number of open credit lines in the borrower's credit file                                                          | 
| zip code       | The first three numbers of the user zipcode                                                                        | 
| tot cur bal    | Total current balance of all accounts  

## Process

I first performed some exploration of the dataset. While I initially tried evaluating the entire 150 feature dataset, I narrowed the scope to the above mentioned criteria. I wrote a function to handle all the preprocessing. For null values, I evaluated what the best way to handle each would be - in the case of annual income, for instance, I imputed the mean annual income while for current balance, I imputed a zero value for missing data. 

Next, I one hot encoded all categorical values. This included: grade, home ownership, purpose, employment length, and term of the loan. 

I next created a weight of evidence table for the continuous variables remaining. Weight of evidence is a standard methodology used in the credit industry. Weight of evidence is the log of % good divided by log % bad. By binning the continuous variables and then bucketing them based on their weight of evidence score, I can create relevant binary categories for each feature. For example, by evaluating a number of bin quantities for interest rate (ranging from 20 to 300), I can determine where I get a reasonable breakdown of counts and clustering of weight of evidence scores. If I determine that there is a large clustering of weight of evidence between 10% and 17% interest rate but then see a spike up, I can bin all rates between 10 and 17% together and create a separate bin for the next cluster. I did this for the following categories: loan amount, annual income, interest rate, revolver utilization, open credit (open acc), first three digits of zip code, and total current balance. 

After completing this process, I had 98 categories to evaluate on, all of which were one hot encoded. 

## Results of Model

### Logistic Regression Model
AUROC - 0.6999 using 10 features and 0.7016 using 13 features. 
_Note: I will show data going forward only with 13 features_ 

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/ROC_Logistic_Regression_13_factors.png)

**Confusion Matrix - Threshold of 0.5 and 13 features**

|           |      |         | 
|-----------|------|---------| 
| Predicted |  0   | 1       | 
| Actual    |      |         | 
| 0         |  769 | 74,985  | 
| 1         |  852 | 502,021 | 
|           | 1,621| 577,006 |

**Classification Report - Threshold of 0.5 and 13 features**

|           |           |        |           |          | 
|-----------|-----------|--------|-----------|----------| 
|           | Precision | Recall | f-1 score | support  | 
| 0         | 0.47      | 0.01   | 0.02      | 75,754   | 
| 1         | 0.87      | 1      | 0.93      | 502,873  | 
| avg/total | 0.82      | 0.87   | 0.81      | 578,627  | 


**Confusion Matrix - Threshold of 0.7 and 13 features**

|           |        |         | 
|-----------|--------|---------| 
| Predicted | 0      | 1       | 
| Actual    |        |         | 
| 0         | 12,650 | 63,104  | 
| 1         | 22,380 | 480,493 | 
|           | 35,030 | 543,597 | 

**Classification Report - Threshold of 0.7 and 13 features**

|           |           |        |           |          | 
|-----------|-----------|--------|-----------|----------| 
|           | Precision | Recall | f-1 score | support  | 
| 0         | 0.36      | 0.17   | 0.23      | 75,754   | 
| 1         | 0.88      | 0.96   | 0.92      | 502,873  | 
| avg/total | 0.82      | 0.85   | 0.83      | 578,627  | 



#### Comments

The logistic regression model turned out to have the best AUCROC score of all the models I used. That said, just using 0.5 for the logistic regression threshold, I did not see good performance on the recall of the model. The model largely defaulted to scoring almost everthing as a good loan, as evidenced by predicting only 0.28% of all loans as bad versus actual of 13.1%. Incidentally, using a threshold of 0.50 for all four models resulted in very low recall scores for all but the models, with bad loan recalls ranging from 0.0% to 0.4%. By increasing the threshold to 0.7, I get a better balance, as seen by the prediction increasing to 6.05% 

### Random Forest - Default Class Weighting
AUROC - 0.667 using 13 features. 

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/ROC_RF_w_13_features.png)

**Confusion Matrix - Threshold of 0.5 and 13 features**

|           |       |         | 
|-----------|-------|---------| 
| Predicted | 0     | 1       | 
| Actual    |       |         | 
| 0         | 1,595 | 74,159  | 
| 1         | 2,729 | 500,144 | 
|           | 4,324 | 574,303 | 

**Classification Report - Threshold of 0.5 and 13 features**

|           |           |        |           |          | 
|-----------|-----------|--------|-----------|----------| 
|           | Precision | Recall | f-1 score | support  | 
| 0         | 0.37      | 0.02   | 0.04      | 75,754   | 
| 1         | 0.87      | 0.99   | 0.93      | 502,873  | 
| avg/total | 0.81      | 0.87   | 0.81      | 578,627  | 

**Confusion Matrix - Threshold of 0.7 and 13 features**

|           |        |         | 
|-----------|--------|---------| 
| Predicted | 0      | 1       | 
| Actual    |        |         | 
| 0         | 15,877 | 59,877  | 
| 1         | 39,081 | 463,792 | 
|           | 54,958 | 523,669 | 


**Classification Report - Threshold of 0.7 and 13 features**

|           |           |        |           |          | 
|-----------|-----------|--------|-----------|----------| 
|           | Precision | Recall | f-1 score | support  | 
| 0         | 0.29      | 0.21   | 0.24      | 75,754   | 
| 1         | 0.89      | 0.92   | 0.9       | 502,873  | 
| avg/total | 0.81      | 0.83   | 0.82      | 578,627  | 



While the AUROC score is lower, the basic threshold evaluation of the Random Forest model captures more of the actual bad loans although still pretty poor. Predicting 2% of actual bad loans, as evidenced by the recall score, is not too helpful and barely better than 1% from the logistic regression model. At 0.7 threshold, I capture 21% of the actual bad loans. 

### Random Forest with 10:1 class weighting bad:good
AUROC at 0.6485 for 13 features

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/ROC_RF_w_10-1_Class_Weighting_13_features.png)

**Confusion Matrix - Threshold of 0.5 and 13 features**

|           |        |         | 
|-----------|--------|---------| 
| Predicted | 0      | 1       | 
| Actual    |        |         | 
| 0         | 2,731  | 73,023  | 
| 1         | 7,493  | 495,380 | 
|           | 10,224 | 568,403 | 


**Classification Report - Threshold of 0.5 and 13 features**

|           |           |        |           |          | 
|-----------|-----------|--------|-----------|----------| 
|           | Precision | Recall | f-1 score | support  | 
| 0         | 0.27      | 0.04   | 0.06      | 75,754   | 
| 1         | 0.87      | 0.99   | 0.92      | 502,873  | 
| avg/total | 0.79      | 0.86   | 0.81      | 578,627  | 



**Confusion Matrix - Threshold of 0.7 and 13 features**

|           |        |         | 
|-----------|--------|---------| 
| Predicted | 0      | 1       | 
| Actual    |        |         | 
| 0         | 18,340 | 57,414  | 
| 1         | 60,261 | 442,612 | 
|           | 78,601 | 500,026 | 


**Classification Report - Threshold of 0.7 and 13 features**

|           |           |        |           |          | 
|-----------|-----------|--------|-----------|----------| 
| x         | Precision | Recall | f-1 score | support  | 
| 0         | 0.23      | 0.24   | 0.24      | 75,754   | 
| 1         | 0.89      | 0.88   | 0.88      | 502,873  | 
| avg/total | 0.80      | 0.80   | 0.80      | 578,627  | 



#### Comments 

Again, there is a similar theme - as we try to improve prediction of bad loans, we inevitable see a trade-off on opportunity cost. Looking at the confusion matrix, for 0.7 threshold on the random forest imbalance, I have lowered my false positives but dramatically increased my true negatives. My combined f-1 scores also seem to be in the 0.80-0.83 range through all the reports so far. 

### Gradient Boosted Model
AUROC score of 0.6765

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/ROC_Gradient_Boosted_13_feature.png)

**Confusion Matrix - Threshold of 0.5 and 13 features**

|           |        |         | 
|-----------|--------|---------| 
| Predicted | 0      | 1       | 
| Actual    |        |         | 
| 0         |      0 | 75,754  | 
| 1         |      0 | 502,873 | 
|           |      0 | 578,627 | 


**Classification Report - Threshold of 0.5 and 13 features**

|           |           |        |           |          | 
|-----------|-----------|--------|-----------|----------| 
|           | Precision | Recall | f-1 score | support  | 
| 0         | 0.00      | 0.00   | 0.00      | 75,754   | 
| 1         | 0.87      | 1.00   | 0.93      | 502,873  | 
| avg/total | 0.76      | 0.87   | 0.81      | 578,627  | 



#### Comments

Well, why not... The gradient boosted model clearly needs some additional parameter tuning. Given the time it takes to run the model, it will probably be best to leverage some faster processing capacity with AWS EC2. 

## Conclusions

Overall, this is not going to make you rich. The models appear to struggle to capture actual bad loans by defaulting to optimizing the number of good loans. 

**Areas to consider for future iterations:**

* The additional of 3 features had minimal impact on the logistic regression model but perhaps other features could result in better enhancements. 

* Running the non categorical data directly on the random forest and gradient boosted models may allow the models to determine better splits and thus better predictability than the one hot encoded solutions I fed into the models. 

* Definitely an opportunity to work on improving on hyperparameters for the gradient boosted model. The inital pass was pretty disappointing but I think with some work it can be dramatically improved. 

