# CreditRisk

## Background

Lending Club is a peer to peer lending platform that emerged in 2007. The platform allows individuals to review loan applications from others and evaluate a variety of inputs and then based on one's conclusions, the individual can commit a designated dollar amount to the applicants. Lending Club provides an API to download information about all historical loans on the platform. 

For my project, I have looked at Lending Club data from 2007 through Q2 2019. In total, there are 2,384,848 entries and a total of 150 features. While many categories have incomplete data, most of the relevant categories appear to have a pretty thorough details about the loans. 

## Objective

For my project, I am looking to select some features from the dataset and see if I can build a model to predict if a loan is considered a good loan or a bad loan. I have defined good loans as Fully Paid or Current while bad loans are defined as all other categories. 

For the dataset, good loans accounted for 86% of the total dataset while bad loans were 14%. Of the bad loans, 88% were charged off. 

I will one hot encode two different sets of data: my first set is based on 10 criteria that I feel could be good predictors and the second set is based on an additional 13 criteria. The criteria are listed below with a brief description. 

I begin by applying a logistic regression model. I will split my dataset into a train/test split (75%/25%) and then train the model on the one hot encoded data. From that, I will test against the test dataset and see what my area under curve is for my ROC curve. 

I will then do the same with a random forest, one test on a default class weighting and a second over-weighting the bad loans. I am using n_estimators at 100 and no max depth to allow the trees to grow fully. For the class weighted, I am using a 10:1 weighting for bad:good to better represent the bad loan category. 

Finally I will run a gradient boosted test. For the boosted model, I am using a learning rate of 0.001, n_estimators of 300, and max depth of 2. In the future, I can apply more tuning to this model, potentially including multiple models on an EC2 instance to better determine optimal hyper parameters. As it is, with the parameters I chose, given the large dataset, it takes roughly 20min to train the gradient boosted model. 

## Loan Characteristics

Below is a simple breakdown of the total loans in the dataset. 

| Status                                              | # Loans   | 
|-----------------------------------------------------|-----------| 
| Fully Paid                                          | 1,191,125 | 
| Current                                             |   868,848 | 
| Charged Off                                         |   287,174 | 
| Late (31-120 days)                                  |    20,775 | 
| In Grace Period                                     |     9,242 | 
| Late (16-30 days)                                   |     4,826 | 
| Does not meet the credit policy. Status:Fully Paid  |     1,988 | 
| Does not meet the credit policy. Status:Charged Off |       761 | 
| Default                                             |        42 | 

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



