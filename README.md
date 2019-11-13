# CreditRisk

## Background

Lending Club is a peer to peer lending platform that emerged in 2007. The platform allows individuals to review loan applications from others and evaluate a variety of inputs and then based on one's conclusions, the individual can commit a designated dollar amount to the applicants. Lending Club provides an API to download information about all historical loans on the platform. 

For my project, I have looked at Lending Club data from 2007 through Q2 2019. In total, there are 2,384,781 entries and a total of 150 features. While many categories have incomplete data, most of the relevant categories appear to have pretty thorough details about the loans. 

_While the scope of this project was just on identifying good versus bad loans, I would like to also apply the loan score with the associated interest rate and specifically the higher interest rate segment to identify how to maximize yield._ 

## Objective

For my project, I am looking to select some features from the dataset and see if I can build a model to predict if a loan is considered a good loan or a bad loan. I have defined good loans as Fully Paid while bad loans are defined as all other categories. 

For the dataset, good loans accounted for 78% of the total dataset while bad loans were 22%.  

I will run two separate experiments - one using a logistic regression model to establish a baseline for the test. Since logistic regression is commonly used in the banking industry, I feel this should be the benchmark for my test. Then, I will try to improve on the performance of the model by using a random forest classifier to identify good versus bad loans. 

Throughout this process, my goal is not just to get the highest accuracy of score but also minimize the false positive predictions (where I predict a loan to be good but it turns out to be bad). I will target a 5% threshold for both models, the logistic regression and the random forest. The better classifier should provide more recommended good loans and that will be the basis for my application to the Current set of loans. 
 

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

Again, for my modeling, I will exclude "Current" loans since I want to apply the model to the "Current" data set after getting a result. That leaves "Fully Paid" as my good loan bucket and all others as bad loans. I will ignore the "Does not meet the credit policy" groupings.  

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/Loan_Counts.png)

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/Distribution_of_Loans_by_Grade.png)

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/Loan_Count_by_Amount.png)

![alt text](https://github.com/fayadabbasi/CreditRisk/blob/master/ROC_Images/Loan_Count_by_IR.png)


## Critera

I have used nearly 40 criteria for my model before any of the data preprocessing. These include:
 
Funded Amount, Funded Amount, Investment, Installment, Debt to Income (DTI), Delinquencies in past 2 years, FICO range, Inquiries in past 6 months, number of Public derogatory Records, Revolver Balance, Accounts Now Delinquent, Chargeoffs in the past 12 months, Tax Liens, Employment Length, Term of the Loan, Grade (provided by Lending Club) and Sub-Grade, Home Ownership Status, Verification Status, Purpose of the Loan, Initial Listing Status, Address State, Loan Amount, Annual Income (self-reported), Interest Rate, Revolver Utilization, Open Accounts, Zip Code, Public Record of Bankruptcies, All debt Utilization, Collections in the past 12 months excluding medical, Delinquent Amount, Number of Personal Finance Inquiries, Months since last Delinquency, Num of accounts over 120 days past due, number of active revolver trades, number of revolving accounts, total recorded late fees. 

Future tests could include adding more features depending on information gain from each additional feature addition. 

## Process

I first performed some exploration of the dataset. While I initially tried evaluating the entire 150 feature dataset, I narrowed the scope to the above mentioned criteria. I wrote a function to handle all the preprocessing, which can be found here: https://github.com/fayadabbasi/CreditRisk/blob/master/preprocessing.py

For null values, I evaluated what the best way to handle each would be - in the case of annual income, for instance, I imputed the mean annual income while for current balance, I imputed a zero value for missing data. 

Next, I one hot encoded all categorical values. This included: grade and sub grade, home ownership,  verification status, purpose, initial list status, address state, term of loan, employment length. https://github.com/fayadabbasi/CreditRisk/blob/master/ohe.py

I next created a weight of evidence table for the continuous variables. Weight of evidence is a standard methodology used in the credit industry. Weight of evidence is the log of % good divided by log % bad. By binning the continuous variables and then bucketing them based on their weight of evidence score or information value, I can create relevant binary categories for each feature. This is useful for the logisitc regression model. https://github.com/fayadabbasi/CreditRisk/blob/master/woe.py

For example, by evaluating a number of bin quantities for interest rate (ranging from 20 to 300), I can determine where I get a reasonable breakdown of counts and clustering of weight of evidence scores. If I determine that there is a large clustering of weight of evidence between 10% and 17% interest rate but then see a spike up, I can bin all rates between 10 and 17% together and create a separate bin for the next cluster. It is important to verify that the binning is consistent between the training and test data sets. 

As a side note, this is a very labor intensive process and something I can look to automate in the future.  

After completing this process, I had 167 categories to evaluate on for my logistic regression model and 96 categories for my random forest model. The random forest model retained the continuous variables so that the model can determine where to split the trees.  

## Hyperparameter tuning

For the logistic regression, the model was pretty straight forward - I did adjust for the class weightings by overweighting bad loans by a factor of 5. https://github.com/fayadabbasi/CreditRisk/blob/master/logreg.py

I did a feature search on my random forest model and landed on the following settings - class weighting of 20:1 for bad versus good loans, 70 estimators, no max depth, and a threshold of 0.17. There may be opportunities to further refine these parameters as well. https://github.com/fayadabbasi/CreditRisk/blob/master/randforest.py

## Results of Model

Since I optimized on a 5% false positive rate for the two models, the benchmark for determining the quality of the two models was the number of good loan predictions. I did this for a couple of reasons: 

1. A pure accuracy score would not be helpful for such an imbalanced class. As I was working through the modeling of this dataset, the tradeoff's I faced were related to the number of correctly predicted good loans versus the total number of loans available. 

2. Still, I was able to generate 7,000 more good loan predictions from the random forest model compared to the logistic regression model, or 175,000 overall versus 168,000. While a small percentage improvement, the dollar impact, assuming $100 invested per loan, improves the total investment opportunity by $700,000. Continued feature engineering and model refinement could further increase the predictions of the random forest model. 

3. The random forest model had an AUCROC of 0.73 compared to 0.67 for the logistic regression model, while maintaining similar recall scores on bad loans of about 0.75. 

4. The top five feature importance criteria of the random forest model were: interest rate, debt to income, revolver balance, zip code, and revolver utilization. 

## Conclusions

There are many factors to consider from this test. First, we are analyzing historical performance from 2007 - Q2 2019, which includes one of the strongest period of economic growth and low relative interest rates in the economy in the past century. As such, a decline in economic factors could have a material impact on the previous range of data analyzed. 

Second, a net 12.4% return using the random forest model compared to a straight average return of 17.4% for the S&P 500 over the same time frame is not a compelling investment opportunity. One of the next steps to consider is identifying characteristics of the lower grade / higher interest rate segment of the portfolio to improve on the net return while minimizing predicted bad loans. 

Third, The net yield of 12.4% is still better than the net yield of about 11% for the overall lending club portfolio so the model has added 140 basis points of value. Again, further refinements can lead to greater investment return opportunities. 

**Areas to consider for future iterations:**

* Now that I have a foundational model, future exploration of the dataset can focus on the higher interest rate segment of the portfolio. This will give me an opportunity to fine tune and drive for higher yield in the more lucrative segment of the portofolio. 

* Build a Flask interface to allow the model to update every quarter as Lending Club updates their loan listings and rerun the model as older loans move from Current status to Fully Paid or other status.

