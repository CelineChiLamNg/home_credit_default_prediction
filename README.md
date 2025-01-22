# Home Credit Risk Prediction
*Celine Ng - December 2024*


## Objective
1. Improve risk evaluation accuracy to retail banks. In practice
   meaning target variable classification.
2. Evaluate feature importance to understand influence on home credit default

## Data
The data comes with 10 separate CSV files. It is originally based on a Kaggle
competition that is now closed,
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview).
Original CSV files can be found in the folder "data_csv".

## Technology Used
1. Python
2. Pandas
3. Seaborn
4. Matplotlib
5. scikit-learn
6. xgboost
7. lightgbm
8. optuna
9. SHAP

## Approach & Methodology
This project involves 8 tables containing a large amount of data. My 
initial approach was to aggregate these tables after briefly understanding 
the data types and information provided. I focused on delving deeper only 
into the aspects that had significant relevance to the problem at hand.

Once I aggregated the data into a single table, I followed a structured 
methodology of data exploration, preprocessing, feature engineering, model 
selection, modeling, and evaluation:

1. Data Exploration: I identified columns that needed encoding and assessed 
the information available in each feature. Most features had weak 
   correlations with the target variable, but hypothesis testing showed 
   that the presence or absence of prior information was significant for 
   predicting defaults.

2. Preprocessing: Since only tree-based models were used, this step focused 
solely on encoding categorical features.

3. Feature Engineering: New features were created based on domain knowledge, 
generated with the help of an AI tool. These features were tested using 
   cross-validation with ROC AUC (primary metric) and F1 score (secondary 
   metric). Based on the results, all features were retained for modeling.

4. Model Selection: I compared several models, including Random Forest, 
Logistic Regression, XGBoost, LightGBM, and ensemble methods. After 
   hyperparameter tuning, models were evaluated using AUC ROC and F1 Score. 
   The best threshold for the chosen model was selected.

5. Modeling: The final model was retrained using the full dataset and tested 
on the test data.

6. Evaluation: The final model was interpreted using a confusion matrix, 
feature importance, and SHAP values. 

## Results
The results showed that the model was
still influenced by class imbalance, as the model can correctly predicted 
the majority class better by a large margin. 
The most important feature
identified was 'external source mean', an aggregate feature derived from
external rating data.

## Challenges & Learnings
The largest challenges encountered in this project were mainly derived from 
the large amount of data, which translates to many missing values and
many features. Another surprising challenge faced was that LightGBM did not 
improve results after hyperparameter tuning.

**Future Work :**<br>
1. Another approach to the large amount of data would involve 
first thoroughly understanding the 
data and selecting only the most relevant features for aggregation and 
further processing.
2. The model would benefit from feature selection to 
reduce noise, as most validation/test score were very similar, 
   hyperparameter tuning didn't help much, and in 
   feature engineering it was understood that 52% of the features were not 
   important for the models.
3. Try a simpler product, including only the most 
   important maybe 20 features and quickly predict if there is even 
   potential for this client before all the data collection.
3. To improve imbalance affect, collecting more data from the minority class 
would likely improve the model’s performance. Or improve data collection 
   process to include features that can better distinguish the classes. As 
   for this project several strategies, like added weights, cross 
   validation with stratified k fold, and train test split with stratify, were 
   applied to reduce the affect.
4. It would 
also be valuable to investigate the nature of the 'external source score', 
as this feature had a significant impact on the final model.


## Instructions
1. Run the notebooks to create all the necessary files, including the final 
   model.
2. Run the Flask application with:
   python app.py
3. Once server is running, app will be accessible at: http://127.0.0.1:5000

4. The model will return a csv file with the following content: <br>
SK_ID_CURR,predictions <br>
100001,0<br>
100005,0<br>
100013,0<br>
100028,0<br>
100038,1<br>
100042,0<br>
100057,0<br>
...