# Goal:
 - analyze the dataset and come up with a model that will best detect fraudalent transactions
 - compare different popular models and determine which ones perform better
 - explore machine learning validation metrics to determine quality of each model (AUPRC vs AUC-ROC)
 - with the highly imbalanced data set, try different data analysis techiniques (oversampling/undersampling)
 - go back and use some oversampling method to see how the models change with the availability of more data
 - dig into each method's parameters and tune it to see if we can get better results. Determine whether or not the dataset was optimized for this kind of problem

## Notes:
 - Dataset was financial transactions dataset from Kaggle
 - Reduced Featureset: 28 Features determined from prior PCA analysis. Original features were scrubbed for user anonymity
 - Time & Amount are the only two original features
 - Total Samples in DataSet: 284,807. Number of Fraudalent transactions: 492 (0.172%) of all transactions. Represented by "Class" Feature

## TODO:
- [x] get XGBoost working
- create baseline model comparison dataframe with confusion matrix results of all the models
- do AUPRC vs AUC-ROC comparison / analysis
- do randomundersampler (but better version)
- do oversampling method for 5k, 10k, 100k, equal parity
- re-run the same algos, see how they change over time with more data, or as the data changes
- visualize the efficacy of each model with more time 

## Helpful Articles:
- Kaggle Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/3
- Useful for understanding SMOTE and why we need to oversample on imbalanced datasets: https://towardsdatascience.com/detecting-financial-fraud-using-machine-learning-three-ways-of-winning-the-war-against-imbalanced-a03f8815cce9 
- walkthrough of same dataset and analysis of different models: https://towardsdatascience.com/detecting-credit-card-fraud-using-machine-learning-a3d83423d3b8
- standardization vs normalization: https://wikidiff.com/standardization/normalization
- AUCROC vs AUPRC: https://stats.stackexchange.com/questions/338826/auprc-vs-auc-roc
- how to build logistic regression: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
- using pandas to create training sets for models: https://www.ritchieng.com/pandas-scikit-learn/
- SVM in pandas: http://benalexkeen.com/support-vector-classifiers-in-python-using-scikit-learn/
- Getting XGBoost to work: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
