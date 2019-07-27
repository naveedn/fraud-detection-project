# Notes:
 - Reduced FeatureSet: 28 Features determined from prior PCA analysis. Original features were scrubbed for user anonymity
 - Time & Amount are the only two original features
 - Total Samples in DataSet: 284,807. Number of Fraudalent transactions: 492 (0.172%) of all transactions. Represented by "Class" Feature
 
# Goal:
 - with the highly imbalanced data set, normalize the data by normalizing  underfitting the data
 - compare different models and estimate which ones perform better
 - go back and use some oversampling method to see how the models change with the availability of more data
 - dig into each method and tune it to see if we can get better results

# Helpful Articles
- Kaggle Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/3
- Useful for understanding SMOTE and why we need to oversample on imbalanced datasets: https://towardsdatascience.com/detecting-financial-fraud-using-machine-learning-three-ways-of-winning-the-war-against-imbalanced-a03f8815cce9 
- walkthrough of same dataset and analysis of different models: https://towardsdatascience.com/detecting-credit-card-fraud-using-machine-learning-a3d83423d3b8
- standardization vs normalization: https://wikidiff.com/standardization/normalization
- AUCROC vs AUPRC: https://stats.stackexchange.com/questions/338826/auprc-vs-auc-roc
- how to build logistic regression: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
- using pandas to create training sets for models: https://www.ritchieng.com/pandas-scikit-learn/
- SVM in pandas: http://benalexkeen.com/support-vector-classifiers-in-python-using-scikit-learn/
- Getting XGBoost to work: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/