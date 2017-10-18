# TalkingData Mobile User Demographics
## 1.	Overview of data
The code is trained to predict the user demographic. Input data includes the users’ the phone brands, apps installed in the phone, and usage of the phones. The usage of the phone is characterized by events each related with a specific app along with detailed location and time information. Output data is the user demographic based on the input data, i.e. the probability for each of the 12 age classes. To estimate the accuracy of the prediction, we used logarithmic loss as evaluation metrics:

![alt text](https://latex.codecogs.com/gif.latex?-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Csum_%7Bj%3D1%7D%5E%7BM%7Dy_%7Bij%7Dlog%28p_%7Bij%7D%29)


Where N is the number of instance and M is the number of labels.  is the indicator value that whether instance i belongs to labels j.  is the predicted probability.

## 2.	Feature engineering
### 2.1	Separate data into two groups
We noticed that 69 percent of the data has no event information. This could come from the fact that the users disable the data sharing function or they are inactive users. If it is because of the first reason, treating the features as no active usage will clearly mislead the model. So we separated the data into two groups and built models for both sets of data. Test data were also separated into two groups and fed into the corresponding models, after which the output was combined into the final output.
The techniques used for both sets of data are similar so the text below will not specify which set of data is used unless there is an obvious difference when processing them. 
### 2.2	Generate dummy variables
Some features such as phone brands, phone device mode, installed apps, app label and active app are nominal data, so we encoded them into multiple dummy variables with each nominal value as a new feature. The value will be 1 if it is some nominal value and 0 otherwise. NA data are easily taken care of during the process because the value will be 0 in every feature in the current feature set. Even though the active time period are ordinal data, we discretize the data into 24 intervals each representing one hour period. Dummy variables are generated for the active time period data too.
These sets of features are stored in different data frames, so we can select any combination of the features during model training and selection process. We can simply use hstack function to stack any desired features.

## 3.	Model training
Before training the model, 20% of the data are reserved as validation data. When splitting the data into training and validation data, we stratify by the group label, i.e. the percentage of different groups are the same before and after splitting.
### 3.1 Logistic regression
As a simple starter, we use a multinomial logistic regression model. The target group are represented by response values of 1 to 12. Default l2 regulation was used. We used cross validation to grid search the best regulation strength.
### 3.2 XGboost
As one of the most competitive boosted tree tool in Kaggle competition, XGboost was used to build the second model. Cross validation was also used to grid search the best hyperparameters including the maximum tree depth, the learning rate and the two regulation strength parameters. The final model is trained with the best hyperparameters with the whole data.
### 3.3 Neural Network
We also build a two-layer neural network to capture any possible complicated relationship in the data set. During each epoch (one iteration of all data), mini-batch gradient decent was used to conform with the memory requirement of the computer and make the model suffer less from local minimum. Normal initialization was used in parameter initialization. Dropout layers were used to prevent over fitting. Different parameter update rules including sgd, adadelta, adam and RMSprop etc. were tested to reduce the loss function, we found that both adadelta and adam work the best.

## 4.	Ensemble
Using the methods in Section 3, we trained one simple logistic regression model, 2 XGboost models and 3 neural network models. We performed a weighted average of all results generated from different models, with larger weights in XGboost and neural network models. This gave us a non-trivial boost in the overall score.


