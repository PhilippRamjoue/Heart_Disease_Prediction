<div class="cell code" data-execution_count="1" data-collapsed="true" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
```

</div>

<div class="cell markdown" data-collapsed="false">

Following features are included in the dataset:

  - age
  - sex (1 = man, 0 = woman)
  - cp; chest pain type (4 values)
  - trestbps; resting blood pressure
  - chol; serum cholestoral in mg/dl
  - fbs; fasting blood sugar \> 120 mg/dl
  - restecg; resting electrocardiographic results (values 0,1,2)
  - thalach; maximum heart rate achieved
  - exang; exercise induced angina -\> durch Belastung ausgelöste Angina
  - oldpeak = ST depression induced by exercise relative to rest
  - slope; the slope of the peak exercise ST segment
  - ca; number of major vessels (0-3) colored by flourosopy -\> große
    Gefäße
  - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

The target is 1 (heart disease) or 0 (no heart disease).

</div>

<div class="cell code" data-execution_count="2" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
# Check out the data https://www.kaggle.com/ronitf/heart-disease-uci
dataset = pd.read_csv('heart.csv', sep=',')

dataset.head()
```

<div class="output execute_result" data-execution_count="2">

``` 
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \
0   63    1   3       145   233    1        0      150      0      2.3      0   
1   37    1   2       130   250    0        1      187      0      3.5      0   
2   41    0   1       130   204    0        0      172      0      1.4      2   
3   56    1   1       120   236    0        1      178      0      0.8      2   
4   57    0   0       120   354    0        1      163      1      0.6      2   

   ca  thal  target  
0   0     1       1  
1   0     2       1  
2   0     2       1  
3   0     2       1  
4   0     2       1  
```

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

The next step is to analyze the dataset and clean the data. To achieve
good results with neural networks the features have to be in a range of
0 - 1.

</div>

<div class="cell code" data-execution_count="3" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
# Prepare the data
def prepare_data(org_dataset):

    # use clearer column names
    clear_names_column = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
                          'serum_cholestoral', 'fasting_blood_sugar', 'resting_electrocardiographic_results',
                          'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', 'slope', 'major_vessels',
                          'thal', 'target']

    dataset = copy.deepcopy(org_dataset)

    # Insert cleaner names
    dataset.columns = clear_names_column

    # cleaning data
    # 'ca' (number of major vessels) is in the range from 0-3 -> there a datapoints with the value 4
    # 'thal' ranges from 1-3 (1 = normal; 2 = fixed defect; 3 = reversable defect) -> there aere datapoints with the value 0
    # We don't know what the correct data shall be so we delete the rows with this values
    dataset['major_vessels'] = dataset.major_vessels.apply(lambda s: np.NaN if s == 4 else s)
    dataset['thal'] = dataset.thal.apply(lambda s: np.NaN if s == 0 else s)

    dataset.dropna(axis=0, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # min-max scaling for:
    # age
    # trestbps; resting_blood_pressure
    # chol; serum_cholestoral
    # thalach; maximum_heart_rate_achieved
    # oldpeak

    list_for_minmax_scaling = ['age', 'resting_blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved',
                               'oldpeak']

    scaler = MinMaxScaler()

    dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset[list_for_minmax_scaling]),
                                  columns=list_for_minmax_scaling)

    dataset.drop(columns=list_for_minmax_scaling, inplace=True, axis=1)

    dataset = dataset.join(dataset_scaled)

    # dummy encoding for:
    # cp; chest_pain_type
    # restecg; resting_electrocardiographic_results
    # slope
    # ca; major_vessels
    # thal

    # Dummy encoding
    list_for_dummy_encoding = ['chest_pain_type', 'resting_electrocardiographic_results', 'slope', 'major_vessels',
                               'thal']

    dataset_finalized = pd.get_dummies(dataset, columns=list_for_dummy_encoding)

    dataset_finalized = dataset_finalized.rename(
        columns = {'chest_pain_type_0':'chest_pain_typical_angina',
                   'chest_pain_type_1':'chest_pain_atypical_angina',
                   'chest_pain_type_2':'chest_pain_non_anginal_pain',
                   'chest_pain_type_3':'chest_pain_asymptomatic',
                   'resting_electrocardiographic_results_0':'rest_ecg_normal',
                   'resting_electrocardiographic_results_1':'rest_ecg_wave_abnormality',
                   'resting_electrocardiographic_results_2':'rest_ecg_ventricular_hypertrophy',
                   'slope_0':'slope_upsloping',
                   'slope_1':'slope_flat',
                   'slope_2':'slope_downsloping',
                   'thal_1.0':'thal_normal',
                   'thal_2.0':'thal_fixed_defect',
                   'thal_3.0':'thal_reversable_defect',
                   })

    # no tranformation nedded for :
    # sex
    # fbs; fasting_blood_sugar
    # exang; exercise_induced_angina

    label_sklearn = dataset_finalized.pop("target")

    return dataset_finalized, label_sklearn
```

</div>

<div class="cell code" data-execution_count="4" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
features, labels = prepare_data(dataset)

features.head()
```

<div class="output execute_result" data-execution_count="4">

``` 
   sex  fasting_blood_sugar  exercise_induced_angina       age  \
0    1                    1                        0  0.708333   
1    1                    0                        0  0.166667   
2    0                    0                        0  0.250000   
3    1                    0                        0  0.562500   
4    0                    0                        1  0.583333   

   resting_blood_pressure  serum_cholestoral  maximum_heart_rate_achieved  \
0                0.481132           0.244292                     0.603053   
1                0.339623           0.283105                     0.885496   
2                0.339623           0.178082                     0.770992   
3                0.245283           0.251142                     0.816794   
4                0.245283           0.520548                     0.702290   

    oldpeak  chest_pain_typical_angina  chest_pain_atypical_angina  ...  \
0  0.370968                          0                           0  ...   
1  0.564516                          0                           0  ...   
2  0.225806                          0                           1  ...   
3  0.129032                          0                           1  ...   
4  0.096774                          1                           0  ...   

   slope_upsloping  slope_flat  slope_downsloping  major_vessels_0.0  \
0                1           0                  0                  1   
1                1           0                  0                  1   
2                0           0                  1                  1   
3                0           0                  1                  1   
4                0           0                  1                  1   

   major_vessels_1.0  major_vessels_2.0  major_vessels_3.0  thal_normal  \
0                  0                  0                  0            1   
1                  0                  0                  0            0   
2                  0                  0                  0            0   
3                  0                  0                  0            0   
4                  0                  0                  0            0   

   thal_fixed_defect  thal_reversable_defect  
0                  0                       0  
1                  1                       0  
2                  1                       0  
3                  1                       0  
4                  1                       0  

[5 rows x 25 columns]
```

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

**Split the dataset in train and test set**

</div>

<div class="cell code" data-execution_count="5" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33,
                                                                    random_state=42)
```

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

For comparison three different sklearn models are trained:

  - [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gradientboost#sklearn.ensemble.GradientBoostingClassifier)
  - [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression)

</div>

<div class="cell code" data-execution_count="6" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
random_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train, y_train)
random_score = random_clf.score(X_test, y_test)
print("RandomForest Score: %f" % random_score)

grad_clf = GradientBoostingClassifier().fit(X_train, y_train)
grad_score = grad_clf.score(X_test, y_test)
print("GradientBoosting Score: %f" % grad_score)

lr_clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, y_train)
lr_score = lr_clf.score(X_test, y_test)
print("LogisticRegression Score: %f" % lr_score)
```

<div class="output stream stdout">

    RandomForest Score: 0.836735
    GradientBoosting Score: 0.826531
    LogisticRegression Score: 0.877551

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

The *score* function only measures the accuracy. For me the most
important metric is the **recall/sensitivity/true positive rate**
because it is especially important that the models detect as many heart
problems as possible.

To achieve this, the confusion matrices of the models are analyzed:

</div>

<div class="cell code" data-execution_count="7" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
def visualize_confusion(conf_matrix, name):
    # normalize values
    normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    conf_figure, ax = plt.subplots(figsize=(7.5, 7.5))

    ax.matshow(normalized_conf_matrix, cmap=plt.cm.BuPu)  # , alpha=0.3)
    for i in range(normalized_conf_matrix.shape[0]):
        for j in range(normalized_conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=round(normalized_conf_matrix[i, j], 2), va='center', ha='center', size='xx-large')

    conf_figure.suptitle('Confusion matrix')
    tick_marks = np.arange(len(conf_matrix))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(['no_heart_disease', 'heart_disease'])
    ax.set_yticklabels(['no_heart_disease', 'heart_disease'])
    ax.set_yticks(tick_marks)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    filename = name + '_Confusion_matrix'

    conf_figure.savefig(filename, bbox_inches='tight')
    plt.show()

    return normalized_conf_matrix[1, 1]
```

</div>

<div class="cell code" data-execution_count="8" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
rf_name = 'random_forest'
prediction = random_clf.predict(X_test.astype(np.float32))
rounded_prediction = tf.math.round(prediction)
# create a confusion matrix
conf_matrix = confusion_matrix(y_true=y_test.astype(np.float32), y_pred=rounded_prediction)
tpr = visualize_confusion(conf_matrix, rf_name)

print('The accuracy is: ' +'{:.1%}'.format(random_score))
print('The true positive rate is: ' +'{:.1%}'.format(tpr))
```

<div class="output stream stderr">

    FixedFormatter should only be used together with FixedLocator

</div>

<div class="output display_data">

![](809f47dd1453bed0bc34578d36132f528db88477.png)

</div>

<div class="output stream stdout">

    The accuracy is: 83.7%
    The true positive rate is: 82.8%

</div>

</div>

<div class="cell code" data-execution_count="9" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
grad_name = 'gradient_boost'
prediction = grad_clf.predict(X_test.astype(np.float32))
rounded_prediction = tf.math.round(prediction)
# create a confusion matrix
conf_matrix = confusion_matrix(y_true=y_test.astype(np.float32), y_pred=rounded_prediction)
tpr = visualize_confusion(conf_matrix, grad_name)

print('The accuracy is: ' +'{:.1%}'.format(grad_score))
print('The true positive rate is: ' +'{:.1%}'.format(tpr))
```

<div class="output stream stderr">

    FixedFormatter should only be used together with FixedLocator

</div>

<div class="output display_data">

![](aa40b43a7fc53845a64c9287ee0f272bc58b55e0.png)

</div>

<div class="output stream stdout">

    The accuracy is: 82.7%
    The true positive rate is: 84.5%

</div>

</div>

<div class="cell code" data-execution_count="10" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
lr_name = 'logistic_regression'
prediction = lr_clf.predict(X_test.astype(np.float32))
rounded_prediction = tf.math.round(prediction)
# create a confusion matrix
conf_matrix = confusion_matrix(y_true=y_test.astype(np.float32), y_pred=rounded_prediction)
tpr = visualize_confusion(conf_matrix, lr_name)

print('The accuracy is: ' +'{:.1%}'.format(lr_score))
print('The true positive rate is: ' +'{:.1%}'.format(tpr))
```

<div class="output stream stderr">

    FixedFormatter should only be used together with FixedLocator

</div>

<div class="output display_data">

![](b0e98e526953d3bf697c15506902ab4ea6fcc13b.png)

</div>

<div class="output stream stdout">

    The accuracy is: 87.8%
    The true positive rate is: 87.9%

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

Next step is to train a neural network. For this a sequential model with
25 input nodes, 13 hidden nodes, 1 dropout layer and 1 node output layer
with sigmoid activation function is created.

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

**Build the model**

</div>

<div class="cell code" data-execution_count="11" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(13, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
```

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

To achieve good results the Adam optimizer and a binary crossentropy
loss function is chosen. For comparison the metrics 'accuracy' and
'recall' are used.

</div>

<div class="cell code" data-execution_count="12" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Recall(name='recall'), 'accuracy'])
```

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

To avoid overfitting an EarlyStopping callback is used. The monitoring
feature is the loss. The callback will stop the training if the loss
will not decrease within 100 iterations (*patience*). The weights of the
best run will be restored for the final model.

</div>

<div class="cell code" data-execution_count="13" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
early_stopping_monitor = EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=50,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)
```

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

**Train the model**

</div>

<div class="cell code" data-execution_count="14" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
history = model.fit(X_train.astype(np.float32),
                    y_train.astype(np.float32),
                    callbacks=[early_stopping_monitor],
                    epochs=1000,
                    verbose=2)
```

<div class="output stream stdout">

    Epoch 1/1000
    7/7 - 0s - loss: 0.8237 - recall: 0.0392 - accuracy: 0.5000
    Epoch 2/1000
    7/7 - 0s - loss: 0.7507 - recall: 0.0588 - accuracy: 0.5051
    Epoch 3/1000
    7/7 - 0s - loss: 0.7178 - recall: 0.0980 - accuracy: 0.5000
    Epoch 4/1000
    7/7 - 0s - loss: 0.6897 - recall: 0.1471 - accuracy: 0.5354
    Epoch 5/1000
    7/7 - 0s - loss: 0.6884 - recall: 0.1667 - accuracy: 0.5152
    Epoch 6/1000
    7/7 - 0s - loss: 0.6671 - recall: 0.2157 - accuracy: 0.5152
    Epoch 7/1000
    7/7 - 0s - loss: 0.6611 - recall: 0.3235 - accuracy: 0.5808
    Epoch 8/1000
    7/7 - 0s - loss: 0.6280 - recall: 0.4608 - accuracy: 0.6414
    Epoch 9/1000
    7/7 - 0s - loss: 0.6151 - recall: 0.4706 - accuracy: 0.6818
    Epoch 10/1000
    7/7 - 0s - loss: 0.6062 - recall: 0.5686 - accuracy: 0.7020
    Epoch 11/1000
    7/7 - 0s - loss: 0.5729 - recall: 0.7157 - accuracy: 0.7727
    Epoch 12/1000
    7/7 - 0s - loss: 0.5652 - recall: 0.6863 - accuracy: 0.7677
    Epoch 13/1000
    7/7 - 0s - loss: 0.5811 - recall: 0.7059 - accuracy: 0.7626
    Epoch 14/1000
    7/7 - 0s - loss: 0.5677 - recall: 0.7059 - accuracy: 0.7475
    Epoch 15/1000
    7/7 - 0s - loss: 0.5521 - recall: 0.7353 - accuracy: 0.7727
    Epoch 16/1000
    7/7 - 0s - loss: 0.5561 - recall: 0.7353 - accuracy: 0.7475
    Epoch 17/1000
    7/7 - 0s - loss: 0.5384 - recall: 0.7745 - accuracy: 0.7576
    Epoch 18/1000
    7/7 - 0s - loss: 0.5119 - recall: 0.7647 - accuracy: 0.7929
    Epoch 19/1000
    7/7 - 0s - loss: 0.5063 - recall: 0.8137 - accuracy: 0.8030
    Epoch 20/1000
    7/7 - 0s - loss: 0.5060 - recall: 0.8039 - accuracy: 0.8081
    Epoch 21/1000
    7/7 - 0s - loss: 0.5039 - recall: 0.7941 - accuracy: 0.8030
    Epoch 22/1000
    7/7 - 0s - loss: 0.5074 - recall: 0.7843 - accuracy: 0.8131
    Epoch 23/1000
    7/7 - 0s - loss: 0.4812 - recall: 0.7843 - accuracy: 0.8030
    Epoch 24/1000
    7/7 - 0s - loss: 0.4830 - recall: 0.7647 - accuracy: 0.8081
    Epoch 25/1000
    7/7 - 0s - loss: 0.4812 - recall: 0.8039 - accuracy: 0.8182
    Epoch 26/1000
    7/7 - 0s - loss: 0.4636 - recall: 0.8333 - accuracy: 0.8232
    Epoch 27/1000
    7/7 - 0s - loss: 0.4745 - recall: 0.7745 - accuracy: 0.7727
    Epoch 28/1000
    7/7 - 0s - loss: 0.4683 - recall: 0.8333 - accuracy: 0.8182
    Epoch 29/1000
    7/7 - 0s - loss: 0.4502 - recall: 0.8235 - accuracy: 0.8131
    Epoch 30/1000
    7/7 - 0s - loss: 0.4521 - recall: 0.8431 - accuracy: 0.8232
    Epoch 31/1000
    7/7 - 0s - loss: 0.4591 - recall: 0.7843 - accuracy: 0.7980
    Epoch 32/1000
    7/7 - 0s - loss: 0.4287 - recall: 0.8137 - accuracy: 0.8384
    Epoch 33/1000
    7/7 - 0s - loss: 0.4357 - recall: 0.8137 - accuracy: 0.8182
    Epoch 34/1000
    7/7 - 0s - loss: 0.4469 - recall: 0.8529 - accuracy: 0.8333
    Epoch 35/1000
    7/7 - 0s - loss: 0.4299 - recall: 0.8039 - accuracy: 0.8081
    Epoch 36/1000
    7/7 - 0s - loss: 0.4370 - recall: 0.8431 - accuracy: 0.8384
    Epoch 37/1000
    7/7 - 0s - loss: 0.4305 - recall: 0.8137 - accuracy: 0.8333
    Epoch 38/1000
    7/7 - 0s - loss: 0.4112 - recall: 0.8529 - accuracy: 0.8535
    Epoch 39/1000
    7/7 - 0s - loss: 0.4093 - recall: 0.8824 - accuracy: 0.8485
    Epoch 40/1000
    7/7 - 0s - loss: 0.4299 - recall: 0.8235 - accuracy: 0.8182
    Epoch 41/1000
    7/7 - 0s - loss: 0.4144 - recall: 0.8235 - accuracy: 0.8131
    Epoch 42/1000
    7/7 - 0s - loss: 0.4000 - recall: 0.8824 - accuracy: 0.8485
    Epoch 43/1000
    7/7 - 0s - loss: 0.4285 - recall: 0.8333 - accuracy: 0.8283
    Epoch 44/1000
    7/7 - 0s - loss: 0.4404 - recall: 0.8431 - accuracy: 0.8182
    Epoch 45/1000
    7/7 - 0s - loss: 0.4110 - recall: 0.8137 - accuracy: 0.8283
    Epoch 46/1000
    7/7 - 0s - loss: 0.4007 - recall: 0.8431 - accuracy: 0.8232
    Epoch 47/1000
    7/7 - 0s - loss: 0.4023 - recall: 0.8824 - accuracy: 0.8434
    Epoch 48/1000
    7/7 - 0s - loss: 0.3789 - recall: 0.8235 - accuracy: 0.8283
    Epoch 49/1000
    7/7 - 0s - loss: 0.3792 - recall: 0.8922 - accuracy: 0.8384
    Epoch 50/1000
    7/7 - 0s - loss: 0.4137 - recall: 0.8333 - accuracy: 0.8030
    Epoch 51/1000
    7/7 - 0s - loss: 0.4298 - recall: 0.8529 - accuracy: 0.8333
    Epoch 52/1000
    7/7 - 0s - loss: 0.4407 - recall: 0.8431 - accuracy: 0.8333
    Epoch 53/1000
    7/7 - 0s - loss: 0.4019 - recall: 0.8529 - accuracy: 0.8333
    Epoch 54/1000
    7/7 - 0s - loss: 0.4002 - recall: 0.8627 - accuracy: 0.8333
    Epoch 55/1000
    7/7 - 0s - loss: 0.3952 - recall: 0.8529 - accuracy: 0.8535
    Epoch 56/1000
    7/7 - 0s - loss: 0.3858 - recall: 0.8431 - accuracy: 0.8384
    Epoch 57/1000
    7/7 - 0s - loss: 0.3994 - recall: 0.8333 - accuracy: 0.8232
    Epoch 58/1000
    7/7 - 0s - loss: 0.3955 - recall: 0.8922 - accuracy: 0.8283
    Epoch 59/1000
    7/7 - 0s - loss: 0.3712 - recall: 0.8431 - accuracy: 0.8232
    Epoch 60/1000
    7/7 - 0s - loss: 0.3631 - recall: 0.8627 - accuracy: 0.8636
    Epoch 61/1000
    7/7 - 0s - loss: 0.3647 - recall: 0.8627 - accuracy: 0.8687
    Epoch 62/1000
    7/7 - 0s - loss: 0.3953 - recall: 0.8725 - accuracy: 0.8586
    Epoch 63/1000
    7/7 - 0s - loss: 0.3861 - recall: 0.8333 - accuracy: 0.8232
    Epoch 64/1000
    7/7 - 0s - loss: 0.3710 - recall: 0.8627 - accuracy: 0.8485
    Epoch 65/1000
    7/7 - 0s - loss: 0.3869 - recall: 0.8529 - accuracy: 0.8333
    Epoch 66/1000
    7/7 - 0s - loss: 0.3812 - recall: 0.8431 - accuracy: 0.8434
    Epoch 67/1000
    7/7 - 0s - loss: 0.3939 - recall: 0.8235 - accuracy: 0.8283
    Epoch 68/1000
    7/7 - 0s - loss: 0.3631 - recall: 0.8824 - accuracy: 0.8586
    Epoch 69/1000
    7/7 - 0s - loss: 0.3608 - recall: 0.8824 - accuracy: 0.8788
    Epoch 70/1000
    7/7 - 0s - loss: 0.3939 - recall: 0.8627 - accuracy: 0.8283
    Epoch 71/1000
    7/7 - 0s - loss: 0.3568 - recall: 0.8922 - accuracy: 0.8586
    Epoch 72/1000
    7/7 - 0s - loss: 0.3734 - recall: 0.8627 - accuracy: 0.8384
    Epoch 73/1000
    7/7 - 0s - loss: 0.3961 - recall: 0.8235 - accuracy: 0.8283
    Epoch 74/1000
    7/7 - 0s - loss: 0.3657 - recall: 0.8922 - accuracy: 0.8636
    Epoch 75/1000
    7/7 - 0s - loss: 0.3731 - recall: 0.8627 - accuracy: 0.8535
    Epoch 76/1000
    7/7 - 0s - loss: 0.3856 - recall: 0.8627 - accuracy: 0.8384
    Epoch 77/1000
    7/7 - 0s - loss: 0.3826 - recall: 0.8333 - accuracy: 0.8232
    Epoch 78/1000
    7/7 - 0s - loss: 0.3638 - recall: 0.8627 - accuracy: 0.8485
    Epoch 79/1000
    7/7 - 0s - loss: 0.3518 - recall: 0.9020 - accuracy: 0.8687
    Epoch 80/1000
    7/7 - 0s - loss: 0.3621 - recall: 0.8922 - accuracy: 0.8788
    Epoch 81/1000
    7/7 - 0s - loss: 0.3750 - recall: 0.8137 - accuracy: 0.8485
    Epoch 82/1000
    7/7 - 0s - loss: 0.3710 - recall: 0.8529 - accuracy: 0.8384
    Epoch 83/1000
    7/7 - 0s - loss: 0.3784 - recall: 0.8039 - accuracy: 0.8232
    Epoch 84/1000
    7/7 - 0s - loss: 0.3590 - recall: 0.8627 - accuracy: 0.8485
    Epoch 85/1000
    7/7 - 0s - loss: 0.3756 - recall: 0.8431 - accuracy: 0.8384
    Epoch 86/1000
    7/7 - 0s - loss: 0.3514 - recall: 0.8235 - accuracy: 0.8384
    Epoch 87/1000
    7/7 - 0s - loss: 0.3769 - recall: 0.8627 - accuracy: 0.8384
    Epoch 88/1000
    7/7 - 0s - loss: 0.3247 - recall: 0.8725 - accuracy: 0.8788
    Epoch 89/1000
    7/7 - 0s - loss: 0.3478 - recall: 0.8725 - accuracy: 0.8333
    Epoch 90/1000
    7/7 - 0s - loss: 0.3698 - recall: 0.8529 - accuracy: 0.8384
    Epoch 91/1000
    7/7 - 0s - loss: 0.3841 - recall: 0.8431 - accuracy: 0.8434
    Epoch 92/1000
    7/7 - 0s - loss: 0.3542 - recall: 0.8725 - accuracy: 0.8838
    Epoch 93/1000
    7/7 - 0s - loss: 0.3752 - recall: 0.8333 - accuracy: 0.8384
    Epoch 94/1000
    7/7 - 0s - loss: 0.3391 - recall: 0.8725 - accuracy: 0.8687
    Epoch 95/1000
    7/7 - 0s - loss: 0.3790 - recall: 0.8627 - accuracy: 0.8636
    Epoch 96/1000
    7/7 - 0s - loss: 0.3486 - recall: 0.8824 - accuracy: 0.8535
    Epoch 97/1000
    7/7 - 0s - loss: 0.3451 - recall: 0.8824 - accuracy: 0.8535
    Epoch 98/1000
    7/7 - 0s - loss: 0.3571 - recall: 0.8627 - accuracy: 0.8434
    Epoch 99/1000
    7/7 - 0s - loss: 0.3138 - recall: 0.8627 - accuracy: 0.8687
    Epoch 100/1000
    7/7 - 0s - loss: 0.3542 - recall: 0.8529 - accuracy: 0.8485
    Epoch 101/1000
    7/7 - 0s - loss: 0.3646 - recall: 0.8922 - accuracy: 0.8687
    Epoch 102/1000
    7/7 - 0s - loss: 0.3452 - recall: 0.8529 - accuracy: 0.8535
    Epoch 103/1000
    7/7 - 0s - loss: 0.3530 - recall: 0.8627 - accuracy: 0.8384
    Epoch 104/1000
    7/7 - 0s - loss: 0.3467 - recall: 0.8627 - accuracy: 0.8636
    Epoch 105/1000
    7/7 - 0s - loss: 0.3494 - recall: 0.8529 - accuracy: 0.8636
    Epoch 106/1000
    7/7 - 0s - loss: 0.3318 - recall: 0.8922 - accuracy: 0.8737
    Epoch 107/1000
    7/7 - 0s - loss: 0.3289 - recall: 0.8922 - accuracy: 0.8889
    Epoch 108/1000
    7/7 - 0s - loss: 0.3475 - recall: 0.8627 - accuracy: 0.8586
    Epoch 109/1000
    7/7 - 0s - loss: 0.3576 - recall: 0.8431 - accuracy: 0.8535
    Epoch 110/1000
    7/7 - 0s - loss: 0.3601 - recall: 0.8725 - accuracy: 0.8586
    Epoch 111/1000
    7/7 - 0s - loss: 0.3593 - recall: 0.8725 - accuracy: 0.8687
    Epoch 112/1000
    7/7 - 0s - loss: 0.3486 - recall: 0.8922 - accuracy: 0.8838
    Epoch 113/1000
    7/7 - 0s - loss: 0.3267 - recall: 0.9216 - accuracy: 0.8788
    Epoch 114/1000
    7/7 - 0s - loss: 0.3571 - recall: 0.9020 - accuracy: 0.8788
    Epoch 115/1000
    7/7 - 0s - loss: 0.3277 - recall: 0.8627 - accuracy: 0.8485
    Epoch 116/1000
    7/7 - 0s - loss: 0.3116 - recall: 0.8725 - accuracy: 0.8737
    Epoch 117/1000
    7/7 - 0s - loss: 0.3111 - recall: 0.9020 - accuracy: 0.8889
    Epoch 118/1000
    7/7 - 0s - loss: 0.3393 - recall: 0.8922 - accuracy: 0.8636
    Epoch 119/1000
    7/7 - 0s - loss: 0.3446 - recall: 0.8824 - accuracy: 0.8737
    Epoch 120/1000
    7/7 - 0s - loss: 0.3473 - recall: 0.8529 - accuracy: 0.8636
    Epoch 121/1000
    7/7 - 0s - loss: 0.3697 - recall: 0.9020 - accuracy: 0.8586
    Epoch 122/1000
    7/7 - 0s - loss: 0.3613 - recall: 0.8333 - accuracy: 0.8586
    Epoch 123/1000
    7/7 - 0s - loss: 0.3248 - recall: 0.8235 - accuracy: 0.8485
    Epoch 124/1000
    7/7 - 0s - loss: 0.3472 - recall: 0.8137 - accuracy: 0.8131
    Epoch 125/1000
    7/7 - 0s - loss: 0.3133 - recall: 0.8529 - accuracy: 0.8636
    Epoch 126/1000
    7/7 - 0s - loss: 0.3303 - recall: 0.9216 - accuracy: 0.8838
    Epoch 127/1000
    7/7 - 0s - loss: 0.3386 - recall: 0.8627 - accuracy: 0.8586
    Epoch 128/1000
    7/7 - 0s - loss: 0.3390 - recall: 0.8529 - accuracy: 0.8434
    Epoch 129/1000
    7/7 - 0s - loss: 0.3445 - recall: 0.8725 - accuracy: 0.8434
    Epoch 130/1000
    7/7 - 0s - loss: 0.3312 - recall: 0.8725 - accuracy: 0.8636
    Epoch 131/1000
    7/7 - 0s - loss: 0.3407 - recall: 0.8922 - accuracy: 0.8788
    Epoch 132/1000
    7/7 - 0s - loss: 0.3413 - recall: 0.8627 - accuracy: 0.8737
    Epoch 133/1000
    7/7 - 0s - loss: 0.3559 - recall: 0.8431 - accuracy: 0.8434
    Epoch 134/1000
    7/7 - 0s - loss: 0.2967 - recall: 0.8725 - accuracy: 0.8687
    Epoch 135/1000
    7/7 - 0s - loss: 0.3513 - recall: 0.8529 - accuracy: 0.8485
    Epoch 136/1000
    7/7 - 0s - loss: 0.3491 - recall: 0.8824 - accuracy: 0.8737
    Epoch 137/1000
    7/7 - 0s - loss: 0.3464 - recall: 0.8529 - accuracy: 0.8737
    Epoch 138/1000
    7/7 - 0s - loss: 0.3239 - recall: 0.9020 - accuracy: 0.8990
    Epoch 139/1000
    7/7 - 0s - loss: 0.3180 - recall: 0.8824 - accuracy: 0.8838
    Epoch 140/1000
    7/7 - 0s - loss: 0.3300 - recall: 0.9020 - accuracy: 0.8889
    Epoch 141/1000
    7/7 - 0s - loss: 0.3155 - recall: 0.8922 - accuracy: 0.8788
    Epoch 142/1000
    7/7 - 0s - loss: 0.3387 - recall: 0.8824 - accuracy: 0.8535
    Epoch 143/1000
    7/7 - 0s - loss: 0.3306 - recall: 0.8725 - accuracy: 0.8586
    Epoch 144/1000
    7/7 - 0s - loss: 0.3273 - recall: 0.8725 - accuracy: 0.8687
    Epoch 145/1000
    7/7 - 0s - loss: 0.3103 - recall: 0.9118 - accuracy: 0.8990
    Epoch 146/1000
    7/7 - 0s - loss: 0.3058 - recall: 0.8922 - accuracy: 0.8838
    Epoch 147/1000
    7/7 - 0s - loss: 0.3078 - recall: 0.8922 - accuracy: 0.8788
    Epoch 148/1000
    7/7 - 0s - loss: 0.3422 - recall: 0.8725 - accuracy: 0.8535
    Epoch 149/1000
    7/7 - 0s - loss: 0.3387 - recall: 0.8431 - accuracy: 0.8384
    Epoch 150/1000
    7/7 - 0s - loss: 0.3261 - recall: 0.8627 - accuracy: 0.8636
    Epoch 151/1000
    7/7 - 0s - loss: 0.3058 - recall: 0.8922 - accuracy: 0.8838
    Epoch 152/1000
    7/7 - 0s - loss: 0.3078 - recall: 0.8922 - accuracy: 0.8737
    Epoch 153/1000
    7/7 - 0s - loss: 0.3215 - recall: 0.9020 - accuracy: 0.8838
    Epoch 154/1000
    7/7 - 0s - loss: 0.3422 - recall: 0.8431 - accuracy: 0.8434
    Epoch 155/1000
    7/7 - 0s - loss: 0.3256 - recall: 0.9020 - accuracy: 0.8737
    Epoch 156/1000
    7/7 - 0s - loss: 0.3618 - recall: 0.8627 - accuracy: 0.8182
    Epoch 157/1000
    7/7 - 0s - loss: 0.3223 - recall: 0.8824 - accuracy: 0.8939
    Epoch 158/1000
    7/7 - 0s - loss: 0.3414 - recall: 0.8627 - accuracy: 0.8687
    Epoch 159/1000
    7/7 - 0s - loss: 0.3187 - recall: 0.8725 - accuracy: 0.8737
    Epoch 160/1000
    7/7 - 0s - loss: 0.3308 - recall: 0.8725 - accuracy: 0.8737
    Epoch 161/1000
    7/7 - 0s - loss: 0.3230 - recall: 0.8824 - accuracy: 0.8636
    Epoch 162/1000
    7/7 - 0s - loss: 0.3188 - recall: 0.9020 - accuracy: 0.8838
    Epoch 163/1000
    7/7 - 0s - loss: 0.3138 - recall: 0.9216 - accuracy: 0.8889
    Epoch 164/1000
    7/7 - 0s - loss: 0.2992 - recall: 0.9020 - accuracy: 0.8838
    Epoch 165/1000
    7/7 - 0s - loss: 0.3136 - recall: 0.8922 - accuracy: 0.8788
    Epoch 166/1000
    7/7 - 0s - loss: 0.2936 - recall: 0.9020 - accuracy: 0.8838
    Epoch 167/1000
    7/7 - 0s - loss: 0.3487 - recall: 0.8922 - accuracy: 0.8737
    Epoch 168/1000
    7/7 - 0s - loss: 0.3208 - recall: 0.9118 - accuracy: 0.8737
    Epoch 169/1000
    7/7 - 0s - loss: 0.3117 - recall: 0.9118 - accuracy: 0.8687
    Epoch 170/1000
    7/7 - 0s - loss: 0.3344 - recall: 0.8725 - accuracy: 0.8586
    Epoch 171/1000
    7/7 - 0s - loss: 0.3349 - recall: 0.8627 - accuracy: 0.8535
    Epoch 172/1000
    7/7 - 0s - loss: 0.2893 - recall: 0.9020 - accuracy: 0.8939
    Epoch 173/1000
    7/7 - 0s - loss: 0.3066 - recall: 0.8529 - accuracy: 0.8636
    Epoch 174/1000
    7/7 - 0s - loss: 0.3107 - recall: 0.9118 - accuracy: 0.8939
    Epoch 175/1000
    7/7 - 0s - loss: 0.3233 - recall: 0.8922 - accuracy: 0.8889
    Epoch 176/1000
    7/7 - 0s - loss: 0.2887 - recall: 0.8824 - accuracy: 0.8838
    Epoch 177/1000
    7/7 - 0s - loss: 0.3043 - recall: 0.8922 - accuracy: 0.8838
    Epoch 178/1000
    7/7 - 0s - loss: 0.2961 - recall: 0.8922 - accuracy: 0.8838
    Epoch 179/1000
    7/7 - 0s - loss: 0.3112 - recall: 0.8824 - accuracy: 0.8788
    Epoch 180/1000
    7/7 - 0s - loss: 0.3075 - recall: 0.8922 - accuracy: 0.8788
    Epoch 181/1000
    7/7 - 0s - loss: 0.3033 - recall: 0.8922 - accuracy: 0.8788
    Epoch 182/1000
    7/7 - 0s - loss: 0.3232 - recall: 0.8725 - accuracy: 0.8737
    Epoch 183/1000
    7/7 - 0s - loss: 0.3094 - recall: 0.9020 - accuracy: 0.8939
    Epoch 184/1000
    7/7 - 0s - loss: 0.2774 - recall: 0.9216 - accuracy: 0.9040
    Epoch 185/1000
    7/7 - 0s - loss: 0.2801 - recall: 0.9020 - accuracy: 0.8889
    Epoch 186/1000
    7/7 - 0s - loss: 0.3090 - recall: 0.9216 - accuracy: 0.8838
    Epoch 187/1000
    7/7 - 0s - loss: 0.2949 - recall: 0.8824 - accuracy: 0.8838
    Epoch 188/1000
    7/7 - 0s - loss: 0.3224 - recall: 0.8824 - accuracy: 0.8737
    Epoch 189/1000
    7/7 - 0s - loss: 0.3018 - recall: 0.8725 - accuracy: 0.8636
    Epoch 190/1000
    7/7 - 0s - loss: 0.2859 - recall: 0.9020 - accuracy: 0.8889
    Epoch 191/1000
    7/7 - 0s - loss: 0.3022 - recall: 0.9020 - accuracy: 0.8889
    Epoch 192/1000
    7/7 - 0s - loss: 0.3017 - recall: 0.9118 - accuracy: 0.8889
    Epoch 193/1000
    7/7 - 0s - loss: 0.2994 - recall: 0.9020 - accuracy: 0.8838
    Epoch 194/1000
    7/7 - 0s - loss: 0.3146 - recall: 0.9118 - accuracy: 0.8889
    Epoch 195/1000
    7/7 - 0s - loss: 0.3075 - recall: 0.9020 - accuracy: 0.8687
    Epoch 196/1000
    7/7 - 0s - loss: 0.3156 - recall: 0.8725 - accuracy: 0.8737
    Epoch 197/1000
    7/7 - 0s - loss: 0.2940 - recall: 0.9020 - accuracy: 0.8889
    Epoch 198/1000
    7/7 - 0s - loss: 0.2903 - recall: 0.9118 - accuracy: 0.8889
    Epoch 199/1000
    7/7 - 0s - loss: 0.3245 - recall: 0.8824 - accuracy: 0.8838
    Epoch 200/1000
    7/7 - 0s - loss: 0.2895 - recall: 0.9118 - accuracy: 0.8889
    Epoch 201/1000
    7/7 - 0s - loss: 0.2916 - recall: 0.8725 - accuracy: 0.8788
    Epoch 202/1000
    7/7 - 0s - loss: 0.3397 - recall: 0.8725 - accuracy: 0.8586
    Epoch 203/1000
    7/7 - 0s - loss: 0.2932 - recall: 0.9020 - accuracy: 0.8889
    Epoch 204/1000
    7/7 - 0s - loss: 0.3206 - recall: 0.8725 - accuracy: 0.8737
    Epoch 205/1000
    7/7 - 0s - loss: 0.3045 - recall: 0.9118 - accuracy: 0.8788
    Epoch 206/1000
    7/7 - 0s - loss: 0.3158 - recall: 0.8725 - accuracy: 0.8737
    Epoch 207/1000
    7/7 - 0s - loss: 0.2937 - recall: 0.9118 - accuracy: 0.9040
    Epoch 208/1000
    7/7 - 0s - loss: 0.3065 - recall: 0.8725 - accuracy: 0.8636
    Epoch 209/1000
    7/7 - 0s - loss: 0.3121 - recall: 0.9020 - accuracy: 0.8838
    Epoch 210/1000
    7/7 - 0s - loss: 0.2746 - recall: 0.8922 - accuracy: 0.8838
    Epoch 211/1000
    7/7 - 0s - loss: 0.3053 - recall: 0.9020 - accuracy: 0.8737
    Epoch 212/1000
    7/7 - 0s - loss: 0.2992 - recall: 0.8922 - accuracy: 0.8788
    Epoch 213/1000
    7/7 - 0s - loss: 0.3081 - recall: 0.8627 - accuracy: 0.8636
    Epoch 214/1000
    7/7 - 0s - loss: 0.2706 - recall: 0.8922 - accuracy: 0.8990
    Epoch 215/1000
    7/7 - 0s - loss: 0.2885 - recall: 0.9020 - accuracy: 0.8838
    Epoch 216/1000
    7/7 - 0s - loss: 0.3071 - recall: 0.9216 - accuracy: 0.8788
    Epoch 217/1000
    7/7 - 0s - loss: 0.2844 - recall: 0.9020 - accuracy: 0.8939
    Epoch 218/1000
    7/7 - 0s - loss: 0.3035 - recall: 0.8922 - accuracy: 0.8889
    Epoch 219/1000
    7/7 - 0s - loss: 0.2925 - recall: 0.9412 - accuracy: 0.9040
    Epoch 220/1000
    7/7 - 0s - loss: 0.3023 - recall: 0.8824 - accuracy: 0.8737
    Epoch 221/1000
    7/7 - 0s - loss: 0.2814 - recall: 0.8922 - accuracy: 0.8838
    Epoch 222/1000
    7/7 - 0s - loss: 0.2913 - recall: 0.9020 - accuracy: 0.8737
    Epoch 223/1000
    7/7 - 0s - loss: 0.3009 - recall: 0.9020 - accuracy: 0.8889
    Epoch 224/1000
    7/7 - 0s - loss: 0.2910 - recall: 0.9118 - accuracy: 0.8889
    Epoch 225/1000
    7/7 - 0s - loss: 0.2894 - recall: 0.8824 - accuracy: 0.8788
    Epoch 226/1000
    7/7 - 0s - loss: 0.3055 - recall: 0.9020 - accuracy: 0.8889
    Epoch 227/1000
    7/7 - 0s - loss: 0.3220 - recall: 0.8627 - accuracy: 0.8485
    Epoch 228/1000
    7/7 - 0s - loss: 0.3039 - recall: 0.8627 - accuracy: 0.8737
    Epoch 229/1000
    7/7 - 0s - loss: 0.3070 - recall: 0.8627 - accuracy: 0.8788
    Epoch 230/1000
    7/7 - 0s - loss: 0.2775 - recall: 0.8922 - accuracy: 0.8788
    Epoch 231/1000
    7/7 - 0s - loss: 0.3082 - recall: 0.9020 - accuracy: 0.8889
    Epoch 232/1000
    7/7 - 0s - loss: 0.2701 - recall: 0.9216 - accuracy: 0.8939
    Epoch 233/1000
    7/7 - 0s - loss: 0.2814 - recall: 0.8431 - accuracy: 0.8788
    Epoch 234/1000
    7/7 - 0s - loss: 0.2976 - recall: 0.8922 - accuracy: 0.8737
    Epoch 235/1000
    7/7 - 0s - loss: 0.3023 - recall: 0.9216 - accuracy: 0.8939
    Epoch 236/1000
    7/7 - 0s - loss: 0.3140 - recall: 0.9020 - accuracy: 0.8838
    Epoch 237/1000
    7/7 - 0s - loss: 0.2799 - recall: 0.9020 - accuracy: 0.8889
    Epoch 238/1000
    7/7 - 0s - loss: 0.2953 - recall: 0.8824 - accuracy: 0.8636
    Epoch 239/1000
    7/7 - 0s - loss: 0.2938 - recall: 0.9118 - accuracy: 0.8889
    Epoch 240/1000
    7/7 - 0s - loss: 0.2954 - recall: 0.8824 - accuracy: 0.8889
    Epoch 241/1000
    7/7 - 0s - loss: 0.2701 - recall: 0.9118 - accuracy: 0.9091
    Epoch 242/1000
    7/7 - 0s - loss: 0.2999 - recall: 0.8922 - accuracy: 0.8939
    Epoch 243/1000
    7/7 - 0s - loss: 0.2901 - recall: 0.8922 - accuracy: 0.8687
    Epoch 244/1000
    7/7 - 0s - loss: 0.3058 - recall: 0.8725 - accuracy: 0.8737
    Epoch 245/1000
    7/7 - 0s - loss: 0.2906 - recall: 0.9020 - accuracy: 0.8889
    Epoch 246/1000
    7/7 - 0s - loss: 0.3166 - recall: 0.8627 - accuracy: 0.8586
    Epoch 247/1000
    7/7 - 0s - loss: 0.2826 - recall: 0.9118 - accuracy: 0.8990
    Epoch 248/1000
    7/7 - 0s - loss: 0.2961 - recall: 0.8725 - accuracy: 0.8788
    Epoch 249/1000
    7/7 - 0s - loss: 0.2966 - recall: 0.9216 - accuracy: 0.8889
    Epoch 250/1000
    7/7 - 0s - loss: 0.3098 - recall: 0.8725 - accuracy: 0.8687
    Epoch 251/1000
    7/7 - 0s - loss: 0.2984 - recall: 0.8922 - accuracy: 0.8838
    Epoch 252/1000
    7/7 - 0s - loss: 0.2683 - recall: 0.8824 - accuracy: 0.8838
    Epoch 253/1000
    7/7 - 0s - loss: 0.2810 - recall: 0.9118 - accuracy: 0.9091
    Epoch 254/1000
    7/7 - 0s - loss: 0.2968 - recall: 0.8922 - accuracy: 0.8838
    Epoch 255/1000
    7/7 - 0s - loss: 0.2709 - recall: 0.8824 - accuracy: 0.8838
    Epoch 256/1000
    7/7 - 0s - loss: 0.2757 - recall: 0.8824 - accuracy: 0.8838
    Epoch 257/1000
    7/7 - 0s - loss: 0.2833 - recall: 0.8922 - accuracy: 0.8939
    Epoch 258/1000
    7/7 - 0s - loss: 0.3039 - recall: 0.8627 - accuracy: 0.8737
    Epoch 259/1000
    7/7 - 0s - loss: 0.2884 - recall: 0.8824 - accuracy: 0.8838
    Epoch 260/1000
    7/7 - 0s - loss: 0.2900 - recall: 0.8824 - accuracy: 0.8687
    Epoch 261/1000
    7/7 - 0s - loss: 0.2621 - recall: 0.8824 - accuracy: 0.8939
    Epoch 262/1000
    7/7 - 0s - loss: 0.2843 - recall: 0.8922 - accuracy: 0.8889
    Epoch 263/1000
    7/7 - 0s - loss: 0.2934 - recall: 0.8922 - accuracy: 0.8788
    Epoch 264/1000
    7/7 - 0s - loss: 0.3022 - recall: 0.9118 - accuracy: 0.8939
    Epoch 265/1000
    7/7 - 0s - loss: 0.2575 - recall: 0.8922 - accuracy: 0.9091
    Epoch 266/1000
    7/7 - 0s - loss: 0.2863 - recall: 0.8824 - accuracy: 0.8939
    Epoch 267/1000
    7/7 - 0s - loss: 0.2723 - recall: 0.9118 - accuracy: 0.9091
    Epoch 268/1000
    7/7 - 0s - loss: 0.2814 - recall: 0.8824 - accuracy: 0.8889
    Epoch 269/1000
    7/7 - 0s - loss: 0.2839 - recall: 0.8824 - accuracy: 0.8636
    Epoch 270/1000
    7/7 - 0s - loss: 0.3105 - recall: 0.8922 - accuracy: 0.8838
    Epoch 271/1000
    7/7 - 0s - loss: 0.2974 - recall: 0.8725 - accuracy: 0.8939
    Epoch 272/1000
    7/7 - 0s - loss: 0.2902 - recall: 0.8725 - accuracy: 0.8788
    Epoch 273/1000
    7/7 - 0s - loss: 0.2532 - recall: 0.8725 - accuracy: 0.8889
    Epoch 274/1000
    7/7 - 0s - loss: 0.2729 - recall: 0.8922 - accuracy: 0.8990
    Epoch 275/1000
    7/7 - 0s - loss: 0.3025 - recall: 0.8922 - accuracy: 0.8939
    Epoch 276/1000
    7/7 - 0s - loss: 0.2701 - recall: 0.9216 - accuracy: 0.9091
    Epoch 277/1000
    7/7 - 0s - loss: 0.2824 - recall: 0.8824 - accuracy: 0.8838
    Epoch 278/1000
    7/7 - 0s - loss: 0.2609 - recall: 0.9314 - accuracy: 0.9192
    Epoch 279/1000
    7/7 - 0s - loss: 0.2887 - recall: 0.8922 - accuracy: 0.8939
    Epoch 280/1000
    7/7 - 0s - loss: 0.2510 - recall: 0.9020 - accuracy: 0.9040
    Epoch 281/1000
    7/7 - 0s - loss: 0.2558 - recall: 0.9216 - accuracy: 0.9141
    Epoch 282/1000
    7/7 - 0s - loss: 0.2635 - recall: 0.9020 - accuracy: 0.9040
    Epoch 283/1000
    7/7 - 0s - loss: 0.2895 - recall: 0.9020 - accuracy: 0.8939
    Epoch 284/1000
    7/7 - 0s - loss: 0.2569 - recall: 0.9020 - accuracy: 0.8889
    Epoch 285/1000
    7/7 - 0s - loss: 0.2644 - recall: 0.9118 - accuracy: 0.8990
    Epoch 286/1000
    7/7 - 0s - loss: 0.2661 - recall: 0.9020 - accuracy: 0.8939
    Epoch 287/1000
    7/7 - 0s - loss: 0.2803 - recall: 0.8725 - accuracy: 0.8838
    Epoch 288/1000
    7/7 - 0s - loss: 0.2784 - recall: 0.8824 - accuracy: 0.8889
    Epoch 289/1000
    7/7 - 0s - loss: 0.2580 - recall: 0.9314 - accuracy: 0.9293
    Epoch 290/1000
    7/7 - 0s - loss: 0.2791 - recall: 0.8725 - accuracy: 0.8687
    Epoch 291/1000
    7/7 - 0s - loss: 0.2755 - recall: 0.9216 - accuracy: 0.9141
    Epoch 292/1000
    7/7 - 0s - loss: 0.2964 - recall: 0.8725 - accuracy: 0.8889
    Epoch 293/1000
    7/7 - 0s - loss: 0.2718 - recall: 0.8824 - accuracy: 0.8889
    Epoch 294/1000
    7/7 - 0s - loss: 0.2874 - recall: 0.8922 - accuracy: 0.8889
    Epoch 295/1000
    7/7 - 0s - loss: 0.2734 - recall: 0.9510 - accuracy: 0.9242
    Epoch 296/1000
    7/7 - 0s - loss: 0.3009 - recall: 0.9020 - accuracy: 0.8737
    Epoch 297/1000
    7/7 - 0s - loss: 0.2959 - recall: 0.8922 - accuracy: 0.8889
    Epoch 298/1000
    7/7 - 0s - loss: 0.2713 - recall: 0.9020 - accuracy: 0.8990
    Epoch 299/1000
    7/7 - 0s - loss: 0.2875 - recall: 0.9020 - accuracy: 0.8939
    Epoch 300/1000
    7/7 - 0s - loss: 0.2645 - recall: 0.8824 - accuracy: 0.8838
    Epoch 301/1000
    7/7 - 0s - loss: 0.2816 - recall: 0.8922 - accuracy: 0.8788
    Epoch 302/1000
    7/7 - 0s - loss: 0.2879 - recall: 0.9020 - accuracy: 0.8889
    Epoch 303/1000
    7/7 - 0s - loss: 0.2626 - recall: 0.9216 - accuracy: 0.9192
    Epoch 304/1000
    7/7 - 0s - loss: 0.2767 - recall: 0.8922 - accuracy: 0.8990
    Epoch 305/1000
    7/7 - 0s - loss: 0.2896 - recall: 0.8922 - accuracy: 0.8939
    Epoch 306/1000
    7/7 - 0s - loss: 0.2529 - recall: 0.8922 - accuracy: 0.8939
    Epoch 307/1000
    7/7 - 0s - loss: 0.2628 - recall: 0.9314 - accuracy: 0.9040
    Epoch 308/1000
    7/7 - 0s - loss: 0.2760 - recall: 0.9118 - accuracy: 0.9091
    Epoch 309/1000
    7/7 - 0s - loss: 0.2593 - recall: 0.9020 - accuracy: 0.8990
    Epoch 310/1000
    7/7 - 0s - loss: 0.2663 - recall: 0.9020 - accuracy: 0.8990
    Epoch 311/1000
    7/7 - 0s - loss: 0.3037 - recall: 0.9118 - accuracy: 0.8939
    Epoch 312/1000
    7/7 - 0s - loss: 0.2539 - recall: 0.9118 - accuracy: 0.8990
    Epoch 313/1000
    7/7 - 0s - loss: 0.2908 - recall: 0.8824 - accuracy: 0.8687
    Epoch 314/1000
    7/7 - 0s - loss: 0.2591 - recall: 0.9314 - accuracy: 0.9040
    Epoch 315/1000
    7/7 - 0s - loss: 0.2635 - recall: 0.9314 - accuracy: 0.9040
    Epoch 316/1000
    7/7 - 0s - loss: 0.2422 - recall: 0.9118 - accuracy: 0.9242
    Epoch 317/1000
    7/7 - 0s - loss: 0.3040 - recall: 0.9020 - accuracy: 0.8838
    Epoch 318/1000
    7/7 - 0s - loss: 0.2710 - recall: 0.8725 - accuracy: 0.8889
    Epoch 319/1000
    7/7 - 0s - loss: 0.2608 - recall: 0.9216 - accuracy: 0.8990
    Epoch 320/1000
    7/7 - 0s - loss: 0.2805 - recall: 0.9118 - accuracy: 0.9141
    Epoch 321/1000
    7/7 - 0s - loss: 0.2768 - recall: 0.9118 - accuracy: 0.8990
    Epoch 322/1000
    7/7 - 0s - loss: 0.2651 - recall: 0.9020 - accuracy: 0.8939
    Epoch 323/1000
    7/7 - 0s - loss: 0.2927 - recall: 0.9118 - accuracy: 0.8990
    Epoch 324/1000
    7/7 - 0s - loss: 0.2761 - recall: 0.8824 - accuracy: 0.8939
    Epoch 325/1000
    7/7 - 0s - loss: 0.2736 - recall: 0.8627 - accuracy: 0.8939
    Epoch 326/1000
    7/7 - 0s - loss: 0.2601 - recall: 0.9118 - accuracy: 0.9040
    Epoch 327/1000
    7/7 - 0s - loss: 0.2980 - recall: 0.8922 - accuracy: 0.8889
    Epoch 328/1000
    7/7 - 0s - loss: 0.2505 - recall: 0.9020 - accuracy: 0.9040
    Epoch 329/1000
    7/7 - 0s - loss: 0.2644 - recall: 0.9216 - accuracy: 0.8990
    Epoch 330/1000
    7/7 - 0s - loss: 0.2584 - recall: 0.9510 - accuracy: 0.9192
    Epoch 331/1000
    7/7 - 0s - loss: 0.2610 - recall: 0.9118 - accuracy: 0.8889
    Epoch 332/1000
    7/7 - 0s - loss: 0.2791 - recall: 0.9020 - accuracy: 0.8939
    Epoch 333/1000
    7/7 - 0s - loss: 0.2332 - recall: 0.9216 - accuracy: 0.9141
    Epoch 334/1000
    7/7 - 0s - loss: 0.2605 - recall: 0.9216 - accuracy: 0.9091
    Epoch 335/1000
    7/7 - 0s - loss: 0.2880 - recall: 0.8824 - accuracy: 0.8838
    Epoch 336/1000
    7/7 - 0s - loss: 0.2577 - recall: 0.9020 - accuracy: 0.8939
    Epoch 337/1000
    7/7 - 0s - loss: 0.2348 - recall: 0.9314 - accuracy: 0.9091
    Epoch 338/1000
    7/7 - 0s - loss: 0.2400 - recall: 0.9412 - accuracy: 0.9242
    Epoch 339/1000
    7/7 - 0s - loss: 0.2692 - recall: 0.9216 - accuracy: 0.9192
    Epoch 340/1000
    7/7 - 0s - loss: 0.2677 - recall: 0.9118 - accuracy: 0.9040
    Epoch 341/1000
    7/7 - 0s - loss: 0.2805 - recall: 0.9020 - accuracy: 0.8838
    Epoch 342/1000
    7/7 - 0s - loss: 0.2685 - recall: 0.9118 - accuracy: 0.9091
    Epoch 343/1000
    7/7 - 0s - loss: 0.2584 - recall: 0.9020 - accuracy: 0.8990
    Epoch 344/1000
    7/7 - 0s - loss: 0.2522 - recall: 0.9118 - accuracy: 0.8939
    Epoch 345/1000
    7/7 - 0s - loss: 0.2792 - recall: 0.8922 - accuracy: 0.8788
    Epoch 346/1000
    7/7 - 0s - loss: 0.2578 - recall: 0.9216 - accuracy: 0.9091
    Epoch 347/1000
    7/7 - 0s - loss: 0.2628 - recall: 0.9216 - accuracy: 0.8838
    Epoch 348/1000
    7/7 - 0s - loss: 0.2973 - recall: 0.8725 - accuracy: 0.8838
    Epoch 349/1000
    7/7 - 0s - loss: 0.2703 - recall: 0.8824 - accuracy: 0.8889
    Epoch 350/1000
    7/7 - 0s - loss: 0.2553 - recall: 0.9216 - accuracy: 0.9141
    Epoch 351/1000
    7/7 - 0s - loss: 0.2617 - recall: 0.9020 - accuracy: 0.9040
    Epoch 352/1000
    7/7 - 0s - loss: 0.2533 - recall: 0.9314 - accuracy: 0.9141
    Epoch 353/1000
    7/7 - 0s - loss: 0.2586 - recall: 0.9020 - accuracy: 0.9141
    Epoch 354/1000
    7/7 - 0s - loss: 0.2520 - recall: 0.8824 - accuracy: 0.8990
    Epoch 355/1000
    7/7 - 0s - loss: 0.2570 - recall: 0.9020 - accuracy: 0.9091
    Epoch 356/1000
    7/7 - 0s - loss: 0.2752 - recall: 0.8824 - accuracy: 0.8939
    Epoch 357/1000
    7/7 - 0s - loss: 0.2610 - recall: 0.8725 - accuracy: 0.8838
    Epoch 358/1000
    7/7 - 0s - loss: 0.2678 - recall: 0.8824 - accuracy: 0.8889
    Epoch 359/1000
    7/7 - 0s - loss: 0.2471 - recall: 0.9118 - accuracy: 0.8939
    Epoch 360/1000
    7/7 - 0s - loss: 0.2653 - recall: 0.9020 - accuracy: 0.9091
    Epoch 361/1000
    7/7 - 0s - loss: 0.2802 - recall: 0.8824 - accuracy: 0.8889
    Epoch 362/1000
    7/7 - 0s - loss: 0.2486 - recall: 0.9216 - accuracy: 0.9091
    Epoch 363/1000
    7/7 - 0s - loss: 0.2767 - recall: 0.8725 - accuracy: 0.8939
    Epoch 364/1000
    7/7 - 0s - loss: 0.2776 - recall: 0.8725 - accuracy: 0.8939
    Epoch 365/1000
    7/7 - 0s - loss: 0.2364 - recall: 0.9510 - accuracy: 0.9343
    Epoch 366/1000
    7/7 - 0s - loss: 0.2532 - recall: 0.8725 - accuracy: 0.8939
    Epoch 367/1000
    7/7 - 0s - loss: 0.2433 - recall: 0.9020 - accuracy: 0.9141
    Epoch 368/1000
    7/7 - 0s - loss: 0.2532 - recall: 0.9118 - accuracy: 0.9040
    Epoch 369/1000
    7/7 - 0s - loss: 0.2567 - recall: 0.9216 - accuracy: 0.9091
    Epoch 370/1000
    7/7 - 0s - loss: 0.2669 - recall: 0.8824 - accuracy: 0.8838
    Epoch 371/1000
    7/7 - 0s - loss: 0.2487 - recall: 0.8922 - accuracy: 0.9040
    Epoch 372/1000
    7/7 - 0s - loss: 0.2543 - recall: 0.9020 - accuracy: 0.8889
    Epoch 373/1000
    7/7 - 0s - loss: 0.2353 - recall: 0.9118 - accuracy: 0.9141
    Epoch 374/1000
    7/7 - 0s - loss: 0.2597 - recall: 0.9314 - accuracy: 0.9242
    Epoch 375/1000
    7/7 - 0s - loss: 0.2648 - recall: 0.8922 - accuracy: 0.9040
    Epoch 376/1000
    7/7 - 0s - loss: 0.2755 - recall: 0.8922 - accuracy: 0.8990
    Epoch 377/1000
    7/7 - 0s - loss: 0.2385 - recall: 0.9216 - accuracy: 0.9141
    Epoch 378/1000
    7/7 - 0s - loss: 0.2586 - recall: 0.9216 - accuracy: 0.9040
    Epoch 379/1000
    7/7 - 0s - loss: 0.2594 - recall: 0.8922 - accuracy: 0.9091
    Epoch 380/1000
    7/7 - 0s - loss: 0.2605 - recall: 0.9118 - accuracy: 0.8990
    Epoch 381/1000
    7/7 - 0s - loss: 0.2763 - recall: 0.9118 - accuracy: 0.9141
    Epoch 382/1000
    7/7 - 0s - loss: 0.2415 - recall: 0.9118 - accuracy: 0.8939
    Epoch 383/1000
    7/7 - 0s - loss: 0.2898 - recall: 0.9118 - accuracy: 0.9040

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

**Evaluate the model**

</div>

<div class="cell code" data-execution_count="15" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
eval_results = model.evaluate(X_test.astype(np.float32), y_test.astype(np.float32), verbose=2)

loss = history.history['loss']
rec = history.history['recall']
acc = history.history['accuracy']

epochs = range(1, len(loss) + 1)

fig_SI = plt.figure()

ax1_SI = fig_SI.add_subplot(111)

ax1_SI.plot(epochs, loss, 'g.', label='Training loss')
ax1_SI.plot(epochs, rec, 'b.', label='recall')
ax1_SI.plot(epochs, acc, 'r.', label='accuracy')

fig_SI.suptitle('Training loss and accuracy')
ax1_SI.set_xlabel('Epochs')
ax1_SI.legend()
fig_SI.show()
```

<div class="output stream stdout">

    4/4 - 0s - loss: 0.3053 - recall: 0.8966 - accuracy: 0.8776

</div>

<div class="output stream stderr">

    Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.

</div>

<div class="output display_data">

![](2b8db79146512bbfec8f6ed97402bdb280022fcf.png)

</div>

</div>

<div class="cell code" data-execution_count="16" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
prediction = model.predict(X_test.astype(np.float32))
rounded_prediction = tf.math.round(prediction)
# create a confusion matrix
conf_matrix = confusion_matrix(y_true=y_test.astype(np.float32), y_pred=rounded_prediction)
keras_name = 'neural_net'
tpr = visualize_confusion(conf_matrix,keras_name)

print('The accuracy is: ' +'{:.1%}'.format(eval_results[2]))
print('The true positive rate is: ' +'{:.1%}'.format(tpr))
```

<div class="output stream stderr">

    FixedFormatter should only be used together with FixedLocator

</div>

<div class="output display_data">

![](d74b5f951ba35a41fab360fa1c786590611ae2f5.png)

</div>

<div class="output stream stdout">

    The accuracy is: 87.8%
    The true positive rate is: 89.7%

</div>

</div>

<div class="cell code" data-execution_count="36" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
#find indices with wrong predcition
heart_disease_predicion_wrong = [] #y_true = 1; pred = 0
no_heart_disease_predicion_wrong = [] #y_true = 0; pred = 1

heart_disease_X_test = []
no_heart_disease_X_test = []

for i in range(len(X_test)):

    tmp_pred = rounded_prediction[i].numpy()
    prediction_for_test = int(tmp_pred)
    y_test_for_test = y_test.iloc[i]
    if 0 == y_test_for_test:
        if y_test_for_test != prediction_for_test:
            no_heart_disease_predicion_wrong.append(i)
        else:
            no_heart_disease_X_test.append(i)
    elif 1 == y_test_for_test:
        if y_test_for_test != prediction_for_test:
            heart_disease_predicion_wrong.append(i)
        else:
            heart_disease_X_test.append(i)

print(heart_disease_predicion_wrong)
print(no_heart_disease_predicion_wrong)
```

<div class="output stream stdout">

    [6, 10, 23, 75, 88, 89]
    [4, 17, 21, 42, 86, 96]

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

**Visualize the features with biggest impact on the output**

For visualization the [shap](https://github.com/slundberg/shap) package
is used.

</div>

<div class="cell code" data-execution_count="33" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
shap.initjs()

col_names = list(X_train)
tmp = X_train.astype(np.float32)
X_train_flatten = tmp.to_numpy()

tmp_2 = X_test.astype(np.float32)
X_test_flatten = tmp_2.to_numpy()

e = shap.DeepExplainer(model, X_train_flatten)
shap_values = e.shap_values(X_test_flatten)
shap.summary_plot(shap_values, X_test_flatten, feature_names=col_names)
```

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output stream stdout">

    WARNING:tensorflow:From c:\users\ramjoue\pycharmprojects\kessel_ki_repo\venv\lib\site-packages\shap\explainers\_deep\deep_tf.py:239: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.

</div>

<div class="output stream stderr">

    keras is no longer supported, please use tf.keras instead.

</div>

<div class="output display_data">

![](854b736ffe4c65888ceb9ed414327661a6a452e4.png)

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

This chart shows the average impact of the dataset features on the model
output. It's visible that the feature \_major\_\_vessels\_*0.0* has the
biggest impact on the model output. The value 0 means that no major
vessels are visible on the flourosopy picture. It's a significant
feature of a heart disease. On the other hand a large number of major
vessels like the feature \_major\_\_vessels\_*3.0* is an indicator of a
low heart disease probability.

It's also visible that the feature chest pain (as typical angina and non
anginal pain) has an big impact on the output and also the sex.

The drawback of this kind of visualization is the missing information,
if the feature impacts the output positively or negatively.

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

#### Heart disease datapoints

To get a feeling which features have a positive or negative impact on
the output, a ***force plot*** is used. The next three plots are
datapoints that are initially labeled as *heart disease* and were
correct predicted.

</div>

<div class="cell code" data-execution_count="40" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
expected_value= tf.keras.backend.get_value(e.expected_value[0])
```

</div>

<div class="cell code" data-execution_count="51" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
index = heart_disease_X_test[0]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="51">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c8b8d4e0>

</div>

</div>

<div class="cell code" data-execution_count="52" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python

index = heart_disease_X_test[1]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="52">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c8b81208>

</div>

</div>

<div class="cell code" data-execution_count="53" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
index = heart_disease_X_test[2]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="53">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c8b704a8>

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

#### No heart disease datapoints

On the other hand it's interesting to analyze which features are most
important for datapoints that where initially labeled as *no heart
disease* and correct predicted.

</div>

<div class="cell code" data-execution_count="58" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python

index = no_heart_disease_X_test[0]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="58">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c9baa710>

</div>

</div>

<div class="cell code" data-execution_count="60" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python

index = no_heart_disease_X_test[1]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="60">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c9ba4400>

</div>

</div>

<div class="cell code" data-execution_count="61" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
index = no_heart_disease_X_test[2]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="61">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c8b82a90>

</div>

</div>

<div class="cell markdown" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%% md\n&quot;}">

#### Heart disease datapoints incorrectly predicted as no heart disease

</div>

<div class="cell code" data-execution_count="62" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%&quot;}">

``` python
# predicted to be not a heart disease but true label is 1
index = heart_disease_predicion_wrong[0] #y_true = 1; pred = 0[2]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="62">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c8b86cf8>

</div>

</div>

<div class="cell code" data-execution_count="64" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
# predicted to be not a heart disease but true label is 1
index = heart_disease_predicion_wrong[1] #y_true = 1; pred = 0[2]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="64">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c8b82eb8>

</div>

</div>

<div class="cell code" data-execution_count="65" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
# predicted to be not a heart disease but true label is 1
index = heart_disease_predicion_wrong[2] #y_true = 1; pred = 0[2]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="65">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c9bbbdd8>

</div>

</div>

<div class="cell code" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
#### No Heart disease datapoints incorrectly predicted as heart disease
```

</div>

<div class="cell code" data-execution_count="63" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
# predicted to be a heart disease but true label is 0
index = no_heart_disease_predicion_wrong[0]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="63">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c9ba4470>

</div>

</div>

<div class="cell code" data-execution_count="66" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
# predicted to be a heart disease but true label is 0
index = no_heart_disease_predicion_wrong[1]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="66">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c9baa1d0>

</div>

</div>

<div class="cell code" data-execution_count="67" data-collapsed="false" data-pycharm="{&quot;name&quot;:&quot;#%%\n&quot;}">

``` python
# predicted to be a heart disease but true label is 0
index = no_heart_disease_predicion_wrong[2]
shap.force_plot(expected_value, shap_values[0][index, :], X_test.iloc[index, :])
```

<div class="output execute_result" data-execution_count="67">

    <shap.plots._force.AdditiveForceVisualizer at 0x199c9baa940>

</div>

</div>
