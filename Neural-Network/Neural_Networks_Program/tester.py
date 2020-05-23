import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load training dataset.
data_frame = read_csv("training_data", delim_whitespace=True, header=None)
dataset = data_frame.values
training_data = dataset[:, :13]
labels = dataset[:, 13]

data_frame = read_csv("prediction_data", delim_whitespace=True, header=None)
testing_data = data_frame.values

# create model
def network_model():

    model = Sequential()
    model.add(Dense(500, input_dim=13, kernel_initializer='normal', activation='relu'))
    '''
    model.add(Dense(13, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(13, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(13, kernel_initializer='normal', activation='sigmoid'))
    '''
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


# Pre-process with StandardScaler, create the model with KerasClassifier and train using .fit.
# Then, predict using .predict on trained model

estimators = [
    ('standardize', StandardScaler()),
    ('mlp', KerasRegressor(build_fn=network_model,epochs=1000,batch_size=128,verbose=0))
]

pipeline = Pipeline(estimators)

pipeline.fit(training_data, labels)
testp = pipeline.predict(training_data)


for i in range(len(testp)):
    print(f'{testp[i]}, {labels[i]}')

print(pipeline.predict(testing_data))
