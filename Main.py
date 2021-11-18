import tensorflow as tf
import pandas as pd
# skąd to jest????
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# https://www.youtube.com/watch?v=6_2hzRopPbQ
print('hello')
# not sure what TensorFlow is installed.
# Probably 2.7.0, but Python Interpreter shows 2.3.0 with option to upgrade to 2.6.0
tensor1 = tf.ones([1,2,3])
print(tf.__version__) # shows 2.3.0
df = pd.read_csv('F:/churn.csv')
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis= 1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2)
X_train.head()
y_train.head()

# build model:
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# compile model:
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

# fit:
model.fit(X_train, y_train, epochs=10, batch_size=32)

# prediction
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

fdsf = accuracy_score(y_test, y_hat)

print(type(fdsf)) # <class 'numpy.float64'>
model.save('tfmodel')


del model
model = load_model('tfmodel');
