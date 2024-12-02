
import numpy as np
import pandas as pd 
#from tensorflow import keras
#from tensorflow.keras import layers
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras import callbacks
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.metrics import make_scorer, f1_score
# from sklearn.model_selection import cross_val_score
# import shutil
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
# from keras.models import load_model
# from sklearn.feature_selection import RFE
# from sklearn.metrics import classification_report
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from scipy.stats import randint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.metrics import accuracy_score

print("starting reading")

df = pd.read_csv("ACI-IoT-2023.csv")


print("dropping values")

df.drop(["Flow Bytes/s","Timestamp","Flow Packets/s"],axis=1,inplace=True)

df.head()

df.Label.value_counts()

print("cleaning features")

features = df.columns.tolist()

features.remove("Label")
features.remove("Flow ID")
features.remove("Src IP")
features.remove("Dst IP")

#print(df)

print("project 1")

X = df[features]
y= df["Label"]
X = pd.get_dummies(X, dtype=int)
y = pd.get_dummies(y,dtype=int)
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, stratify=y, random_state = 42)

index_bigger = int(y_train.shape[0] * 0.7)

print("project 2")

X_val = X_train.iloc[index_bigger:,]
X_train = X_train.iloc[:index_bigger,]
y_val = y_train.iloc[index_bigger:]
y_train = y_train.iloc[:index_bigger]


print("project 3")

num_classes = 12  

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(units=16, activation='relu'),
    layers.BatchNormalization(), 
    layers.Dropout(0.5), 
    layers.Dense(units=4, activation='relu'),
    layers.BatchNormalization(),  
    layers.Dropout(0.5), 
    layers.Dense(units=num_classes, activation='softmax')  
])


print("project 4")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)


print("project 5")
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=2, # how many epochs to wait before stopping
    restore_best_weights=True,
)

print("project 6")
history= model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=10,
    epochs=1,
    callbacks =[early_stopping],
    # put your callbacks in a list
    #verbose=0 # turn off training log
)

print("project 7")
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

y_preds = model.predict(X_test)

y_preds.shape

print("project 8")

threshold = 0.5
binary_preds = np.where(y_preds >= threshold, 1, 0)

accuracy = accuracy_score(y_test, binary_preds)
precision = precision_score(y_test, binary_preds, average='micro')
recall = recall_score(y_test, binary_preds, average='micro')
f1 = f1_score(y_test, binary_preds, average='micro')
hamming_loss_val = hamming_loss(y_test, binary_preds)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Hamming loss:", hamming_loss_val)


print("saving model?")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.weights.h5")
print("Saved model to disk")

print("finished")
 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")








#pd.set_option('display.max_columns', None)
#df.describe(include='all')
#df.drop(["Flow Bytes/s","Timestamp","Flow Packets/s"],axis=1,inplace=True)
#df.head()

# df.Label.value_counts()
# features = df.columns.tolist()
# features.remove("Label")
# features.remove("Flow ID")
# features.remove("Src IP")
# features.remove("Dst IP")
# X = df[features]
# y= df["Label"]

# X = pd.get_dummies(X, dtype=int)
# encoder = LabelEncoder()
# y = encoder.fit_transform(y)
# y = pd.Series(y)


