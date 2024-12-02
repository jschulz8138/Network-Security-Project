
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from tensorflow import keras, saved_model
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, callbacks
from sklearn.metrics import accuracy_score

#Set this to 1 if you want to train a model, 0 if you want to load a trained model
TRAIN_MODEL = 0


def train_model():
    df = pd.read_csv("ACI-IoT-2023.csv")
    df.drop(["Flow Bytes/s","Timestamp","Flow Packets/s"],axis=1,inplace=True)
    df.head()
    df.Label.value_counts()
    features = df.columns.tolist()
    features.remove("Label")
    features.remove("Flow ID")
    features.remove("Src IP")
    features.remove("Dst IP")

    X = df[features]
    y= df["Label"]
    X = pd.get_dummies(X, dtype=int)
    y = pd.get_dummies(y,dtype=int)
    X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, stratify=y, random_state = 42)

    index_bigger = int(y_train.shape[0] * 0.7)

    X_val = X_train.iloc[index_bigger:,]
    X_train = X_train.iloc[:index_bigger,]
    y_val = y_train.iloc[index_bigger:]
    y_train = y_train.iloc[:index_bigger]
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

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=2, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    history= model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=10,
        epochs=1,
        callbacks =[early_stopping],
        #Turn on or off for debugging, displays training
        #verbose=0
    )

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss']].plot();
    print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

    model.save('my_model.h5') 


def predict():
    model = load_model('my_model.h5')


    y_preds = model.predict(X_test)
    y_preds.shape
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


if __name__ == "__main__":
    predict()