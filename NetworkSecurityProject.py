
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


df = pd.read_csv("ACI-IoT-2023.csv")


print(df)

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


