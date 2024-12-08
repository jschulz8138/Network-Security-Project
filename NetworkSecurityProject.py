import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from tensorflow import keras, saved_model
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, callbacks
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Set this to 1 if you want to train a model, 0 if you want to load a trained model
TRAIN_MODEL = 0

# Load and process data.
df = pd.read_csv("ACI-IoT-2023.csv")
df.drop(["Flow Bytes/s","Timestamp","Flow Packets/s"],axis=1,inplace=True)

# Display dataset details.
df.head()
df.Label.value_counts()

# Extract features and labels.
features = df.columns.tolist()
features.remove("Label")
features.remove("Flow ID")
features.remove("Src IP")
features.remove("Dst IP")

X = df[features]
y= df["Label"]
X = pd.get_dummies(X, dtype=int)
y = pd.get_dummies(y,dtype=int)

# Split data between training, validation and testing.
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, stratify=y, random_state = 42)

index_bigger = int(y_train.shape[0] * 0.7)

X_val = X_train.iloc[index_bigger:,]
X_train = X_train.iloc[:index_bigger,]
y_val = y_train.iloc[index_bigger:]
y_train = y_train.iloc[:index_bigger]

# Train or load model from file save
if(TRAIN_MODEL):
    num_classes = 12  

    model = keras.Sequential([
        layers.BatchNormalization(),
        layers.Dense(units=64, activation='relu'),
        layers.BatchNormalization(), 
        layers.Dropout(0.5), 
        layers.Dense(units=32, activation='relu'),
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
        patience=4, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    history= model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=256,
        epochs=64,
        callbacks =[early_stopping],
        #Turn on or off for debugging, displays training
        #verbose=0
    )

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss']].plot();
    print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

    model.save('my_model.h5') 

else:
    model = load_model('my_model.h5')

# Evaluate model on test set
y_preds = model.predict(X_test)
y_preds.shape
threshold = 0.5
binary_preds = np.where(y_preds >= threshold, 1, 0)

accuracy = accuracy_score(y_test, binary_preds)
# Propotion of correctly predicted positives out of all positives (true or false)
precision = precision_score(y_test, binary_preds, average='micro')
# Propotion of correctly predicted positives out of all ACTUAL positives (true positives & false negatives)
recall = recall_score(y_test, binary_preds, average='micro')
# F1 balances both precision and recall (penalizes imbalance beween precision and recall)
f1 = f1_score(y_test, binary_preds, average='micro')
hamming_loss_val = hamming_loss(y_test, binary_preds)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Hamming loss:", hamming_loss_val)



#############################################
# Experiment 1: Flood Types Attack Evaluation
# Evaluates performance of model in detecting
# specifically flood attack types
#############################################
print("\nFlood Types Attack Evaluation:")

# Defined list of flood attack types
flood_types = ['ARP Spoofing', 'DNS Flood', 'ICMP Flood', 'SYN Flood', 'UDP Flood']
# Get attack types (labels) to evaluate only from flood type attacks
attack_types = [attack for attack in y.columns.tolist() if attack in flood_types]

# Evaluate for each included flood attack type
for attack in attack_types:
    # Filter for specific attack types
    attack_indices = y_test[attack] == 1
    X_test_attack = X_test[attack_indices]
    y_test_attack = y_test[attack_indices]

    if len(X_test_attack) > 0:
        # Make predictions on specific attack type
        y_preds_attack = model.predict(X_test_attack)
        # Convert to binary
        binary_preds_attack = np.where(y_preds_attack >= threshold, 1, 0)

        # Calculate statistics for each specific flood attack type
        attack_accuracy = accuracy_score(y_test_attack, binary_preds_attack)
        attack_precision = precision_score(y_test_attack, binary_preds_attack, average='micro', zero_division=1)
        attack_recall = recall_score(y_test_attack, binary_preds_attack, average='micro', zero_division=1)
        attack_f1 = f1_score(y_test_attack, binary_preds_attack, average='micro', zero_division=1)

        # Print calculated statistics
        print(f"Metrics for {attack}:")
        print(f"  Accuracy: {attack_accuracy:.4f}")
        print(f"  Precision: {attack_precision:.4f}")
        print(f"  Recall: {attack_recall:.4f}")
        print(f"  F1-score: {attack_f1:.4f}")
    else:
        print(f"No test samples found for attack type: {attack}")


##############################################
# Experiment 2: Threshold Sensitivity Analysis
# Experiment helps find optimal threshold for
# better F1 score
##############################################
print("\nThreshold Sensitivity Analysis:")

# Defined list of flood attack types
flood_types = ['ARP Spoofing', 'DNS Flood', 'ICMP Flood', 'SYN Flood', 'UDP Flood']

# Threshold values to evaluate (from 0.1 to 0.9 in 0.1 increments)
threshold_values = np.arange(0.1, 1.0, 0.1)
# Directory to store results per attack types, containing a list of
# threshold and F1-score pairs for each entry
threshold_sensitivity_results = {attack: [] for attack in flood_types}

# Loop through diffrent thresholds to analyze performance
for threshold in threshold_values:
    print(f"\nEvaluating for threshold: {threshold:.1f}")
    for attack in flood_types:
        if attack in y.columns:
            # Filter for specific attack types
            attack_indices = y_test[attack] == 1
            X_test_attack = X_test[attack_indices]
            y_test_attack = y_test[attack_indices]

            if len(X_test_attack) > 0:
                # Make predictions on specific attack type
                y_preds_attack = model.predict(X_test_attack)
                # Convert to binary labels at current threshold
                binary_preds_attack = np.where(y_preds_attack >= threshold, 1, 0)

                # Calculate f1-score for specific attack type
                attack_f1 = f1_score(y_test_attack, binary_preds_attack, average='micro', zero_division=1)
                # Store threshold, f1-score pair in results
                threshold_sensitivity_results[attack].append((threshold, attack_f1))
            else:
                print(f"No test samples found for attack type: {attack}")

# Plot the results of sensitivity analysis
plt.figure(figsize=(10, 6))
for attack, results in threshold_sensitivity_results.items():
    thresholds, f1_scores = zip(*results)   #unpack threshold f1-score pair 
    plt.plot(thresholds, f1_scores, label=f"{attack}") #plot for attack type

# Add tiyle and labels for visualization
plt.title("Threshold Sensitivity Analysis (F1-Score)")
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.legend()
plt.grid()
plt.show()