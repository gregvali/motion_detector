import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler

# Read the CSV
df = pd.read_csv('new_motion_data.csv')

print(df)

# If your timestamp column is named 'timestamp'
timestamps = pd.to_datetime(df['timestamp'], format='%H:%M:%S %d/%m/%Y')


# # Convert timestamps to datetime
# timestamps = pd.to_datetime(timestamps)
# # Plot events as vertical lines
# plt.figure(figsize=(12, 4))
# plt.plot(timestamps, [1]*len(timestamps), '|', markersize=20)
# plt.ylabel('Motion Events')
# plt.xlabel('Time')
# plt.title('Motion Detection Timeline')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



print(f"Total events: {len(timestamps)}")
print(f"Date range: {timestamps.min()} to {timestamps.max()}")

# # See what times of day you get most motion
# hour_counts = timestamps.dt.hour.value_counts().sort_index()
# hour_counts.plot(kind='bar')
# plt.title('Motion Events by Hour of Day')
# plt.xlabel('Hour (24-hour format)')
# plt.ylabel('Number of Events')
# plt.show()

#
# Create Time-Based Features
#

# Assuming you have your timestamps already parsed
df = pd.DataFrame({'timestamp': timestamps})

df.to_csv('output_data.csv', index=False)

# Extract basic time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Time since last event (in minutes)
df['time_since_last'] = df['timestamp'].diff().dt.total_seconds() / 60

# Remove first row (NaN for time_since_last)
df = df.dropna()

#
# Create Window-Based Features
#

# Events in the last hour (rolling count)
df = df.set_index('timestamp')
# Create a column of 1s to count
df['event'] = 1

# Now you can sum them in rolling windows
df['events_last_hour'] = df['event'].rolling('1h').sum()

df.to_csv('output_data.csv', index=False)

# Average time between events in last hour
df['avg_interval_last_hour'] = df['time_since_last'].rolling('1h').mean()

df = df.reset_index()

# #
# # Create Target Variable
# #

# Option 1: Classify activity level
def classify_activity(events_per_hour):
    if events_per_hour <= 2:
        return 'low'
    elif events_per_hour <= 6:
        return 'medium' 
    else:
        return 'high'

df['activity_level'] = df['events_last_hour'].apply(classify_activity)

# Option 2: Predict if next event will be within X minutes
# df['next_event_soon'] = (df['time_since_last'].shift(-1) < 0.25).astype(int)  # within 10 min

# Predict when there will be a long gap (>5 minutes until next motion)
# df['long_gap_coming'] = (df['time_since_last'].shift(-1) > 5).astype(int)
# print(df['long_gap_coming'].value_counts())

# Predict when you're entering a high-activity period
df['next_event_soon'] = (df['time_since_last'].shift(-1) < 3).astype(int)

print(df)

# df.to_csv('output_data.csv', index=False)

# #
# # Look at Features
# #

print(df['activity_level'].value_counts())

features = ['hour', 'day_of_week', 'is_weekend', 'time_since_last', 
           'events_last_hour', 'avg_interval_last_hour']

X = df[features]
y = df['next_event_soon']

# Clean the data (same as before)
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

# Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# NOW USE DECISION TREE instead of Random Forest
simple_model = DecisionTreeClassifier(max_depth=3, random_state=42)
simple_model.fit(X_train, y_train)

# Test it
y_pred = simple_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Full dataset distribution:")
print(y.value_counts())
print(f"Total: {len(y)}")

print("\nTest set distribution:")  
print(y_test.value_counts())
print(f"Total: {len(y_test)}")

print(f"Decision Tree Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

tree_rules = export_text(simple_model, feature_names=features)
print("Decision Tree Rules:")
print(tree_rules)


# Scale your features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Number of features: {X_train_scaled.shape[1]}")

# Create a simple neural network
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Look at the model structure
model.summary()

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Compare to your decision tree
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))