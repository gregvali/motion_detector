import pandas as pd
import matplotlib.pyplot as plt
import numpy

# Read the CSV
df = pd.read_csv('motion_data.csv')

print(df)

# If your timestamp column is named 'timestamp'
timestamps = pd.to_datetime(df['timestamp'], format='%H:%M:%S %d/%m/%Y')

# print(f"Total events: {len(timestamps)}")
# print(f"Date range: {timestamps.min()} to {timestamps.max()}")

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
df['events_last_hour'] = df['event'].rolling('1H').sum()
df['events_last_30min'] = df['event'].rolling('30min').sum()

df.to_csv('output_data.csv', index=False)

# Average time between events in last hour
df['avg_interval_last_hour'] = df['time_since_last'].rolling('1h').mean()

df = df.reset_index()

# #
# # Create Target Variable
# #

# Option 1: Classify activity level
def classify_activity(events_per_hour):
    if events_per_hour <= 40:
        return 'low'
    elif events_per_hour <= 100:
        return 'medium' 
    else:
        return 'high'

df['activity_level'] = df['events_last_hour'].apply(classify_activity)

# Option 2: Predict if next event will be within X minutes
df['next_event_soon'] = (df['time_since_last'].shift(-1) < 10).astype(int)  # within 10 min

print(df)

# df.to_csv('output_data.csv', index=False)

# #
# # Look at Features
# #

print(df['activity_level'].value_counts())