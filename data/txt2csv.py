import csv
import pandas as pd

# with open('collection07222025.txt', 'r') as input_file:
#     with open('motion_data.csv', 'w', newline='') as output_file:
#         writer = csv.writer(output_file)
#         writer.writerow(['timestamp'])
        
#         for line in input_file:
#             line = line.strip()
#             line = line[10:19] + line[22:32]
#             # line = line.replace("/","")
#             # line = line.replace(":","")
#             writer.writerow([line])

# with open('collection07222025.txt', 'r') as input_file:
#     with open('motion_by_hour.csv', 'w', newline='') as output_file:
#         writer = csv.writer(output_file)
#         writer.writerow(['hour', 'events'])
        
#         for i in range(24):
#             writer.writerow([i, ''])

#         for line in input_file:
#             line = line.strip()
#             line = line[10:19] + line[22:32]
#             # line = line.replace("/","")
#             # line = line.replace(":","")
#             writer.writerow([line])

df = pd.read_csv('motion_data.csv')

# If your timestamp column is named 'timestamp'
timestamps = pd.to_datetime(df['timestamp'], format='%H:%M:%S %d/%m/%Y')

hourly_counts = timestamps.dt.hour.value_counts().sort_index()

df_hourly = pd.DataFrame({
    'hour': hourly_counts.index,
    'event_count': hourly_counts.values
})

# Make sure all 24 hours are represented (fill missing hours with 0)
all_hours = pd.DataFrame({'hour': range(24)})
df_hourly = all_hours.merge(df_hourly, on='hour', how='left').fillna(0)

df_hourly['event_count'] = df_hourly['event_count'].astype(int)

print(df_hourly)

df_hourly.to_csv('hourly_motion_counts.csv', index=False)