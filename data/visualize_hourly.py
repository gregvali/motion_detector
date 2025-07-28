import pandas as pd

df = pd.read_csv('hourly_motion_counts.csv')
df = df.set_index('hour')

print(df)