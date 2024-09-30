import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('out.csv')

## Ignore Below Price:1000

filtered_df = df[df['price'] >= 1000]

print(len(filtered_df['price']))
print(max(filtered_df['price']))

plt.figure(figsize=(10, 6))
plt.hist(filtered_df['price'], bins=50, edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution (Ignoring Values Below 1000)')
plt.show()

## Ignore Above Price:1000

df = pd.read_csv('out.csv')

filtered_df = df[df['price'] <= 1000]

print(len(filtered_df['price']))
print(max(filtered_df['price']))

plt.figure(figsize=(10, 6))
plt.hist(filtered_df['price'], bins=50, edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution (Ignoring Values Above 1000)')
plt.xlim(0, 1000) 
plt.show()