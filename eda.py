import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("locomotive_route_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

print("\nTarget Distribution:")
print(df["is_delayed"].value_counts())

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True))
plt.show()