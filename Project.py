import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
df = pd.read_csv("C:\\Users\\Arab Singh\\Downloads\\python dataset.csv")

# Objective. 1 - EV Adoption Trends by Year and Make

ev_counts = df.groupby(['Model Year', 'Make']).size().reset_index(name='Count')
top_makes = df['Make'].value_counts().nlargest(5).index
filtered_ev_counts = ev_counts[ev_counts['Make'].isin(top_makes)]

print("EV Registration Trends for Top 5 Makes by Year:")
print(filtered_ev_counts.sort_values(by=['Model Year', 'Make']))

# Obj. 2 - Electric Range Distribution

range_stats = df['Electric Range'].describe()
print("Electric Range Summary:")
print(range_stats)

# Detect potential outliers
outliers = df[df['Electric Range'] > 350]
print("\nEVs with unusually high electric range (>350 miles):")
print(outliers[['Make', 'Model', 'Electric Range']])

# Obj. 3 - MSRP Comparison Across Vehicle Types

msrp_stats = df.groupby('Electric Vehicle Type')['Base MSRP'].describe()
print("MSRP Statistics by Vehicle Type:")
print(msrp_stats)

# Obj. 4 - Correlation Between Numerical Features

numerical_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_df.corr()

print("Correlation Matrix for Numeric Features:")
print(correlation_matrix)

# Obj. 5 - Electric Vehicle Type Distribution

vehicle_type_counts = df['Electric Vehicle Type'].value_counts()
vehicle_type_percent = df['Electric Vehicle Type'].value_counts(normalize=True) * 100

print("Electric Vehicle Type Distribution (Count and Percentage):")
print(pd.DataFrame({'Count': vehicle_type_counts, 'Percentage': vehicle_type_percent.round(2)}))

plt.figure(figsize=(10, 6))
sns.histplot(df['Electric Range'].dropna(), bins=30, kde=True, color='skyblue')
plt.title("Histogram: Electric Range")
plt.xlabel("Electric Range (miles)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

top_makes_count = df['Make'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_makes_count.values, y=top_makes_count.index, palette="magma")
plt.title("Barplot: Top 10 EV Makes")
plt.xlabel("Count")
plt.ylabel("Make")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Electric Vehicle Type', y='Base MSRP')
plt.title("Boxplot: MSRP by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Base MSRP ($)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
numerical_df = df.select_dtypes(include=['int64', 'float64'])
correlation = numerical_df.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

ev_type_counts = df['Electric Vehicle Type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(ev_type_counts, labels=ev_type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Pie Chart: Electric Vehicle Type Distribution")
plt.tight_layout()
plt.show()