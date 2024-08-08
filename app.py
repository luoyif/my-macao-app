import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
import streamlit as st

# Function to load the data
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.astype(str)
    return data

# Function to create period_data and zodiac_data
def prepare_data(data):
    number_columns = [str(i) for i in range(1, 50)]
    zodiac_columns = data.columns[50:]
    
    period_data = pd.DataFrame(index=data.index, columns=range(1, 50))
    for i in range(1, 50):
        period_data[i] = data[number_columns].apply(lambda row: 1 if row[str(i)].sum() > 0 else 0, axis=1)
    
    zodiac_data = pd.DataFrame(index=data.index, columns=zodiac_columns)
    for zodiac in zodiac_columns:
        zodiac_data[zodiac] = data[zodiac].apply(lambda x: 1 if x > 0 else 0)
    
    return period_data, zodiac_data

# Define analysis functions
def analyze_odd_even(period_data):
    odd_count = period_data[period_data.columns[period_data.columns % 2 != 0]].sum().sum()
    even_count = period_data[period_data.columns[period_data.columns % 2 == 0]].sum().sum()
    return pd.Series([odd_count, even_count], index=["Odd", "Even"])

def analyze_large_small(period_data):
    small_count = period_data[period_data.columns[period_data.columns < 25]].sum().sum()
    large_count = period_data[period_data.columns[period_data.columns >= 25]].sum().sum()
    return pd.Series([small_count, large_count], index=["Small", "Large"])

def analyze_consecutive(period_data):
    consecutive_counts = []
    for index, row in period_data.iterrows():
        count = 0
        numbers = row[row == 1].index.tolist()
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                count += 1
        consecutive_counts.append(count)
    return pd.Series(consecutive_counts)

def analyze_hot_combinations(period_data, top_n=10):
    all_combinations = []
    for index, row in period_data.iterrows():
        numbers = row[row == 1].index.tolist()
        all_combinations.extend(combinations(numbers, 2))
    combination_counts = pd.Series(all_combinations).value_counts()
    return combination_counts.head(top_n)

def analyze_zodiac_combinations(zodiac_data):
    combinations_counts = pd.DataFrame(index=zodiac_data.columns, columns=zodiac_data.columns, data=0)
    for index, row in zodiac_data.iterrows():
        present_zodiacs = row[row == 1].index.tolist()
        for z1 in present_zodiacs:
            for z2 in present_zodiacs:
                combinations_counts.loc[z1, z2] += 1
    return combinations_counts

# Streamlit app
st.title("Macau Lottery Analysis")

# Load data from local file
data = load_data('data/macao.csv')
period_data, zodiac_data = prepare_data(data)

# Plotting
fig, axes = plt.subplots(5, 2, figsize=(20, 25))

# Plot number occurrences
number_columns = [str(i) for i in range(1, 50)]
number_counts = data[number_columns].count()
number_counts.plot(kind='bar', color='skyblue', ax=axes[0, 0])
axes[0, 0].set_title('Occurrences of Each Number')
axes[0, 0].set_xlabel('Number')
axes[0, 0].set_ylabel('Occurrences')
axes[0, 0].grid(axis='y')

# Plot zodiac occurrences
zodiac_columns = data.columns[50:]
zodiac_counts = data[zodiac_columns].count()
zodiac_counts.plot(kind='bar', color='lightgreen', ax=axes[0, 1])
axes[0, 1].set_title('Occurrences of Each Zodiac Sign')
axes[0, 1].set_xlabel('Zodiac Sign')
axes[0, 1].set_ylabel('Occurrences')
axes[0, 1].grid(axis='y')

# Odd/even analysis
odd_even_ratio = analyze_odd_even(period_data)
odd_even_ratio.plot(kind='bar', color=['blue', 'orange'], ax=axes[1, 0])
axes[1, 0].set_title('Odd/Even Ratio')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(True)

# Large/small number analysis
large_small_ratio = analyze_large_small(period_data)
large_small_ratio.plot(kind='bar', color=['green', 'red'], ax=axes[1, 1])
axes[1, 1].set_title('Large/Small Number Ratio')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True)

# Consecutive number analysis
consecutive_counts = analyze_consecutive(period_data)
consecutive_counts.plot(kind='hist', bins=range(consecutive_counts.max() + 2), color='purple', align='left', rwidth=0.8, ax=axes[2, 0])
axes[2, 0].set_title('Frequency of Consecutive Numbers')
axes[2, 0].set_xlabel('Number of Consecutive Numbers')
axes[2, 0].set_ylabel('Number of Periods')
axes[2, 0].grid(True)

# Hot number combinations analysis
hot_combinations = analyze_hot_combinations(period_data)
hot_combinations.plot(kind='bar', color='cyan', ax=axes[2, 1])
axes[2, 1].set_title('Top 10 Most Frequent Number Combinations')
axes[2, 1].set_xlabel('Number Combination')
axes[2, 1].set_ylabel('Occurrences')
axes[2, 1].grid(True)

# Zodiac combinations heatmap
zodiac_combinations = analyze_zodiac_combinations(zodiac_data)
sns.heatmap(zodiac_combinations, cmap="YlGnBu", annot=True, fmt="d", ax=axes[3, 0])
axes[3, 0].set_title('Zodiac Combinations Heatmap')
axes[3, 0].set_xlabel('Zodiac Sign')
axes[3, 0].set_ylabel('Zodiac Sign')

# 隐藏最后一个子图（axes[3, 1]）如果不需要
fig.delaxes(axes[3, 1])
fig.delaxes(axes[4, 0])
fig.delaxes(axes[4, 1])

st.pyplot(fig)
