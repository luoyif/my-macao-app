import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
import streamlit as st
from matplotlib import font_manager

# 加载本地字体文件
font_path = "fonts/SimHei.ttf"  # 确保字体文件路径正确
font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# Function to load the data
@st.cache_data
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
    return pd.Series([odd_count, even_count], index=["单数", "双数"])

def analyze_large_small(period_data):
    small_count = period_data[period_data.columns[period_data.columns < 25]].sum().sum()
    large_count = period_data[period_data.columns[period_data.columns >= 25]].sum().sum()
    return pd.Series([small_count, large_count], index=["小号", "大号"])

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
st.title("澳门六合彩分析")

# Load data from local file
data = load_data('data/macao.csv')

# Select range of data to analyze
st.sidebar.header("选择数据范围")
latest_period = len(data)
start_row = st.sidebar.number_input("起始行（最新一期）", min_value=1, max_value=latest_period, value=1)
end_row = st.sidebar.number_input("结束行（最老一期）", min_value=1, max_value=latest_period, value=latest_period)

# Convert start_row and end_row to match the data indexing
start_index = latest_period - start_row
end_index = latest_period - end_row

filtered_data = data.iloc[end_index:start_index + 1]

# Prepare data
period_data, zodiac_data = prepare_data(filtered_data)

# Plotting number occurrences
fig, ax = plt.subplots(figsize=(12, 6))
number_columns = [str(i) for i in range(1, 50)]
number_counts = filtered_data[number_columns].count()
number_counts.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('每个号码出现的次数', fontproperties=font)
ax.set_xlabel('号码', fontproperties=font)
ax.set_ylabel('出现次数', fontproperties=font)
ax.grid(axis='y')
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
st.pyplot(fig)

# Plotting zodiac occurrences
fig, ax = plt.subplots(figsize=(12, 6))
zodiac_columns = filtered_data.columns[50:]
zodiac_counts = filtered_data[zodiac_columns].count()
zodiac_counts.plot(kind='bar', color='lightgreen', ax=ax)
ax.set_title('每个生肖出现的次数', fontproperties=font)
ax.set_xlabel('生肖', fontproperties=font)
ax.set_ylabel('出现次数', fontproperties=font)
ax.grid(axis='y')
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
st.pyplot(fig)

# Plotting odd/even ratio
fig, ax = plt.subplots(figsize=(6, 6))
odd_even_ratio = analyze_odd_even(period_data)
odd_even_ratio.plot(kind='bar', color=['blue', 'orange'], ax=ax)
ax.set_title('单双号比例', fontproperties=font)
ax.set_xlabel('类别', fontproperties=font)
ax.set_ylabel('次数', fontproperties=font)
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
st.pyplot(fig)

# Plotting large/small number ratio
fig, ax = plt.subplots(figsize=(6, 6))
large_small_ratio = analyze_large_small(period_data)
large_small_ratio.plot(kind='bar', color=['green', 'red'], ax=ax)
ax.set_title('大小号比例', fontproperties=font)
ax.set_xlabel('类别', fontproperties=font)
ax.set_ylabel('次数', fontproperties=font)
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
st.pyplot(fig)

# Plotting hot number combinations
fig, ax = plt.subplots(figsize=(12, 6))
hot_combinations = analyze_hot_combinations(period_data)
hot_combinations.plot(kind='bar', color='cyan', ax=ax)
ax.set_title('最常出现的号码组合', fontproperties=font)
ax.set_xlabel('号码组合', fontproperties=font)
ax.set_ylabel('出现次数', fontproperties=font)
ax.grid(True)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
st.pyplot(fig)

# Plotting zodiac combinations heatmap
fig, ax = plt.subplots(figsize=(12, 12))
zodiac_combinations = analyze_zodiac_combinations(zodiac_data)
sns.heatmap(zodiac_combinations, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
ax.set_title('生肖组合热力图', fontproperties=font)
ax.set_xlabel('生肖', fontproperties=font)
ax.set_ylabel('生肖', fontproperties=font)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font)
st.pyplot(fig)
