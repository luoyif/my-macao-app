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
    return pd.Series([odd_count, even_count], index=["单数", "双数"])

def analyze_large_small(period_data):
    small_count = period_data[period_data.columns[period_data.columns < 25]].sum().sum()
    large_count = period_data[period_data.columns[period_data.columns >= 25]].sum().sum()
    return pd.Series([small_count, large_count], index=["小号", "大号"])

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

# Plotting
fig, axes = plt.subplots(5, 2, figsize=(20, 25))

# Plot number occurrences
number_columns = [str(i) for i in range(1, 50)]
number_counts = filtered_data[number_columns].count()
number_counts.plot(kind='bar', color='skyblue', ax=axes[0, 0])
axes[0, 0].set_title('每个号码出现的次数', fontproperties=font)
axes[0, 0].set_xlabel('号码', fontproperties=font)
axes[0, 0].set_ylabel('出现次数', fontproperties=font)
axes[0, 0].grid(axis='y')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), fontproperties=font)

# Plot zodiac occurrences
zodiac_columns = filtered_data.columns[50:]
zodiac_counts = filtered_data[zodiac_columns].count()
zodiac_counts.plot(kind='bar', color='lightgreen', ax=axes[0, 1])
axes[0, 1].set_title('每个生肖出现的次数', fontproperties=font)
axes[0, 1].set_xlabel('生肖', fontproperties=font)
axes[0, 1].set_ylabel('出现次数', fontproperties=font)
axes[0, 1].grid(axis='y')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), fontproperties=font)

# Odd/even analysis
odd_even_ratio = analyze_odd_even(period_data)
odd_even_ratio.plot(kind='bar', color=['blue', 'orange'], ax=axes[1, 0])
axes[1, 0].set_title('单双号比例', fontproperties=font)
axes[1, 0].set_xlabel('类别', fontproperties=font)
axes[1, 0].set_ylabel('次数', fontproperties=font)
axes[1, 0].grid(True)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), fontproperties=font)

# Large/small number analysis
large_small_ratio = analyze_large_small(period_data)
large_small_ratio.plot(kind='bar', color=['green', 'red'], ax=axes[1, 1])
axes[1, 1].set_title('大小号比例', fontproperties=font)
axes[1, 1].set_xlabel('类别', fontproperties=font)
axes[1, 1].set_ylabel('次数', fontproperties=font)
axes[1, 1].grid(True)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), fontproperties=font)

# Consecutive number analysis
consecutive_counts = analyze_consecutive(period_data)
consecutive_counts.plot(kind='hist', bins=range(consecutive_counts.max() + 2), color='purple', align='left', rwidth=0.8, ax=axes[2, 0])
axes[2, 0].set_title('连号频率', fontproperties=font)
axes[2, 0].set_xlabel('连号数量', fontproperties=font)
axes[2, 0].set_ylabel('期数', fontproperties=font)
axes[2, 0].grid(True)
axes[2, 0].set_xticklabels(axes[2, 0].get_xticklabels(), fontproperties=font)

# Hot number combinations analysis
hot_combinations = analyze_hot_combinations(period_data)
hot_combinations.plot(kind='bar', color='cyan', ax=axes[2, 1])
axes[2, 1].set_title('最常出现的号码组合', fontproperties=font)
axes[2, 1].set_xlabel('号码组合', fontproperties=font)
axes[2, 1].set_ylabel('出现次数', fontproperties=font)
axes[2, 1].grid(True)
axes[2, 1].set_xticklabels(axes[2, 1].get_xticklabels(), fontproperties=font)

# Zodiac combinations heatmap
zodiac_combinations = analyze_zodiac_combinations(zodiac_data)
sns.heatmap(zodiac_combinations, cmap="YlGnBu", annot=True, fmt="d", ax=axes[3, 0])
axes[3, 0].set_title('生肖组合热力图', fontproperties=font)
axes[3, 0].set_xlabel('生肖', fontproperties=font)
axes[3, 0].set_ylabel('生肖', fontproperties=font)
axes[3, 0].set_xticklabels(axes[3, 0].get_xticklabels(), fontproperties=font)
axes[3, 0].set_yticklabels(axes[3, 0].get_yticklabels(), fontproperties=font)

# 隐藏最后一个子图（axes[3, 1]）如果不需要
fig.delaxes(axes[3, 1])
fig.delaxes(axes[4, 0])
fig.delaxes(axes[4, 1])

st.pyplot(fig)
