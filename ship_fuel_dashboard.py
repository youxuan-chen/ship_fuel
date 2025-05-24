import streamlit as st
import pandas as pd
import numpy as np

# 標題
st.title("船舶燃油效率分析儀表板")

# 讀取 CSV 檔案
file_path = "ship_fuel_efficiency.csv"
df = pd.read_csv(file_path)

# 側邊欄篩選器
st.sidebar.header("篩選條件")

# 根據資料內容建立篩選條件（以下為常見欄位處理，你可根據實際資料修改）
if 'Ship Type' in df.columns:
    ship_types = ["All"] + sorted(df['Ship Type'].dropna().unique().tolist())
    selected_ship_type = st.sidebar.selectbox("選擇船型", ship_types)
else:
    selected_ship_type = "All"

if 'Fuel Type' in df.columns:
    fuel_types = ["All"] + sorted(df['Fuel Type'].dropna().unique().tolist())
    selected_fuel_type = st.sidebar.selectbox("選擇燃料種類", fuel_types)
else:
    selected_fuel_type = "All"

if 'Efficiency' in df.columns:
    min_eff = float(df['Efficiency'].min())
    max_eff = float(df['Efficiency'].max())
    selected_eff_range = st.sidebar.slider("選擇效率範圍", min_eff, max_eff, (min_eff, max_eff))
else:
    selected_eff_range = None

# 篩選邏輯
filtered_df = df.copy()

if selected_ship_type != "All":
    filtered_df = filtered_df[filtered_df['Ship Type'] == selected_ship_type]

if selected_fuel_type != "All":
    filtered_df = filtered_df[filtered_df['Fuel Type'] == selected_fuel_type]

if selected_eff_range:
    filtered_df = filtered_df[
        (filtered_df['Efficiency'] >= selected_eff_range[0]) &
        (filtered_df['Efficiency'] <= selected_eff_range[1])
    ]

# 顯示篩選後的結果
st.subheader("篩選後的資料")
st.dataframe(filtered_df)
