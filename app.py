#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- Load your data ----------
df = pd.read_csv('contact_messages.csv', parse_dates=['date'])  # Change this to your actual filename
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)

# ---------- Sidebar UI ----------
st.sidebar.title("💌 Message Explorer")

selected_date = st.sidebar.date_input("📅 Pick a date")
search_query = st.sidebar.text_input("🔍 Search messages", value="")

# ---------- Filter data ----------
filtered_df = df.copy()

# Filter by selected date
if selected_date:
    filtered_df = filtered_df[filtered_df.index.date == selected_date]

# Filter by search term
if search_query:
    filtered_df = filtered_df[filtered_df['body'].str.contains(search_query, case=False, na=False)]

# ---------- Main display ----------
st.title("💬 Messages")

if filtered_df.empty:
    st.info("No messages found for this date or search.")
else:
    st.write(f"{len(filtered_df)} message(s) found:")
    for ts, row in filtered_df.iterrows():
        time_str = ts.strftime('%b %d, %Y at %I:%M %p').lstrip('0').replace(' 0', ' ')
        st.markdown(f"**[{time_str}] {row['type']}**: {row['body']}")

# ---------- Optional: Messages per day chart ----------
df_daily = df['body'].groupby(pd.Grouper(freq='D')).count().reset_index()
df_daily.columns = ['Date', 'Message Count']

fig = px.line(df_daily, x='Date', y='Message Count', title='Messages Per Day',
              labels={'Date': 'Date', 'Message Count': 'Messages'},
              markers=True)

st.plotly_chart(fig, use_container_width=True)

