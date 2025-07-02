#!/usr/bin/env python
# coding: utf-8

# # Happy Birthday Tara!!!!!!!

# ## What da fuq is this <a class="anchor" id="description"></a>

# waddup. so you know how we both have a deep love for metadata, and i could have pursued a career in some sort of data science but chose to humble myself through a grueling career in the field of medicine? well what better way to celebrate a tara pavithran birthday than to provide something that will be a gift to both me and you, which is doing a meta analysis of our texts to each other over the past year

# waddup. what the h sure:
# 
# first and last texts
# 
# 
# turnaroudn time between texts
# 
# interactive to show the data
# 
# infuse calls in the data too maybe

# i guess lets get cooking, im gonna explain everything im doing since you always ask what the eff im doing when i do things, which is one of the many things i love about you, such a curious mind. anywho, just importing a bunch of bullshit ass modules and functions to use (what a great explanation)

# In[1]:


from bs4 import BeautifulSoup
import csv
import datetime as dt
from datetime import date
import pandas as pd
import numpy as np
import plotly
import re
import string
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import seaborn as sns
from PIL import Image
sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Extraction

# so i uploaded all of our texts into a file, and here im just having it parse through it all, label your phone number as Tara and my number as Samer, along with the body of the texts

# In[2]:


import xml.etree.ElementTree as ET

def extract_messages(xml_file, target_contact):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    messages = []

    for sms in root.findall('sms'):
        contact = sms.get('address')
        if target_contact in contact:
            body = sms.get('body')
            date = sms.get('readable_date')
            type_ = "Tara" if sms.get('type') == "1" else "Samer"
            messages.append({
                'date': date,
                'type': type_,
                'contact': contact,
                'body': body
            })

    return messages

msgs = extract_messages('sms-20250626003200.xml', target_contact='+12247952515')
from IPython.display import display, HTML

# Build HTML content
html_output = "<div style='max-height: 400px; overflow-y: scroll; font-family: monospace; white-space: pre-wrap; border:1px solid #ccc; padding:10px;'>"

for msg in msgs:
    html_output += f"[{msg['date']}] {msg['type']}: {msg['body']}<br>"

html_output += "</div>"

# Display scrollable box
display(HTML(html_output))


# In[ ]:





# okay now im just putting it all into a csv file cuz why the eff not, but more importantly putting it into a dataframe. speaking of dataframe, remember when you were a cs major? look how far you have come, anyways yeah

# In[3]:


import pandas as pd

df = pd.DataFrame(msgs)
df.to_csv('contact_messages.csv', index=False)


# ## Data Preparation <a class="anchor" id="preparation"></a>

# now im just checking that the table looks fine and also gonna convert the date column to a datetime format so python can know what the eff to do with this

# In[4]:


df.head()


# In[5]:


#convert date to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

pd.set_option('display.max_colwidth', None)


# ## The Money <a class="anchor" id="exploration"></a>

# ### Date Picker <a class="anchor" id="exploration"></a>

# In[32]:


import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

# Ensure datetime index
df.index = pd.to_datetime(df.index)

# Set up calendar widget
date_picker = widgets.DatePicker(
    description='Pick a date:',
    disabled=False,
    layout=widgets.Layout(width='250px'),
    style={'description_width': 'initial'}
)

output = widgets.Output()

# Function to display messages from selected date
def display_messages(change):
    with output:
        clear_output()
        if not change['new']:
            return
        
        selected_date = pd.to_datetime(change['new'])  # Convert to Timestamp
        selected_day = selected_date.normalize()       # Midnight on that day

        # Filter rows where timestamp is on the selected day
        messages_on_date = df[df.index.normalize() == selected_day]

        if messages_on_date.empty:
            print(f"\nNo messages found on {selected_day.strftime('%B %d, %Y')}\n")
            return

        print(f"\nMessages on {selected_day.strftime('%B %d, %Y')}:\n")
        for timestamp, row in messages_on_date.iterrows():
            time_str = timestamp.strftime('%I:%M %p')
            print(f"[{time_str}] {row['type']}: {row['body']}")

# Attach callback
date_picker.observe(display_messages, names='value')

# Show calendar and output area
display(date_picker, output)


# ### Searcher <a class="anchor" id="exploration"></a>

# In[36]:


import ipywidgets as widgets
from IPython.display import display, clear_output

# Make sure timestamps are datetime and index is set
df.index = pd.to_datetime(df.index)

# Create search bar widget
search_box = widgets.Text(
    placeholder='Search messages...',
    description='Search:',
    layout=widgets.Layout(width='50%'),
    style={'description_width': 'initial'}
)

search_output = widgets.Output()

# Search function
def search_messages(change):
    with search_output:
        clear_output()
        query = change['new'].strip()
        if not query:
            print("Enter a word or phrase to search mfer")
            return
        
        # Case-insensitive search in 'body' column
        matches = df[df['body'].str.contains(query, case=False, na=False)]

        if matches.empty:
            print(f'No messages found containing "{query}".')
            return

        print(f'{len(matches)} message(s) containing "{query}":\n')
        for timestamp, row in matches.iterrows():
            datetime_str = timestamp.strftime('%b %d, %Y at %I:%M %p').lstrip('0').replace(' 0', ' ')
            print(f"[{datetime_str}] {row['type']}: {row['body']}")


# Attach listener
search_box.observe(search_messages, names='value')

# Display search bar and results
display(search_box, search_output)


# In[43]:


import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ---------- Load your message DataFrame ----------
# Example: df = pd.read_csv('your_messages.csv', parse_dates=['date'])
# Make sure index is datetime if you haven't already
df = pd.read_csv('your_messages.csv', parse_dates=['date'])  # update path/column as needed
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)

# ---------- Sidebar ----------
st.sidebar.title("Message Explorer")

# Calendar Date Picker
selected_date = st.sidebar.date_input("Pick a date to view messages")

# Search input
search_query = st.sidebar.text_input("Search messages", value="")

# ---------- Filter Data ----------
filtered_df = df.copy()

# Filter by date
if selected_date:
    filtered_df = filtered_df[filtered_df.index.date == selected_date]

# Filter by search query
if search_query:
    filtered_df = filtered_df[filtered_df['body'].str.contains(search_query, case=False, na=False)]

# ---------- Display Messages ----------
st.title("📨 Messages")
if filtered_df.empty:
    st.info("No messages found for this date or search.")
else:
    st.write(f"Showing {len(filtered_df)} message(s):\n")
    for ts, row in filtered_df.iterrows():
        timestamp = ts.strftime('%b %d, %Y at %I:%M %p').lstrip('0').replace(' 0', ' ')
        st.markdown(f"**[{timestamp}] {row['type']}**: {row['body']}")

# ---------- Plotly Chart (optional) ----------
# Example: Messages per day
df_daily = df['body'].groupby(pd.Grouper(freq='D')).count().reset_index()
df_daily.columns = ['Date', 'Message Count']

fig = px.line(df_daily, x='Date', y='Message Count', title='Messages Per Day')
st.plotly_chart(fig, use_container_width=True)


# ### How many messages? <a class="anchor" id="how_many"></a>

# In[6]:


print(f"damn i cant believe we sent each other {len(df)} text messages over the course of the past {(date.today() - date(2024, 8, 8)).days} days since we met")


# first and last 20 texts to each other

# In[7]:


# First 20 messages
print("First 20 messages:")
for _, row in df[['body', 'type']].head(20).iterrows():
    print(f"{row['type']}: {row['body']}")

print("\nLast 20 messages:")
# Last 20 messages
for _, row in df[['body', 'type']].tail(20).iterrows():
    print(f"{row['type']}: {row['body']}")


# i feel like whenever we talk about how often we text each other, you get uncomfortable and want it to stop so get ready to get uncomfortable

# In[8]:


#messages per day
messages_day = (
    df.groupby(pd.Grouper(freq='D'))["body"]
      .count()
)



#words per day
df['words_in_text'] = df.body.apply(lambda x: len(str(x).split(' ')))
words_day = (
      df.groupby(pd.Grouper(freq='D'))["words_in_text"]
        .sum()
)


# In[11]:


import plotly.express as px

#reset index so 'date' becomes a column again
messages_day_reset = messages_day.reset_index()
messages_day_reset.columns = ['Date', 'Message Count']

fig = px.line(messages_day_reset, x='Date', y='Message Count', 
              title='Messages per Day',
              markers=True,
              labels={'Date': 'Date', 'Message Count': 'Number of Messages'})

#show hover tooltip by default
fig.update_traces(mode="lines+markers", hovertemplate='Date: %{x}<br>Messages: %{y}')

#make layout prettier
fig.update_layout(width=1100, height=500)

fig.show()


# In[12]:


# Reset index so date becomes a column
words_day_reset = words_day.reset_index()
words_day_reset.columns = ['Date', 'Word Count']

# Create interactive line plot
fig = px.line(words_day_reset, x='Date', y='Word Count', 
              title='Words per Day',
              markers=True,
              labels={'Date': 'Date', 'Word Count': 'Number of Words'})

# Customize hover tooltip
fig.update_traces(mode="lines+markers", hovertemplate='Date: %{x}<br>Words: %{y}')
fig.update_layout(width=1100, height=500)

fig.show()


# In[15]:


#separate samer and tara
samer_daily = df[df['type'] == 'Samer'].resample('D').size()
tara_daily = df[df['type'] == 'Tara'].resample('D').size()


# In[20]:


import plotly.graph_objects as go

#reset index to get 'date' as a column cuz idk what the EFF im doing half the time
tara_df = tara_daily.reset_index()
tara_df.columns = ['date', 'count']

samer_df = samer_daily.reset_index()
samer_df.columns = ['date', 'count']

#create figure
fig = go.Figure()

#t money
fig.add_trace(go.Scatter(
    x=tara_df['date'], y=tara_df['count'],
    mode='lines+markers',
    name='Tara',
    line=dict(color='#16c45b', width=1),
    hovertemplate='Date: %{x}<br>Tara: %{y} messages'
))

#s dawg
fig.add_trace(go.Scatter(
    x=samer_df['date'], y=samer_df['count'],
    mode='lines+markers',
    name='Samer',
    line=dict(color='#ffa514', width=1),
    hovertemplate='Date: %{x}<br>Samer: %{y} messages'
))

#eff around
fig.update_layout(
    title='Messages Per Day',
    xaxis_title='Month',
    yaxis_title='Number of Messages',
    xaxis=dict(range=[pd.Timestamp('2024-09-01'), pd.Timestamp('2025-07-03')]),
    yaxis=dict(rangemode='tozero'),
    width=1100,
    height=500,
    legend=dict(title='Sender'),
    template='simple_white'
)

fig.show()


# In[26]:


import plotly.graph_objects as go

# Calculate difference
difference = samer_daily - tara_daily
diff_df = difference.reset_index()
diff_df.columns = ['date', 'difference']

# Separate Samer-more and Tara-more days
positive = diff_df[diff_df['difference'] > 0]
negative = diff_df[diff_df['difference'] < 0]

# Create figure
fig = go.Figure()

# Samer-more points (orange, shown as positive)
fig.add_trace(go.Scatter(
    x=positive['date'], y=positive['difference'],
    mode='markers',
    name='More from Samer',
    marker=dict(color='#ffa514'),
    hovertemplate='Date: %{x}<br>Samer sent %{y} more messages'
))

# Tara-more points (green, y is still negative, but hover shows absolute)
fig.add_trace(go.Scatter(
    x=negative['date'], y=negative['difference'],
    mode='markers',
    name='More from Tara',
    marker=dict(color='#16c45b'),
    hovertemplate='Date: %{x}<br>Tara sent %{customdata} more messages',
    customdata=(-negative['difference'])  # to show positive in tooltip
))

# Horizontal reference line at y = 0
fig.add_shape(
    type='line',
    x0=diff_df['date'].min(), x1=diff_df['date'].max(),
    y0=0, y1=0,
    line=dict(color='gray', width=1, dash='dash')
)

# Annotations
fig.add_annotation(
    x=diff_df['date'].min(), y=positive['difference'].max(),
    text='More from Samer', showarrow=False,
    font=dict(color='#ffa514')
)

fig.add_annotation(
    x=diff_df['date'].min(), y=negative['difference'].min(),
    text='More from Tara', showarrow=False,
    font=dict(color='#16c45b')
)

# Layout
fig.update_layout(
    title="Who Sends More Messages Per Day",
    xaxis_title="Date",
    yaxis_title="Difference in Message Count (Samer - Tara)",
    width=1000,
    height=500,
    template="simple_white"
)

fig.show()


# In[26]:


# Filter for September 2024 and May 2025
df = df.reset_index()
filtered = df[(df["date"].dt.year == 2024) & (df["date"].dt.month == 9) |
              (df["date"].dt.year == 2025) & (df["date"].dt.month == 6)]

# Group by date (day) and count messages (rows)
firstnlast30 = filtered.groupby(filtered["date"].dt.date).size()


# In[27]:


firstnlast30.plot(kind='bar', figsize=(15,5), title="First and Last 30 Days of Texts (Per Day)")
plt.axvline(x=15.55, color='red', linestyle='--', linewidth=1)


# ### Message exchange per week <a class="anchor" id="messages_week"></a>

# In[28]:


#seperate messages by week
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
weekly_counts = df.resample('W').size()


# In[29]:


#Plot
plt.figure(figsize=(12, 6))
plt.plot(weekly_counts.index, weekly_counts.values, marker='o', linestyle='-')
plt.title('Messages per Week')
plt.xlabel('Week')
plt.ylabel('Number of Messages')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[30]:


#Separate samer and tara
samer_weekly = df[df['type'] == 'Samer'].resample('W').size()
tara_weekly = df[df['type'] == 'Tara'].resample('W').size()


# In[31]:


# Step 4: Plot
plt.figure(figsize=(12, 6))
plt.plot(tara_weekly.index, tara_weekly.values, label='Tara', marker='o', color = "#16c45b")
plt.plot(samer_weekly.index, samer_weekly.values, label='Samer', marker='o', color = "#ffa514")

# Format x and y axes
plt.xlim(pd.Timestamp('2024-08-01'), pd.Timestamp('2025-06-30'))
plt.ylim(bottom=0)

plt.title('Messages Per Week')
plt.xlabel('Month')
plt.ylabel('Number of Messages')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Frequency <a class="anchor" id="first"></a>

# In[32]:


# Updated inclusive love-message pattern
pattern = r"\b((i\s*)?(really\s*)?(love|luv)\s*(you|u)|ily[a-z]*)\b"

# Filter messages
love_msgs = df[df['body'].str.contains(pattern, case=False, na=False, regex=True)]

# Total count
total_love_msgs = len(love_msgs)
print(f"Total 'I love you' messages: {total_love_msgs}")

# Count by type
count_by_type = love_msgs['type'].value_counts()
print(count_by_type)

# Show first 10 chronologically
display(love_msgs.sort_index().head(100))

import warnings


# In[33]:


pattern_fuck = r"\b(fu+ck|f\*+k|f u c k)\s*(you|u)\b"

fuck_msgs = df[df['body'].str.contains(pattern_fuck, case=False, na=False, regex=True)]

print(f"Total 'Fuck you' messages: {len(fuck_msgs)}")
print(fuck_msgs['type'].value_counts())
display(fuck_msgs.sort_index().head(10))


# In[34]:


loved_msgs = df[df['body'].str.contains(r'le encanta|loved\s*["“]', case=False) == True].sort_values(by='date')

print(f"Total Loved Emoji messages: {len(loved_msgs)}")
print(loved_msgs['type'].value_counts())
display(loved_msgs.sort_index().head(100))


# In[35]:


laughed_msgs = df[df['body'].str.contains(r"le hace gracia|laughed at", case=False) == True].sort_values(by='date')

print(f"Total Laughed Emoji messages: {len(loved_msgs)}")
print(laughed_msgs['type'].value_counts())
display(laughed_msgs.sort_index().head(100))


# In[37]:


import pandas as pd

# Prepare series with phrase label
love_counts = love_msgs['type'].value_counts().rename_axis('type').reset_index(name='count')
love_counts['phrase'] = 'ily'

loved_counts = loved_msgs['type'].value_counts().rename_axis('type').reset_index(name='count')
loved_counts['phrase'] = 'Loved'

fuck_counts = fuck_msgs['type'].value_counts().rename_axis('type').reset_index(name='count')
fuck_counts['phrase'] = 'f u'

# Combine all
counts_df = pd.concat([love_counts, loved_counts, fuck_counts], ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt

custom_palette = {
    'Tara': '#16c45b', 
    'Samer': '#ffa514'   
}

plt.figure(figsize=(10,6))
sns.barplot(data=counts_df, x='phrase', y='count', hue='type', palette=custom_palette)
plt.title('Number of Messages Mentioning Each Phrase by Type')
plt.xlabel('Phrase')
plt.ylabel('Count')
plt.show()


# In[38]:


fran_mentioned = df[df['body'].str.contains(r"\sfran", case=False) == True].sort_values(by='date')

print(f"Total Fran messages: {len(fran_mentioned)}")
print(fran_mentioned['type'].value_counts())
display(fran_mentioned.sort_index().head(100))


# In[39]:


stephanie_mentioned = df[df['body'].str.contains(r"stephanie", case=False) == True].sort_values(by='date')

print(f"Total Stephanie messages: {len(stephanie_mentioned)}")
print(stephanie_mentioned['type'].value_counts())
display(stephanie_mentioned.sort_index().head(500))


# In[40]:


julian_mentioned = df[df['body'].str.contains(r"julian|baby jj|jj", case=False) == True].sort_values(by='date')

print(f"Total Julian messages: {len(julian_mentioned)}")
print(julian_mentioned['type'].value_counts())
display(julian_mentioned.sort_index().head(150))


# In[41]:


tara_mentioned = df[df['body'].str.contains(r"tara", case=False) == True].sort_values(by='date')

print(f"Total Tara messages: {len(tara_mentioned)}")
print(tara_mentioned['type'].value_counts())
display(tara_mentioned.sort_index().head(150))


# In[42]:


samer_mentioned = df[df['body'].str.contains(r"samer", case=False) == True].sort_values(by='date')

print(f"Total Samer messages: {len(samer_mentioned)}")
print(samer_mentioned['type'].value_counts())
display(samer_mentioned.sort_index().head(150))


# In[46]:


import pandas as pd

# Prepare series with phrase label
samer_counts = samer_mentioned['type'].value_counts().rename_axis('type').reset_index(name='count')
samer_counts['phrase'] = 'Samer'

tara_counts = tara_mentioned['type'].value_counts().rename_axis('type').reset_index(name='count')
tara_counts['phrase'] = 'Tara'

fran_counts = fran_mentioned['type'].value_counts().rename_axis('type').reset_index(name='count')
fran_counts['phrase'] = 'Fran'

stephanie_counts = stephanie_mentioned['type'].value_counts().rename_axis('type').reset_index(name='count')
stephanie_counts['phrase'] = 'Stephanie'

julian_counts = julian_mentioned['type'].value_counts().rename_axis('type').reset_index(name='count')
julian_counts['phrase'] = 'Julian'

# Combine all
counts2_df = pd.concat([samer_counts, tara_counts, fran_counts, stephanie_counts, julian_counts], ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt

custom_palette = {
    'Tara': '#16c45b', 
    'Samer': '#ffa514'   
}

plt.figure(figsize=(10,6))
sns.barplot(data=counts2_df, x='phrase', y='count', hue='type', palette=custom_palette)
plt.title('Freak Mentioned')
plt.xlabel('')
plt.ylabel('# of Texts')
plt.show()


# In[47]:


import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


#combine all text in the 'body' column
all_text = ' '.join(df['body'].dropna()).lower()

#extract words
words = re.findall(r'\b\w+\b', all_text)

#remove stopwords
filtered_words = [word for word in words if word not in stop_words]

#word frequencies
word_counts = Counter(filtered_words)

#top 20 most common words
top_words = word_counts.most_common(20)

print(top_words)


# In[48]:


#add custom stop words
custom_words = {'like', 'yeah', 'u', 'im', 'lol', 'also', 'get', 'okay', 'gonna'}  # example words you want to exclude
stop_words.update(custom_words)

#remove stopwords
filtered_words = [word for word in words if word not in stop_words]

#word frequencies
word_counts = Counter(filtered_words)

#top 20 most common words
top_words = word_counts.most_common(20)

print(top_words)


# ### Any preferred days to communicate? <a class="anchor" id="preferred_days"></a>

# In[49]:


week_days = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
df['weekday'] = df.index.to_series().apply(lambda x: week_days[x.weekday()])


# In[50]:


df.groupby('weekday').size().plot(kind='bar', figsize=(10,5))


# Well, it doesn't look much significative (btw, what was happening on Tuesdays and Thursdays?)

# In[64]:


df['hour'] = df.index.hour
hour_counts = df['hour'].value_counts().sort_index()

plt.figure(figsize=(10,5))
hour_counts.plot(kind='bar')
labels = []
for h in hour_counts.index:
    suffix = 'AM' if h < 12 else 'PM'
    hour_12 = h if 1 <= h <= 12 else (h-12 if h > 12 else 12)
    labels.append(f"{hour_12} {suffix}")

plt.xticks(ticks=range(len(labels)), labels=labels)
plt.ylabel('Messages')
plt.title('Poor Sleep Visualized')
plt.tight_layout()
plt.show()


# ### What about the hours? <a class="anchor" id="preferred_hours"></a>

# ## Playing with words (NLP) <a class="anchor" id="nlp"></a>

# ### Text normalization <a class="anchor" id="normalization"></a>

# We perform de basic operations:
# - Sentence tokenization
# - Useful substitutions (i.e. contractions)
# - Word tokenization
# - Eliminating punctuation
# - Conversion to lowercase
# - Removing stop words
# - Removing repetition

# We defined a function called `normalize` that encapsulates all the previous operations 

# In[183]:


def normalize(text, join_sentences = True, stops = True, stemmer = False):
    if type(text) not in [type(''),type(u'')]:
        return text
    #r = RegexpReplacer()
    p_stemmer = PorterStemmer()
    word_punct_tokenizer = WordPunctTokenizer()
    repeat = RepeatReplacer()
    x = re.compile('[%s]' % re.escape(string.punctuation))
    if stops:
        stops = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    normalized_sentences = []
    for sentence in sentences:
        #tokens = word_tokenize(r.replace(sentence.lower()))
        tokens = word_punct_tokenizer.tokenize(r.replace(sentence.lower()))
        no_punct = [token for token in tokens if x.sub(u'', token)]
        no_repeat = [replacer.replace(word) for word in no_punct]
        if stops:
            no_repeat = [word for word in no_repeat if word not in stops]
        if stemmer:
            no_repeat = [p_stemmer.stem(word) for word in no_repeat]
        normalized_sentences.append(' '.join(no_repeat))
    if join_sentences:
        normalized_sentences = ' '.join(normalized_sentences)
        return normalized_sentences
    return [normalized_sentence for normalized_sentence in normalized_sentences]


# In[184]:


import re
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

def normalize(text):
    if not isinstance(text, str):
        return text
    # Lowercase the text
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove punctuation (except spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Tokenize
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    # Stem each token
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Join tokens back to string
    return " ".join(stemmed_tokens)


# In[185]:


df['text_normalized'] = df['body'].apply(normalize)


# ### Word frequency <a class="anchor" id="frequency"></a>

# In[186]:


df_with_text = df.text_normalized.dropna()


# In[187]:


top_1000 = pd.Series(' '.join(df_with_text).split()).value_counts()[:1000]

print(top_1000)


# ### Top 25

# In[193]:


pd.Series(' '.join(df_with_text).split()).value_counts()[:25].plot(kind='bar',figsize=(20,10),color='r')


# See the 12th most frequent word? It's the character __`__ instead of __'__. __She__ would use it all the time. My OCD would give me such a hard time with that specific character and I chose not to filter it so one can realize the magnitude of its use 😂 

# In[68]:


# Sort chronologically
df = df.sort_values(by='date').reset_index(drop=True)

# Shift columns to compare with previous message
df['prev_date'] = df['date'].shift(1)
df['prev_type'] = df['type'].shift(1)

# Calculate time difference in minutes
df['response_time_min'] = (df['date'] - df['prev_date']).dt.total_seconds() / 60

# Only keep rows where the sender changes (i.e., actual response)
df_responses = df[df['type'] != df['prev_type']].copy()


# Plot
plt.figure(figsize=(12, 6))
sns.histplot(
    data=df_responses,
    x='response_time_min',
    hue='type',
    bins=50,
    multiple='stack',
    palette={'Samer': '#FF9933', 'Tara': '#66CC99'}
)
plt.xscale('log')  # log scale for x-axis
plt.xlabel('Response Time (minutes, log scale)')
plt.ylabel('Number of Responses')
plt.title('Log-Scaled Response Time Distribution by Sender')
plt.legend(title='Responder')
plt.tight_layout()
plt.show()


# In[69]:


import pandas as pd

df = df.reset_index()

# Filter for dates AFTER October 27, 2024 (exclusive)
filtered_df = df[df['date'] > pd.Timestamp('2024-10-27')]

# Assuming your response time column and type column are named the same as before
def format_time(minutes):
    if minutes < 60:
        return f"{minutes:.2f} min"
    elif minutes < 1440:
        return f"{minutes / 60:.2f} hr"
    else:
        return f"{minutes / 1440:.2f} days"

summary = filtered_df.groupby('type')['response_time_min'].describe().round(2)

formatted_summary = summary.copy()
for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
    formatted_summary[col] = summary[col].apply(format_time)

print(formatted_summary)


# In[197]:


import pandas as pd

# Ensure date column is datetime
df['date'] = pd.to_datetime(df['date'])

# Filter for messages after October 27, 2024 (exclusive)
filtered_df = df[df['date'] > pd.Timestamp('2024-10-27')].sort_values('date')

# Calculate time difference between consecutive messages in minutes
filtered_df['time_diff_min'] = filtered_df['date'].diff().dt.total_seconds() / 60

# Find the max gap
max_gap = filtered_df['time_diff_min'].max()

# Find the index of the message after the gap
max_gap_idx = filtered_df['time_diff_min'].idxmax()

# Get messages before and after the longest gap
msg_before = filtered_df.loc[max_gap_idx - 1] if max_gap_idx - 1 in filtered_df.index else None
msg_after = filtered_df.loc[max_gap_idx]

print(f"Longest gap is {max_gap:.2f} minutes.")

if msg_before is not None:
    print("\nMessage before the gap:")
    print(msg_before[['date', 'type', 'body']])
    
print("\nMessage after the gap:")
print(msg_after[['date', 'type', 'body']])


# In[82]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

bins = np.logspace(-2, 3, num=50)

# Extract hour
df['hour'] = df['date'].dt.hour

# Bin duration (same bins as above)
df['bin'] = pd.cut(df['response_time_min'], bins)

# Pivot for heatmap
heatmap2 = df.groupby(['bin', 'hour']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap2, cmap="magma", cbar_kws={'label': 'Message Count'})
plt.ylabel("Duration Bin (minutes)")
plt.xlabel("Hour of Day")
plt.title("Message Duration by Hour of Day")
plt.yticks(ticks=np.arange(len(heatmap2.index)), labels=[f"{int(b.left)}–{int(b.right)}" for b in heatmap2.index], rotation=0)
plt.tight_layout()
plt.show()


# In[81]:


# Extrac hour
df['hour'] = df['date'].dt.hour

# Bin duration (same bins as above)
df['bin'] = pd.cut(df['response_time_min'], bins)

# Pivot for heatmap
heatmap2 = df.groupby(['bin', 'hour']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap2, cmap="magma", cbar_kws={'label': 'Message Count'})
plt.ylabel("Duration Bin (minutes)")
plt.xlabel("Hour of Day")
plt.title("Message Duration by Hour of Day")
plt.yticks(ticks=np.arange(len(heatmap2.index)), labels=[f"{int(b.left)}–{int(b.right)}" for b in heatmap2.index], rotation=0)
plt.tight_layout()
plt.show()


# In[84]:


import random
import re

def censor_name(text):
    return re.sub(r'\b(samer|tara)\b', '***', text, flags=re.IGNORECASE)

def guessing_game_10_rounds(df):
    if df.empty:
        print("No messages available to play.")
        return

    score = 0
    rounds = 10

    # Sample 10 random messages once
    sample_df = df.sample(n=rounds).reset_index(drop=True)

    # Pre-censor messages in the sample
    sample_df['censored_body'] = sample_df['body'].str.lower().apply(censor_name)
    sample_df['censored_body'] = sample_df['censored_body'].str.replace(r"[\'’]", "", regex=True)

    print("Try to guess who sent each message.")
    print("Our names are censored.\n")

    for i in range(rounds):
        message = sample_df.loc[i, 'censored_body']
        actual_sender = sample_df.loc[i, 'type']

        print(f"Round {i+1}:")
        print(f"Message: \"{message}\"\n")

        guess = input("Who sent this? ").strip().lower()

        if guess == actual_sender.lower():
            print(f"Correct! 🎉 The sender was {actual_sender}.\n")
            score += 1
        else:
            print(f"Wrong! The correct answer was: {actual_sender}.\n")

    print(f"Game Over! Your final score: {score} out of {rounds}")

# Run the game
guessing_game_10_rounds(df)


# In[167]:


df.columns


# In[180]:


# Sort messages by datetime to make sure order is correct
df = df.sort_values('date').reset_index(drop=True)

# Create a column with the sender of the next message
df['next_type'] = df['type'].shift(-1)
df['next_datetime'] = df['date'].shift(-1)

# Calculate the time difference to the next message in seconds or minutes
df['response_time'] = (df['next_datetime'] - df['date']).dt.total_seconds() / 60  # minutes

# Filter only the rows where the next message is from the other person (a response)
df_responses = df[df['type'] != df['next_type']].copy()


# In[181]:


sns.histplot(data=df_responses, x='response_time', hue='type', bins=30, log_scale=True)
plt.xlabel('Response Time (minutes)')
plt.title('Distribution of Response Times by Sender')
plt.show()


# In[182]:


lowest = df_responses['response_time'].min()
highest = df_responses['response_time'].max()
std_dev = df_responses['response_time'].std()

print(f"Lowest response time: {lowest}")
print(f"Highest response time: {highest}")
print(f"Standard deviation of response time: {std_dev}")

df_responses.groupby('type')['response_time'].agg(['mean', 'median'])


# #### Bigrams <a class="anchor" id="bigrams"></a>

# For bigrams I prefer to have the information that's contained in the stop words. Why? Simple...

# In[411]:


df['consecutive_count'].agg(['max', 'mean','median'])


# In[406]:


df.iloc[4070]


# In[195]:


text = "I'm not happy"
print(normalize(text,stops=False))


# Also, we want to preserve the sentence structure so "_I am __happy. I__ like to sing._" does not consider `('happy','I')` as a bigram.

# In[196]:


text = "I'm happy. I like to sing."
print(normalize(text,stops=False))


# Another flag we added is our salvation!

# In[158]:


text = "I'm happy. I like to sing."
print(normalize(text,stops=False,join_sentences=False))


# In[197]:


df['text_normalized_stop'] = df.text.apply(normalize,stops=False,join_sentences=False)


# In[162]:


df_with_text_stop = df.text_normalized_stop.dropna()


# In[161]:


text_bodies = []
for sentences in df_with_text_stop.values:
    for sentence in sentences:
        text_bodies.append(sentence)


# In[160]:


len(text_bodies)


# In[96]:


vect = CountVectorizer(ngram_range=(2,2))
X = vect.fit_transform(text_bodies)
tot_occ = X.sum(axis=0) # counting total occurrences per token
df = pd.DataFrame(tot_occ, columns=vect.get_feature_names())
df_trans = df.transpose()
df_trans.columns = ['occurrences']
occs = df_trans.sort_values(by='occurrences',ascending=False)[:1000]


# In[97]:


plt.figure(figsize=(20, 10))
wordcloud = WordCloud(background_color="black",width=2000, height=1000).fit_words(zip(occs[:1000].index,occs['occurrences'].tolist()))
plt.imshow(wordcloud)
plt.axis("off")


# In[98]:


# store to file because in the Jupyter notebook we cannot observe all the details
wordcloud.to_file("./imgs/bigrams.png")


# ### Top 25

# In[106]:


occs[:25]


# It turns out bigrams are not very informative. This is because I chose to get the stop words. Now we can observe something pretty obvious which is that stop words usually come in couples (_will be, of the, have to_). Let's find out if trigrams add the extra bit of informative content! (_**of the world**_ maybe?)

# #### Trigrams <a class="anchor" id="trigrams"></a>

# In[198]:


vect = CountVectorizer(ngram_range=(3,3))
X = vect.fit_transform(text_bodies)
tot_occ = X.sum(axis=0) # counting total occurrences per token
df = pd.DataFrame(tot_occ, columns=vect.get_feature_names())
df_trans = df.transpose()
df_trans.columns = ['occurrences']
occs = df_trans.sort_values(by='occurrences',ascending=False)[:1000]


# In[199]:


plt.figure(figsize=(20, 10))
wordcloud = WordCloud(background_color="black",width=3000, height=2000).fit_words(zip(occs[:1000].index,occs['occurrences'].tolist()))
plt.imshow(wordcloud)
plt.axis("off")


# In[94]:


# store to file because in the Jupyter notebook we cannot observe all the details
wordcloud.to_file("./imgs/trigrams.png")


# ### Top 25

# In[200]:


occs[:25]


# It is still full of meaningless forms like `and it is`. But it's interesting to observe how they could be grouped towards more meaningful topics.
# 
# "__you so__ much", "love __you so__", "miss __you so__" 
# 
# "want __to be__", "__to be__ with", "__be__ with you"
# 

# ### Sentiment analysis <a class="anchor" id="sentiment"></a>

# Due to effort economy we will use a trained model.

# In[165]:


sentences = ["Guiem is smart, handsome, and funny.", "this is shit", "this is the shit"]

nltk.download('vader_lexicon')


# In[201]:


sid = SentimentIntensityAnalyzer()


# In[202]:


for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    print(ss)


# We define a function that can be easily applied to our dataframe.

# In[203]:


def compound_senti(sentences,sid):
    if sentences and type(sentences) == type([]):
        compounds = []
        for sentence in sentences:
            compounds.append(sid.polarity_scores(sentence).get('compound'))
        return np.array(compounds).mean()
    else:
        return np.nan


# In[206]:


compound_senti(sentences,sid)


# In[205]:


df['senti'] = df.text_normalized_stop.apply(compound_senti,sid=sid)


# In[171]:


senti_days = df.groupby(pd.Grouper(key="datetime", freq='D'))['senti'].mean()


# In[172]:


senti_days.plot(figsize=(20,10),color='orange')
plt.axhline(y=0.0, xmin=-1, xmax=1, hold=None,color='gray',linestyle='--')


# In[173]:


senti_days.describe()


# #### The happiest and the saddest messages!

# Okaaay! What about checking out those peaks (max. positivie and max. negative) to see if the sentiment analyzer is doing its job?

# In[174]:


senti_days.idxmax()


# In[195]:


df[(df.datetime >= '2017-06-02 00:00:00') & (df.datetime <= '2017-06-02 23:59:59')]


# In[192]:


print(df.iloc[932].text)


# Oh! Well, a little context here: we met by the end of February in Menorca. We spent only 2 days together and then she flew to Californa. She was crazy enough to buy tickets to come to Menorca again on 2nd June. The text corresponds to that day, when she arrived after 3 months of not seeing each other... (btw, "t'estim" means "I love you" Catalan).

# In[176]:


senti_days.idxmin()


# In[199]:


df[(df.datetime >= '2018-01-20 00:00:00') & (df.datetime <= '2018-01-20 23:59:59')]


# Haha! It turns out that I filtered the "video chat" messages from Facebook because that's the platform we used for calls, but I forgot to filter WhatsApp 😅 . I'll leave it here as a funny example. Certainly a sad thing, a missed call! 😂

# #### Word clouds by happiest and saddest messages

# In[220]:


df_pos = df[df.senti>0.5].text_normalized.dropna()
top_1000 = pd.Series(' '.join(df_pos).split()).value_counts()[:1000]
happy_mask = np.array(Image.open("./imgs/happy.jpg"))
stopwords = set()
wc = WordCloud(background_color="white",mask=happy_mask,stopwords=stopwords)
wc.generate_from_frequencies(list(top_1000.to_dict().items()))
plt.figure(figsize=(20,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.axis("off")
plt.show()
# store to file because in the Jupyter notebook we cannot observe all the details
wc.to_file("./imgs/happyuni.png")


# In[223]:


df_neg = df[df.senti<-0.5].text_normalized.dropna()
top_1000 = pd.Series(' '.join(df_neg).split()).value_counts()[:1000]
sad_mask = np.array(Image.open("./imgs/sad.jpg"))
stopwords = set()
wc = WordCloud(background_color="white",mask=sad_mask,stopwords=stopwords)
wc.generate_from_frequencies(list(top_1000.to_dict().items()))
plt.figure(figsize=(20,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.axis("off")
plt.show()
# store to file because in the Jupyter notebook we cannot observe all the details
wc.to_file("./imgs/saduni.png")


# #### Sentiment by week

# In[285]:


senti_weeks = df.groupby(pd.Grouper(key="datetime", freq='W-MON'))['senti'].mean()


# In[72]:


fig, ax = plt.subplots()
ax.plot(messages_week.text,'--',color='b',alpha=0.25,label='# messages')
ax.set_ylabel('Num. messages per week')
fig.set_size_inches(20,10)
ax2 = ax.twinx()
ax2.plot(senti_weeks,color='orange',label='sentiment')
ax2.set_ylabel('Sentiment')
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [line.get_label() for line in lines], loc='best')
plt.axhline(y=0.0, xmin=-1, xmax=1, hold=None,color='gray',linestyle='--')


# ### Topic detection <a class="anchor" id="topic"></a>

# For topic detection I prefer to use a stemmer,  we will reduce topically similar words to their root so we don't see similar terms as separated entities. 

# In[269]:


df['text_normalized_stemmer'] = df.body.apply(normalize)


# In[270]:


df_stemmer_text = df[pd.notnull(df['text_normalized_stemmer'])]


# In[271]:


text_months = df_stemmer_text.groupby(pd.Grouper(key="datetime", freq='M'))['text_normalized_stemmer'].apply(lambda x: x.sum())


# Okay, what I'm going to do here is to use the Latent Dirichlet Allocation (LDA) and I'm gonna ask for 10 topics every month and described by 3 words. 

# In[282]:


from gensim import corpora


# In[279]:


pip install --upgrade gensim


# In[283]:


res = []
for month in text_months:
    texts = []
    for doc in month:
        doc_tokens = doc.split()
        texts.append(doc_tokens)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
    res.append(ldamodel.print_topics(num_topics=10, num_words=3))


# In[284]:


months = ['February','March','April','May','June','July','August','September','October','November','December','January',]
for idx,topics in enumerate(res):
    print(months[idx])
    for idx2,topic in enumerate(topics):
        print(idx2,''.join([i.replace('*','').replace('.','') for i in topic[1] if not i.isdigit()]))
    print('\n')


# ## Annex <a class="anchor" id="annex"></a>

# ### Normalize <a class="anchor" id="normalize"></a>

# In[142]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

text = "So, you are departing around 21:00? It's sooo cooool, much excite. I can't wait!"


# * Sentence tokenization

# In[120]:


sentences = sent_tokenize(text)
print(sentences)


# * Some substitutions to avoid contraction problems, etc.

# In[121]:


replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would'),
    (r'gonna','going to'),
    (r'wanna','want to')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in
        patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s


# In[122]:


r = RegexpReplacer()
for idx,sentence in enumerate(sentences):
    sentences[idx] = r.replace(sentence)
print(sentences)


# * Word tokenization

# In[123]:


tokenized = []
for sentence in sentences:
    tokenized.append(word_tokenize(sentence))
print(tokenized)


# * Eliminating punctuation

# In[124]:


x = re.compile('[%s]' % re.escape(string.punctuation))


# In[125]:


no_punct = [token for token in tokenized[2] if x.sub(u'', token)]
print(no_punct)


# * Conversion into lowercase

# In[126]:


lowercase = [token.lower() for token in no_punct]
print(lowercase)


# * Stop words (optional, sometimes we want to keep the sentence structure)

# In[180]:


stops = set(stopwords.words('english'))
no_stops = [word for word in lowercase if word not in stops]
print(no_stops)


# * Repeating characters (cases like 'happy' are respected, 'too' gets lost... but it's a stop word anyway!)

# In[144]:


class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
    def replace(self, word):
        if len(word) > 100:
            return ''
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


# In[145]:


replacer = RepeatReplacer()
replacer.replace('slapping')


# In[181]:


def normalize(text, join_sentences = True, stops = True, stemmer = False):
    if type(text) not in [type(''),type(u'')]:
        return text
    r = RegexpReplacer()
    p_stemmer = PorterStemmer()
    word_punct_tokenizer = WordPunctTokenizer()
    repeat = RepeatReplacer()
    x = re.compile('[%s]' % re.escape(string.punctuation))
    if stops:
        stops = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    normalized_sentences = []
    for sentence in sentences:
        #tokens = word_tokenize(r.replace(sentence.lower()))
        tokens = word_punct_tokenizer.tokenize(r.replace(sentence.lower()))
        no_punct = [token for token in tokens if x.sub(u'', token)]
        no_repeat = [replacer.replace(word) for word in no_punct]
        if stops:
            no_repeat = [word for word in no_repeat if word not in stops]
        if stemmer:
            no_repeat = [p_stemmer.stem(word) for word in no_repeat]
        normalized_sentences.append(' '.join(no_repeat))
    if join_sentences:
        normalized_sentences = ' '.join(normalized_sentences)
        return normalized_sentences
    return [normalized_sentence for normalized_sentence in normalized_sentences]


# In[182]:


text = "I'm not happy"
print(normalize(text,stops=False,stemmer=True))


# ### Other <a class="anchor" id="other"></a>

# FB words per day.

# In[208]:


df['words_in_text'] = df.body.apply(lambda x: len(str(x).split(' ')))


# In[215]:


messages_day = (
    df.set_index("datetime")
      .groupby(pd.Grouper(freq='W'))["body"]
      .count()
)

# Sum number of words in messages per day
words_day = (
    df.set_index("datetime")
      .groupby(pd.Grouper(freq='D'))["words_in_text"]
      .sum()
)


# In[220]:


print(words_day)


# In[218]:


ax = messages_day.plot(figsize=(20,10))
words_day.plot(ax = ax,secondary_y=True,figsize=(20,10))


# Let's zoom in for the first 3 months.

# In[217]:


ax = messages_day[:90].plot(figsize=(20,10))
words_day[:90].plot(ax = ax,secondary_y=True,figsize=(20,10),style='y')

