#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1) Data Exploration

from sysconfig import get_python_version
import pandas as pd
column_names = ['target','ids','date','flag','user','text']
# Loading the dataset
df = pd.read_csv("twitter_data.csv", names = column_names,encoding='ISO-8859-1')
df


# In[2]:


# Displaying the basic information about the dataset
print(df.info())


# In[3]:


# Displaying the first few rows of the dataset
print(df.head())


# In[4]:


# 2) Data Cleaning

#  (i) Dropping the irrelevant columns
#  Drop unnecessary columns
# df = df[['target', 'text']]

# (i) Handling the missing values
df.dropna(inplace=True)

# (ii) Dropping the duplicate entries
df.drop_duplicates(inplace=True)

# (iii) Displaying the cleaned dataset
print(df.head())


# In[5]:





# In[6]:


# 3) Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Conducting EDA to gain initial insights
# (e.g., summary statistics, distribution of tweet lengths)

# Visualizing sentiment distribution
sns.countplot(x='target', data=df)
plt.title('Sentiment Distribution')
plt.show()


# In[7]:


# 4) Sentiment Distribution

# Visualizing the distribution of sentiment labels
sns.countplot(x='target', data=df)
plt.title('Sentiment Distribution')
plt.show()

# Analyzing the balance of sentiment classes
sentiment_counts = df['target'].value_counts()
print(sentiment_counts)


# In[8]:





# In[9]:


# 5) Word Frequency Analysis

from wordcloud import WordCloud

# Analyzing the word frequency in tweets
positive_tweets = df[df['target'] == 4]['text']
negative_tweets = df[df['target'] == 0]['text']

# Creating the word clouds for positive and negative sentiments
positive_wordcloud = WordCloud().generate(' '.join(positive_tweets))
negative_wordcloud = WordCloud().generate(' '.join(negative_tweets))

# Displaying the word clouds
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Word Cloud')
plt.axis('off')
plt.show()

plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Word Cloud')
plt.axis('off')
plt.show()


# In[10]:


# 6) Temporal Analysis

# Assuming the index represents the order of tweets in the dataset
# Converting the index to datetime (assuming it's a numerical index)
df.index = pd.to_datetime(df.index, unit='s')  # 's' assumes the index is in seconds, adjust if needed

# Exploring how sentiment varies over the "pseudo-time" index
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y='target', data=df)
plt.title('Temporal Analysis of Sentiment (Based on Tweet Order)')
plt.show()


# In[11]:





# In[12]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[13]:





# In[14]:


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


# In[15]:


# 7) Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)
df2 = df.head(15000)
df2['processed_text'] = df2['text'].apply(preprocess_text)
df2
print("After Text Processing:- ")
df2['processed_text']


# In[16]:


# 8) Sentiment Prediction Model

X_train, X_test, y_train, y_test = train_test_split(df2['processed_text'], df2['target'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {accuracy:.2f}')
print(f'F1 Score: {f1:.2f}')


# In[17]:


# 9) Feature Importance
feature_names = vectorizer.get_feature_names_out()
feature_importance = model.feature_importances_

# Creating a DataFrame to store feature names and their importance scores
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
top_features = feature_df.nlargest(20, 'Importance')

# Visualizing feature importance
plt.figure(figsize=(10, 6))
plt.bar(top_features['Feature'], top_features['Importance'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 20 Features Importance')
plt.show()


# In[18]:





# In[19]:


get_python_version().system('pip install ipywidgets')


# In[20]:


# 10. User Interface:
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from ipywidgets import widgets, interact
from IPython.display import display

# Load the dataset
df = pd.read_csv('twitter_data.csv', header=None, encoding='ISO-8859-1')
df.columns = ['target', 'id', 'date', 'query', 'user', 'text']

# Remove unnecessary columns
df = df[['target', 'text']]

# Map target values to sentiment labels
df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})

# Clean the text data (remove URLs, special characters, etc.)
def clean_text(text):
    # Your text cleaning code here
    # For simplicity, let's assume we only want to convert text to lowercase
    cleaned_text = text.lower()
    return cleaned_text

df['cleaned_text'] = df['text'].apply(clean_text)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_vectorized, y)

# Function to predict sentiment for input text
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# User Interface
def analyze_sentiment(input_text):
    result = predict_sentiment(input_text)
    print(f"Sentiment Prediction: {result}")

# Create interactive widget
text_input = widgets.Text(description="Enter text:", value="Lyx is cool")
interact(analyze_sentiment, input_text=text_input)


