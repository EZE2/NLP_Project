import os
from urllib.request import urlopen
import nltk
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('vader_lexicon')
import openpyxl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import csv
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# os.environ["CUDA_VISIBLE_DEVICE"]='0'


# Insert Dataset Path

# This script use its own directory as dataset path, Insert your Dataset with same path of the script
DATASETPATH = os.path.dirname(os.path.abspath(__file__))
DATASET = DATASETPATH + '/IMDB Dataset2.csv'

# Calling Dataset
review = []
sentiment = []
sum_review = ''
f = open(DATASET, 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    review.append(line[0])
    sentiment.append(line[1])
f.close()

# Remove first row
del review[0]
del sentiment[0]

for line in review:
    sum_review = sum_review + line

# Tokenize & Sentiment Analysis

sid = SentimentIntensityAnalyzer()
score_list = []
for content in review:
    lines_list = tokenize.sent_tokenize(content)
    sum = 0
    for sent in lines_list:
        ss = sid.polarity_scores(sent)
        sum = sum + ss['compound']
    finalsum = str(sum / len(lines_list))
    score_list.append(finalsum)

# If you want to see score_list print this
# print(score_list)


# Get Default Accuracy

accuracy = 0
for i in range(len(score_list)):
    if (float)(score_list[i]) > 0.18115:
        if sentiment[i] == 'positive':
            accuracy += 1
    else:
        if sentiment[i] == 'negative':
            accuracy += 1

print(accuracy / len(sentiment) * 100)

# Wordcloud output
# FontPath
FONTPATH = DATASETPATH + '/framd.ttf'

# Wordcloud Generation

# You can add Stopwords here
stop_words = ['the', 'movie', 'film', 'the movie', 'the film', 'this', 'this movie', 'this film', 'part of', 'this is',
              'movie and'] + list(STOPWORDS)


def generate_wordcloud(text):
    wordcloud = WordCloud(font_path=FONTPATH,
                          width=2400, height=1800,
                          ranks_only=None,
                          relative_scaling=0.8,
                          stopwords=stop_words
                          ).generate(text)
    plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


generate_wordcloud(sum_review)

# Test Set Analysis
x_range = list(range(0, len(score_list)))
y_range = []
for i in range(len(score_list)):
    y_range.append((float)(score_list[i]))
plt.figure(2)
plt.scatter(x_range, y_range, c=y_range, cmap='jet')
plt.show()

pos = 0
neg = 0
neu = 0
for i in range(len(y_range)):
    if y_range[i] >= 0.15:
        pos = pos + 1
    elif y_range[i] <= -0.15:
        neg = neg + 1
    else:
        neu = neu + 1

print("Positive Review : ", pos)
print("Negative Review : ", neg)
print("Neutral Review : ", neu)

wrong = 0
neuAndRight = 0
neuAndWrong = 0
# POS / NEU / NEG Classification : Can Change Value of If y_range[i] >= NUMBER :
# neuAndRight stands for which review was analyzed by NTLK but it was Positive-Positive in Test - Prediction Model
# neuAndWrong vice versa
# Thats because we're using only Pos/Neg but our NTLK analysis using Pos/Neu/Neg
# Just note it
for i in range(len(y_range)):
    if y_range[i] >= 0.25:
        if sentiment[i] == 'negative':
            wrong = wrong + 1
    elif y_range[i] <= -0.25:
        if sentiment[i] == 'positive':
            wrong = wrong + 1
    else:
        if y_range[i] > 0.18115:
            if sentiment[i] == 'positive':
                neuAndRight = neuAndRight + 1
            else:
                neuAndWrong = neuAndWrong + 1
        else:
            if sentiment[i] == 'negative':
                neuAndRight = neuAndRight + 1
            else:
                neuAndWrong = neuAndWrong + 1

print("totaly wrong : ", wrong)
print("Neutral by NTLK but Right : ", neuAndRight)
print("Neutral by NTLK but Wrong : ", neuAndWrong)

print("total wrongs : ", wrong + neuAndWrong)
print("Accuracy : ", (len(y_range) - (wrong + neuAndWrong)) / len(y_range) * 100)

print('======================================================================')
print('======================================================================')

# For F-measure Printout

movie_sentiment = []
predict_sentiment = []

# positive = 1, negative = 0

for i in range(len(score_list)):
    if (float)(score_list[i]) > 0.18115:
        predict_sentiment.append(1)
    else:
        predict_sentiment.append(0)
    if sentiment[i] == 'positive':
        movie_sentiment.append(1)
    else:
        movie_sentiment.append(0)

print("::Confusion Matrix")
print(confusion_matrix(movie_sentiment, predict_sentiment))
print("::Classification Report (positive = 1, negative = 0")
print(classification_report(movie_sentiment, predict_sentiment))
print("::Accuracy Score")
print(accuracy_score(movie_sentiment, predict_sentiment))
