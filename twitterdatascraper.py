import re
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import snscrape.modules.twitter as sntwitter

query = "#iphone14 lang:en"
tweets = []
limit = 100
for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    if len(tweets) == limit:
        print('end of scrape')
        break
    else:

        tweets.append([tweet.content])
df = pd.DataFrame(tweets, columns=['Tweet'])


def textrazor(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^\u0000-\u05C0\u2100-\u214F]', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT\S+', '', text)
    text = re.sub(r'https:\/\/\S+', '', text)
    text = re.sub(r'http:\/\/\S+', '', text)
    stopwords(text)
    return text


df['Tweet'] = df['Tweet'].apply(textrazor)


stop_words = stopwords.words('english')
df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


def subjector(text):
    return TextBlob(text).sentiment.subjectivity


def polaritor(text):
    return TextBlob(text).sentiment.polarity


df['Subjectivity'] = df['Tweet'].apply(subjector)
df['Polarity'] = df['Tweet'].apply(polaritor)

allWords = ' '.join([twt for twt in df['Tweet']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allWords)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()


def analyser(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


df['Analysis'] = df['Polarity'].apply(analyser)

plt.figure(figsize=(8, 6))
for i in range(0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i],
                color='Blue', s=1)
    plt.title('Analysis')
    plt.xlabel('Polarity')
    plt.ylabel('subjectivity')

negtwt = df[df.Analysis == 'Negative']
negtwt = negtwt['Tweet']
neutwt = df[df.Analysis == 'Neutral']
neutwt = neutwt['Tweet']
postwt = df[df.Analysis == 'Positive']
postwt = postwt['Tweet']
sec = [negtwt, neutwt, postwt]

for i in range(3):
    res = round((sec[i].shape[0] / df.shape[0]*100), 1)
    print(res)
    a = 0

df_core = df[['Tweet', 'Analysis']]

df_core = df_core[df_core['Analysis'] != 'Neutral']

df_core.to_csv('iphone14_twt.csv', index=False)

print(df_core['Tweet'])
