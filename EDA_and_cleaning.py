# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Introduction
# This notebook implements cleaning and data exploration of the twitter data for the 
# [Kaggle competition](https://www.kaggle.com/c/nlp-getting-started) on predicting which tweets refer to actual disasters.
#
# It is an exploratory work in progress, containing some unfinished tasks and some thoughts.
#
# ## Layout
# The notebook covers
# - Basic inspection of the data
# - Sentiment analysis using [nltk](https://www.nltk.org/).
# - Implementing word vectorization using the GloVe embeddings pre-trained on twitter data.
# - Text cleaning
#   - A general text cleanin method for analysis
#   - Text cleaning specific for the GloVe embeddings
#  
# ## Thoughts
# - Separate model for each keyword in the data?

# # Imports

# + tags=[]
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Data processing
from nltk.corpus import stopwords

# Other
from tqdm import tqdm  # Progress bar
from IPython.display import display, Markdown  # For printing markdown formatted output

# + [markdown] tags=[]
# # Saving objects
# We create some reuseable code for saving objects for later use, so we don't have to re-run time consuming code.

# +
import pickle
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# -

# # Read data 

# + tags=[]
tweet = pd.read_csv('./input/nlp-getting-started/train.csv')
test  = pd.read_csv('./input/nlp-getting-started/test.csv')
# -

# # Inspect data

# ### Size of the data

print(tweet.shape)
print(test.shape)

# The training data contains 7613 observations, while the test data contains 3263 observations.

tweet.iloc[200:203]

# The training data rows contain a `keyword`, a `location` (sometimes not present), `text` containg a tweet and a `target` coding 1 for disaster and 0 for non-disaster.
#
# At first glance location does not seem trustworthy.

# By inspecting a few tweets we see that we need to clean the text before we can analyze it.

# + tags=[]
for i in [677, 2643, 3134, 92, 2290, 2062, 3681, 2343, 2384, 323]:
    # Print using markdown for better formatting
    tg = tweet.iloc[i]["target"]
    tw = tweet.iloc[i]["text"]
    display(Markdown(f"Target: {tg} -- {tw}"))
# -

# We see that
# - There are many unusual symbols, such as in "MenÛªs", and hashtags (#)
# - We need to remove urls
# - Many tweets have date tags, such as "8/6/2015@2:09 PM:"

# ## Keywords
# The keyword column contain an important keyword present in the tweet, such as "sinking", as presented below.

tweet[tweet['keyword']=='sinking'][["text", "target"]].head(10)

# ## Drop duplicates
# 110 tweets in the dataset have/are duplicates. Some have contradictory labelling. We drop the duplicates, as there are few compared to the data size.

# + tags=[]
np.sum(tweet.duplicated(subset=['text']))

# + tags=[]
tweet.drop_duplicates(subset=['text'], inplace=True)
# -

# ## Check dataset balance
# The dataset appears to be fairly balanced

# + tags=[]
value_dist = tweet.target.value_counts()

sns.barplot(x=value_dist.index, y=value_dist)
plt.show()
# -

# # Sentiment analysis
# We will try adding a sentiment analysis score to our tweets. `SentimentIntensityAnalyzer` from `nltk` gives pieces of text a sentiment score between -1 and 1, where 1 is very positive and -1 is very negative

# TODO: Add estimated sentiment to model input 

# +
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

sia_table = []
for tweet_i in tweet['text']:
    sia_table.append(sia.polarity_scores(tweet_i)['compound'])
# -

tweet['sentiment'] = sia_table

# + [markdown] tags=[]
# We inspect ten random tweets
# -

print(f"Target | Sentiment | Tweet")
for i, row in tweet.iloc[[5585, 1708, 2297, 2759, 6229, 7439, 1766, 5314, 5725, 1582]].iterrows():
    print(f"{row.target:6} | {row.sentiment:9.4f} | {row.text}")

sns.boxplot(x=tweet.target, y=tweet.sentiment)
plt.show()


# Sentiment of disaster tweets seem to fall slightly lower than non-disaster tweets on average. It might have stronger predictive quality together with high leverl features of the tweets discovered by the neural network. 

# Sentiment does not seem to separate the two classes, but might be predictive in connection with higher level features detected by the neural network.
# - TODO: How to implement sentiment in the analysis?

# # Ngram analysis
# - Uninformative
# - From https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove#Exploratory-Data-Analysis-of-tweets

def get_top_tweet_bigrams(corpus, n=None):
    from sklearn.feature_extraction.text import CountVectorizer
    
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# # Word embedding vectorization
# We will vectorize words using a library of vectors from a pre trained model.
#
# - TODO: Fine tune on current dataset
#   - see demo.sh in the github repo
# - Alternative dataset: https://allennlp.org/elmo
#
#
# We will use GloVe for vectorization of words found at https://github.com/stanfordnlp/GloVe,
# trying the twitter dataset first.

# +
# import mmap

def get_num_lines(file_path: str):
    """
    Get the number of lines in a file. 
    Used in the tqdm module to get a progress bar for 
    for-loops when iterating over lines in a file.
    """
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


# +
# Run this once to save an embedding dictionary to a file in and obj/ folder in current dir (`obj` dir must be created beforehand)

embedding_dict={}
file_path = './input/glove.twitter.27B.100d.txt'
with open(file_path,'r') as f:
    for line in tqdm(f, total = get_num_lines(file_path)):
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

save_obj(embedding_dict, "embedding_dict")
# -

# Load saved embedding dictionary from previous cell
embedding_dict = load_obj("embedding_dict")

# - TODO: Visualize embeddings with PCA?

# # Clean tweets

# We will need to preprocess the data based on how words are embedded into the pre-trained embeddings.
#
# We will first attempt some boiler-plate text cleaning, courtesy of among others [this notebook](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove#GloVe-for-Vectorization), and inspect how well it coincides with the embedding.

# ## Inital cleaning attempt 

tweet.text.iloc[3639]


def clean_data(df: pd.DataFrame):
    import string
    import re

    punctuation_regex = re.compile(f"[{re.escape(string.punctuation)}:]")
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # emoticons
                               "\U0001F300-\U0001F5FF"  # symbols & pictographs
                               "\U0001F680-\U0001F6FF"  # transport & map symbols
                               "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "\U00002702-\U000027B0"  # Symbols
                               "\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # Clean training data
    df.keyword.str.replace("%20", " ")  # We replace ascii %20 with space in the keywords, e.g 'body%20bags' -> 'body bags'
    df.text = df.text.str.lower()
    df.text = df.text.str.replace(r"https[^\s|$]*", "", regex=True)  # Change all urls to "URL"
    df.text = df.text.str.replace(punctuation_regex, "", regex=True)
    df.text = df.text.str.replace(emoji_pattern, "", regex=True)  # Remove emojis and symbols
    df.text = df.text.str.replace(r"\n", " ", regex = True)       # Change \n to space
    df.text = df.text.str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)  # Remove last non word characters
    df.text = df.text.str.replace(r"<.*?>", "", regex=True)  # Remove html tags (e.g. <div> )

    return df


cleaned_tweet = clean_data(tweet)

cleaned_tweet.text.iloc[3639]


# ### Check embedding coverage
#
# We check how many of the words in our training set tweets are covered by the embedding dictionary

def word_representation(word_dict: dict , word_list: list):
    """Return the words from uq_words not contained in word_dict 
    and the number of words not covered."""
    n_covered = 0
    not_covered = []
    for word in word_list:
        if word in word_dict:
            n_covered += 1
        else:
            not_covered.append(word)

    return n_covered, not_covered


def get_unused_words(word_dict: dict , word_list: list):
    """Returns a list of words from word_dict not contained in word_list"""
    unused_words = word_dict.copy()#.keys())

    # Remove words in tweet data from the dict
    for word in word_list:
        try:
            unused_words.pop(word)
        except KeyError:
            pass

    #Convert dict to list
    return list(unused_words.keys())


uq_words = tweet.text.str.split(expand=True).stack().unique()
n_covered, not_covered = word_representation(word_dict = embedding_dict, word_list = uq_words)
unused_words = get_unused_words(word_dict = embedding_dict, word_list = uq_words)

# Checking the coverage of the tweets, we see that only 56% of the words in our data are in the embedding dictionary.

n_covered/len(uq_words)

# While only 1% of the words in the embedding dictionary is used.

n_covered/len(embedding_dict)

# This calls for further inquiry. Checking the words not covered by the embeddings, we see that there are numbers, URL's and  words with repeated number of letters (elongated words) eg. 'goooooooaaaaaal' among other things.

",  ".join(not_covered[0:20])

# When inspecting some of the unused words in the embedding dictionary we see that many things, such as hashtags, repeated letters in words, allcaps words and smileys are encoded with special placeholders, such as <allcaps>

"  ".join(unused_words[0:100])

# ## Cleaning and pre-processing for the GloVe embedding
# In their [info page](https://nlp.stanford.edu/projects/glove/) the writers of the GloVe algorithm supply a ruby regex used for text pre-processing for the twitter model.
#
# In an [issue thread](https://github.com/stanfordnlp/GloVe/issues/107) discussing text pre-processing for tweets on their Github page user [skondrashov](https://github.co/skondrashov) supplies a useful python conversion of this ruby script, that also illustrates how words are adjusted to fit in a standard dictionary, and tagged for special characters or rewritings, such as being prefixed with a hashtag or elongated.

# ### Remove contractions
# Copied from https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/, with a small tweak for not matching `'s` in words surrounded by single quotation mark, like `'sylvester stallone'`.

# + jupyter={"outputs_hidden": true} tags=[]
import re

# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "i'd": "i would", "i'd've": "i would have","i'll": "i will",
                     "i'll've": "i will have","i'm": "i am","i've": "i have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
#    adding positive lookbehind for `'s` in the regex to make sure a letter is preceeding
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()).lower().replace("|'s", "|(?<=[a-zA-Z])'s"))

# + tags=[]
tweet= pd.read_csv('./input/nlp-getting-started/train.csv')
test=pd.read_csv('./input/nlp-getting-started/test.csv')


# +
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(1)]
    return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews
df = tweet.copy()
df.loc[:, 'text']=df.loc[:, 'text'].apply(expand_contractions)

# +
# Print the ten first edited tweets

i = 0
j = 0
while j < 5:
    a = tweet.loc[i,'text']
    b = df.loc[i,'text']
    if (len(a) != len(b)):
        print(a)
        print(b)
        j += 1
    i += 1
# -

# ### Cleaning function for repeated letters

# +
# TODO: 

import re


word = "goooaaaallls"  # We want to match go[oo]a[aaa]l[ll]s

# Get list of matches: [(group1, group2, ...), ...] where match group 3 is the repeated letters
m = re.findall(r"(\S*?)(\w)(\2{1,})(\S*?)", word)
m
# -

repeated_letters = [match[2] for match in m]
repeated_letters

# +
from itertools import combinations

# Loop over all combinations of repeated letters
print(f"{'Cleaned word':12s} - Removed letters")
print("=============================")
for i in range(len(repeated_letters), 0, -1):
    for combination in combinations(repeated_letters, r=i+1):
        tword = word
        for letters in combination:
            tword = re.sub(letters, "", tword)
        print(f"{tword:14s}{combination}")
# -

# ### Cleaning function for tweets

# +
from itertools import combinations

def clean_tweets(df):
    import re

    def sub(pattern, output, string, whole_word=False):
        token = output
        if whole_word:
            pattern = r'(\s|^)' + pattern + r'(\s|$)'

        if isinstance(output, str):
            token = ' ' + output + ' '
        else:
            token = lambda match: ' ' + output(match) + ' '

        return re.sub(pattern, token, string)


    def hashtag(token):
        """ Replace hashtag `#` with `<hashtag>` and split following joined words."""
        token = token.group('tag')
        if token != token.upper():
            token = ' '.join(re.findall('[a-zA-Z][^A-Z]*', token))

        return '<hashtag> ' + token

    def punc_repeat(token):
        return token.group(0)[0] + " <repeat>"

    def punc_separate(token):
        return token.group()

    def number(token):
        return token.group() + ' <number>';

    def word_end_repeat(token):
        return token.group(1) + token.group(2) + ' <elong>'
    
    def allcaps(token):
        return token.group() + ' <allcaps>'

    def clean_repeated_letters(tweet: str, embedding_dict: dict):
        """
        Splits a tweet into words, finds repeated letters in the word and
        removes combinations of the repeated letters until the word is matched by a key in
        embedding_dict
        """

        cleaned_tweet = []

        for word_i in tweet.split():
            word_found = False
            if word_i in embedding_dict:
                cleaned_tweet.append(word_i)
                continue

            matches = re.findall(r"""(\S*?)    # 1: Optional preceeding letters
                                     (\w)      # 2: A letter that might be repeated
                                     (\2{1,})  # 3: Repetead instances of the preceeding letter (group 2)
                                     (\S*?)    # 4: Optional trailing letters""",
                                 word_i,
                                 flags=re.X)  # Verbose regex, for commenting
                                 
            repeated_letters = [match[2] for match in matches]
                    
            # Loop over all combinations of repeated letters
            for i in range(len(repeated_letters), 0, -1):  # i decides length of combination
                if word_found:
                    continue
                    
                for combination in combinations(repeated_letters, r = i):
                    if word_found:
                        continue
                        
                    tword = word_i 
                    
                        
                    for letters in combination:
                        tword = re.sub(letters, "", tword, count=1)
                                        
                        # Word in the embedding dict?
                        if (tword in embedding_dict):
                            # Keep the word and stop searching
                            word_found = True
                            tword = tword + " <elong>"
                            continue  
            if not word_found:
                # No match, we simply keep the word
                tword = word_i
                
            cleaned_tweet.append(tword)
            
        return " ".join(cleaned_tweet)



    eyes        = r"[8:=;]"
    nose        = r"['`\-\^]?"
    sad_front   = r"[(\[/\\]+"
    sad_back    = r"[)\]/\\]+"
    smile_front = r"[)\]]+"
    smile_back  = r"[(\[]+"
    lol_front   = r"[DbpP]+"
    lol_back    = r"[d]+"
    neutral     = r"[|]+"
    sadface     = eyes + nose + sad_front   + '|' + sad_back   + nose + eyes
    smile       = eyes + nose + smile_front + '|' + smile_back + nose + eyes
    lolface     = eyes + nose + lol_front   + '|' + lol_back   + nose + eyes
    neutralface = eyes + nose + neutral     + '|' + neutral    + nose + eyes
    punctuation = r"""[ '!"#$%&'()+,/:;=?@_`{|}~\*\-\.\^\\\[\]]+""" ## < and > omitted to avoid messing up tokens

    for i in range(df.shape[0]):
        df.loc[i,'text'] = sub(r'[\s]+',                             '  ',            df.loc[i,'text']) # ensure 2 spaces between everything
        df.loc[i,'text'] = sub(r'(?:(?:https?|ftp)://|www\.)[^\s]+', '<url>',         df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(r'@\w+',                              '<user>',        df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(r'#(?P<tag>\w+)',                     hashtag,         df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(sadface,                              '<sadface>',     df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(smile,                                '<smile>',       df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(lolface,                              '<lolface>',     df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(neutralface,                          '<neutralface>', df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(r'(?:<3+)+',                          '<heart>',       df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(r'\b[A-Z]+\b',                         allcaps,       df.loc[i,'text'], True) 
        # Allcaps tag
        df.loc[i,'text'] = df.loc[i,'text'].lower()
        df.loc[i,'text'] = expand_contractions(df.loc[i, 'text'])
        df.loc[i,'text'] = sub(r'[-+]?[.\d]*[\d]+[:,.\d]*',          number,          df.loc[i,'text'], True)
        df.loc[i,'text'] = sub(punctuation,                          punc_separate,   df.loc[i,'text'])
        df.loc[i,'text'] = sub(r'([!?.])\1+',                        punc_repeat,     df.loc[i,'text'])
#     df.loc[i,'text'] = sub(r'(\S*?)(\w)\2+\b',                   word_end_repeat, df.loc[i,'text'])
        
        df.loc[i,'text'] = clean_repeated_letters(df.loc[i,'text'], embedding_dict)
#     tweet = sub(r"(\S*?)(\w)(\2{1,})(\S*?)",          word_repeat,     tweet)
#     tweet = sub(r'(\S*?)(\w*(\w)\2+\w*)\2+\b',                   word_repeat, tweet)
    return df
# -


# #### Test the cleaning function
# We test the cleaning on some text to see the effect.

# +
temp = pd.DataFrame({"text": [
    u"I'm hoping they're helping, they've got to",
    u'goooooooaaaaaallll, hey its a goall gooal',
    u'http://foo.com/blah_blah http://foo.com/blah_blah/ http://foo.com/blah_blah_(wikipedia) https://foo_bar.example.com/',
    u':\\ :-/ =-( =`( )\'8 ]^; -.- :/',
    u':) :-] =`) (\'8 ;`)',
    u':D :-D =`b d\'8 ;`P',
    u':| 8|',
    u'<3<3 <3 <3',
    u'#swag #swa00-= #as ## #WOOP #Feeling_Blessed #helloWorld',
    u'holy crap!! i won!!!!@@!!!',
    u'holy *IUYT$)(crap!! @@#i@%#@ swag.lord **won!!!!@@!!! wahoo....!!!??!??? Im sick lol.',
    u'this SENTENCE consisTS OF slAyYyyy slayyyyyy #WEIRD caPITalIZAtionn',
    ]})
temp_uncleaned = temp.copy()
clean_tweets(df = temp)

for i in range(temp.shape[0]):
#     print("====================")
    print("Original: ", temp_uncleaned.iloc[i].text)
    print("Cleaned:  ", temp.iloc[i].text)


# + [markdown] tags=[]
# ---
# ### Cleaning and checking embedding coverage

# + tags=[]
tweet= pd.read_csv('./input/nlp-getting-started/train.csv')
test=pd.read_csv('./input/nlp-getting-started/test.csv')
# -

uq_words = tweet.text.str.split(expand=True).stack().unique()
n_covered, not_covered = word_representation(word_dict = embedding_dict, word_list = uq_words)
unused_words = get_unused_words(word_dict = embedding_dict, word_list = uq_words)

#  

# Now 82% of the words in our data are in the embedding dictionary.

n_covered/len(uq_words)

#  

# Still only 1% of the words in the embedding dictionary is used.

n_covered/len(embedding_dict)

#  

# The words not covered by the embedding dictionary seem to be joined words such as `myreligion` many uncommon symbols and words and numbers. Hopefully these do not carry much meaning, and as they are uncommon it will be hard for our model to descipher their meaning.

# Words  in our data not in the embedding dictionary
"  ".join(not_covered[0:100])

# ---
# Unused words from the embedding dictionary are mainly symbols and foreign words. Which means we seem to have captured most of the important meaning-bearing words.

# Unused words in the embedding dictionary
"  ".join(unused_words[0:100])

tweet.text[tweet.text.str.match(r".*jonvoyage")]

# # Spell correction

# +
# # !pip install pyspellchecker
# -

from spellchecker import SpellChecker

# +
spell = SpellChecker()

misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))
# -

# Seems to work nicely.

# Implementation copied from https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

# +
from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


# -

# - TODO implement into the data cleaning

# # TODO
# - Remove non words
# - encode urls?
# - Encode complexity of text: https://pypi.org/project/textstat/
# - Removal of stop words? (library for specific language)
# - Remove URL's
# - Spelling/grammar correction?
#   - Companies like Google and Microsoft have achieved a decent accuracy level in automated spell correction. One can use algorithms like the Levenshtein Distances, Dictionary Lookup etc. or other modules and packages to fix these errors.
#   - Number of misspelled words

# # Remove stopwords
# We can use NLTK to remove common words.
# These words contain little information on their own, but they might convey information in the sentence structure.

# + jupyter={"outputs_hidden": true} tags=[]
from nltk.corpus import stopwords
print(stopwords.words('english'))
