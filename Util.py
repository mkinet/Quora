from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
import pandas as pd
from nltk.stem import SnowballStemmer
from collections import Counter

nclass = 2

data_path = "./data/"

# DATA LOADING
def load_data():
    train_file = data_path + "train.csv"
    test_file = data_path + "test.csv"
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print('Training set has %d rows and %d columns'%(train_df.shape[0],train_df.shape[1]))
    print('Test set has %d rows and %d columns'%(test_df.shape[0],test_df.shape[1]))

    return train_df, test_df

# PERFORMANCE EVALUATION
def mlogloss(true,pred):
    return log_loss(true,pred,labels=[0,1,2])


def cv_score(clf, X,y, nfolds=10, random_state=125):

    cv = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=random_state)
    score = cross_val_score(clf,X,y,scoring='neg_log_loss',cv=cv)

    return -score.mean(),score.std()


def cv_pred(clf, X, y, nfolds=10, random_state=125):

    cv = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=random_state)
    pred = cross_val_predict(clf,X,y,cv=cv,method='predict_proba')

    return pred


def scale(train,test):
    standard_scaler = StandardScaler()
    xtrain = standard_scaler.fit_transform(train)
    xtest = standard_scaler.transform(test)

    return xtrain,xtest

# MODELS
def logistic_lasso(lam):
    lam = max(lam,1e-8)
    return


def logistic_ridge(lam):
    lam = max(lam,1e-8)
    return LogisticRegression(penalty='l2',C=1/lam,multi_class='multinomial',solver='lbfgs')


# text features
stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those',
              'then','just','so','than','such','both','through','about','for','is','of','while','during','to',
              'What','Which','Is','If','While','This']
#punctuation='["\'?,\.]' # I will replace all these punctuation with ''
abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",
    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",
    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",
    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",
    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    "\'s": " ",
    "\'ve": " have ",
    "can't": "cannot ",
    "n't": " not ",
    "I'm": "I am",
    " m ": " am ",
    "\'re": " are ",
    "\'d": " would ",
    "\'ll": " will ",
    "60k": " 60000 ",
    " e g ": " eg ",
    " b g ": " bg ",
    "\0s": "0",
    " 9 11 ": "911",
    "e-mail": "email",
    "\s{2,}": " ",
    "quikly": "quickly",
    " usa ": " America ",
    " USA ": " America ",
    " u s ": " America ",
    " uk ": " England ",
    " UK ": " England ",
    "imrovement": "improvement",
    "intially": "initially",
    " dms ": "direct messages ",
    "demonitization": "demonetization",
    "actived": "active",
    "kms": " kilometers ",
    "KMs": " kilometers ",
    " cs ": " computer science ",
    " upvotes ": " up votes ",
    " iPhone ": " phone ",
    "\0rs ": " rs ",
    "calender": "calendar",
    "ios": "operating system",
    "programing": "programming",
    "bestfriend": "best friend",
    "dna": "DNA",
    "III": "3",
    "the US": "America",
    " J K ": " JK ",
    '[!#$%&"\'()*+,-?,\./:;<=>@^`{|}~]':' ',
    "[^A-Za-z0-9]": " ",
     '\s+':' ', # replace multi space with one single space
 }

def get_words_weights(train):
    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
    def get_weight(count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    train_qs = train['question1'].tolist() + train['question2'].tolist()
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    return weights

def text_to_wordlist(text, remove_stop_words=False, stem_words=False):
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text

def clean_text_df(df):
    # There are two lines with NaN in field ' Question 2? Note for later : for those two lines question 1 are duplicates...
    df.fillna('kapweeeeeee', inplace=True)
    df.replace(abbr_dict, regex=True, inplace=True)
    df['question1'] = df['question1'].apply(text_to_wordlist)
    df['question2'] = df['question2'].apply(text_to_wordlist)

    return df


def clean_text(train,test):
    train = clean_text_df(train)
    test = clean_text_df(test)

    return train, test


