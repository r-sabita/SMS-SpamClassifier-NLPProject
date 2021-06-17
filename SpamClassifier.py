# First import the required library

import pandas as pd
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# download stopwords
#nltk.download('stopwords') 

ps = PorterStemmer()

# load the dataset
messages = pd.read_csv('data/SMSSpamCollection', sep='\t', names=['label', 'message'])

# Data cleaning and Preprocessing
corpus = []
for i in range(0, len(messages)):
    mgs = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) 
    mgs = mgs.lower() 
    mgs = mgs.split() 

    mgs = [ps.stem(word) for word in mgs if not word in stopwords.words('english')] 
    mgs = ' '.join(mgs) 
    corpus.append(mgs) 

# Create Bag Of Words model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

# Train Test Split (Splitting dataset into 80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model(X_test)

# Compare the test data and predicted data using confusion matrix
confusion_m = confusion_matrix(y_test, y_pred)

# Get the accuracy using accuracy score
accuracy = accuracy_score(y_test, y_pred)
