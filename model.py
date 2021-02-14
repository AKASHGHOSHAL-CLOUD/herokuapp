import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
import nltk
from nltk.corpus import stopwords


df = pd.read_csv('Copy of wa_all_labels.csv')
X = df['message']
y = df['label_maj']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train,y_train)

# Alternative Usage of Saved Model
# joblib.dump(clf, 'NB_spam_model.pkl')
# NB_spam_model = open('NB_spam_model.pkl','rb')
# clf = joblib.load(NB_spam_model)

pickle.dump(clf, open('model.pkl','wb'))
pickle.dump(cv, open('vec.pkl','wb'))
