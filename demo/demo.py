#
# if model_save_file != None:
#     pickle.dump(model_save, open(model_save_file, 'wb'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris=load_iris()
X, y = iris.data, iris.target
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)
print(X_new.scores )
vectorizer = CountVectorizer(ngram_range=(1, 2),
                                token_pattern=r'\b\w+\b', min_df=1)
corpus = [
  'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',]
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
print(X.toarray())
X = vectorizer.fit_transform(corpus).toarray()
print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
print(X)
