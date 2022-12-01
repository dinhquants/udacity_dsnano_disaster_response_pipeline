import sys

import re
import pickle
import pandas as pd 
import numpy as np
import nltk 
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['wordnet', 'punkt', 'stopwords','averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Load Data Function
    Input: database_filepath -> path to SQLite db
    
    Return:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)  
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X, Y, category_names
    
def tokenize(text):
    """Split text into words and return the root form of the words
    Input: text(str): the message
    
    Return:
      lemm(list of str): a list of the root form of the message words
    """
    # Tokenize text  
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """Build model.
    Input: n/a
    
    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20]
    }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model
    Input:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    
    # Predicting using the tuned model 
    Y_pred = model.predict(X_test)

    # Printing the classification report
    print(classification_report(Y_test.iloc[:, 1:].values, np.array([x[1:] for x in Y_pred]), target_names = category_names))
    
    # Printing mean accuracy
    accuracy = (Y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))

def save_model(model, model_filepath):
    """ This function saves trained model as Pickle file, to be loaded later.
    
    Input:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()