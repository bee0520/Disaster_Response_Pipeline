import sys
# import libraries

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
import joblib


#evluate
from sklearn.model_selection import train_test_split, GridSearchCV
#database
from sqlalchemy import create_engine

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    x=df['message']
    y=df.drop(columns=['id','message','original','genre'])
    return x,y,y.columns


def tokenize(text):
    ''' 
    words list: Processed text after tokenizing, remove stop words and      
    lemmatizing
     
    ''' 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(text).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)
    
    
    return clean_tokens


def build_model():
    '''
    Use RandomForest to build model with GridSearchCV
    Trained model after performing grid search
    
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
       ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
])
        
    
    parameters ={
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50] #pipeline.get_params().keys()
    } 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performance of each category target column
    
    model: train model
    X_test: test feature set
    Y_test: test target set
    category_names: target category names
    '''
    # Use model to predict
    Y_pred = model.predict(X_test)
    # Turn prediction into DataFrame
    y_pred = pd.DataFrame(Y_pred,columns=category_names)

    # For each category column, print performance
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(Y_test[col],y_pred[col]))

    accuracy = (Y_pred == Y_test.values).mean()
    print(accuracy)


def save_model(model, model_filepath):
    '''
    save as pickle file
    '''
    joblib.dump(model, model_filepath)


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