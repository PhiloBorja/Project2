import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath) 
    df = pd.read_sql_table('message', engine)
    
    # We take the values of the column 'message'    
    X = df.message.values
    
    # We take the values from the fourth column ('related','request','offer',...)
    Y = df.iloc[:, 4:].values    
    category_names = df.iloc[:, 4:].columns.values  
    
    return X, Y, category_names
   # pass


def tokenize(text):
    
    # Normalize and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9_]", " ", text.lower())
    
    # Write a tokenization function to process the text data
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for t in tokens:
        clean_tok = lemmatizer.lemmatize(t).lower().strip()
        clean_tokens.append(clean_tok)
        
    # Word that are not in the stopword that we download from NLTK. 
    clean_tokens = [word for word in clean_tokens if word not in stop_words]   
    
    return clean_tokens 
   # pass


def build_model():
    
     parameters = { 
        # GridSearchCV implements a “fit” and a “score” method. 
        # It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used
          'min_samples_leaf': 2,
          'min_samples_split': 2
          'max_depth': None,
          'max_features': None,}
      
    # If we want train with optimized parameters, optimized_parameters = True. This is more computing and timing consuming
    optimized_parameters = False
    
    # CountVectorizer,TfidfTransformer, MultiOutputClassifier and RandomForestClassifier from sklearn.feature_extraction.text
    if optimized_parameters:
        return Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(**best_params)))
            ])
    else:
        return Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
    
   # pass


def evaluate_model(model, X_test, Y_test, category_names):
    
    # Model
    Y_pred = model.predict(X_test)
    
    # classification report from sklearn.metrics 
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
    
    #pass


def save_model(model, model_filepath):
    
    # We open the file, we write to the file and we wirte in binary mode (0 and 1). We need it for the pickle
    with open(model_filepath, 'wb') as handle:
        
        # We write the pickled representation of the object to the open file. We use the highest protocol version available.
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
   # pass


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