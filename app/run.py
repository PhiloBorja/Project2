import json
import plotly
import pandas as pd
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.plots_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # create visuals
    plots = []
    
    
    # 1. plots: We plot the messages per categories
    
     
    df_categories = pd.DataFrame(df.iloc[:, 5:].sum())

    print(df)
    print(df_categories)
    
    fig = px.bar(df_categories, y='count', x=df_categories, text_auto='.2s',
                title="Messages per categories")
                labels = {'index':' Categories', 'count' : '# Messages'}
        
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    plots.append(fig.to_plotly_json())
    
    
    # 2. plots: We plot the message genres per categories
 
    df_categories_genre = df.iloc[:, 4:].groupby('genre').sum().T

    fig = go.Figure(data=[
        go.Bar(df_categories_genre, x=df_categories_genre, y=['direct', 'news', 'social'])])

    fig.update_layout(barmode='stack')

    plots.append(fig.to_plotly_json())
    
    
   
    # 3 plots: We plot the the number of categories per messasge

    df_categories_message = pd.DataFrame()
    
    df_categories_message['# categories'] = df.iloc[:, 5:].sum(axis=1)

    fig = px.bar(df_categories_message, x='# categories')

    plots.append(fig.to_plotly_json())
    
   
      
    # encode plotly plots in JSON
    ids = ["plots-{}".format(i) for i, _ in enumerate(plots)]
    plotsJSON = json.dumps(plots, cls=plotly.utils.PlotlyJSONEncoder)
    print('plotsJSON:')
    print(plotsJSON)
    
    # render web page with plotly plots
    return render_template('master.html', ids=ids, plotsJSON=plotsJSON)
    
    
    
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()