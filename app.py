from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy as database
from sqlalchemy import create_engine
from datetime import datetime
import joblib
import pickle
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import re
import nltk
import glob
import os
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///review.db'
db = SQLAlchemy(app)


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                          text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


def tokenizer(text):
    return text.split()


porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


stop = stopwords.words('english')


def model_fit(x, y):
    vectorizer = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf': [False],
                   'vect__norm': [None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]}]

    model = LogisticRegression(random_state=0, solver='liblinear')

    pipe = Pipeline([('vect', vectorizer),
                         ('clf', model)])

    gs_model = GridSearchCV(pipe, param_grid,
                               scoring='accuracy',
                               cv=5, verbose=2,
                               n_jobs=-1)
    print("start train......")
    gs_model.fit(x, y)

    print("TRAIN FINISHED!!!!")
    n = 1

    while True:
        if os.path.exists(f'models/{n}_model.pkl'):
            n += 1
            continue
        else:
            break

    pkl_filename = f"models/{n}_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(gs_model, file)

    print("Save model successfully")
    return "Model trained successfully"


class Reviews(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)


def get_model():
    list_of_files = glob.glob('models/*')  # * means all if need specific format then *.csv
    last_model = min(list_of_files, key=os.path.getctime)

    filename = last_model
    model = joblib.load(filename)
    return model


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        try:
            sentiment = request.form['sentiment']
        except:
            sentiment = None

        if request.form['button'] == 'predict':
            if get_model().predict([review]) == '0':
                result = 'Negative'
            else:
                result = 'Positive'
            return render_template('index.html', noclear=review, result=result)

        elif request.form['button'] == 'reset':
            return render_template("index.html")

        elif request.form['button'] == 'to_db':

            write_db = Reviews(review=review, sentiment=sentiment)
            try:
                db.session.add(write_db)
                db.session.commit()
                return render_template("index.html")
            except:
                return "Error DataBase"

            return render_template("index.html")

        elif request.form['button'] == 'train':
            train_df = pd.read_csv('train_dataset_5k.csv', on_bad_lines='skip', sep=',', usecols=['Review', 'Rating'], encoding='utf-8')
            train_df['Review'] = train_df['Review'].apply(preprocessor)
            data = Reviews.query.all()
            df_from_db = pd.DataFrame(columns=['Review', 'Rating'])

            for i in data:
                df_from_db.loc[len(df_from_db.index)] = [i.review, i.sentiment]

            df_from_db.to_csv('df_from_db.csv', encoding='utf-8')
            dfdb = pd.read_csv('df_from_db.csv', on_bad_lines='skip', encoding='utf-8', usecols=['Review', 'Rating'])

            train_df = pd.concat([train_df, dfdb], ignore_index=True)

            train_df['Review'] = train_df['Review'].apply(preprocessor)

            x_train = train_df.loc[:2500, 'Review'].values
            y_train = train_df.loc[:2500, 'Rating'].values
            x_test = train_df.loc[2500:, 'Review'].values
            y_test = train_df.loc[2500:, 'Rating'].values

            model_fit(x_train, y_train)
            #print('score - ', get_model().score(x_test, y_test))

            return render_template("index.html")

    else:
        return render_template("index.html")


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=8000)