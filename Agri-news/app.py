from flask import Flask, render_template
import requests
from dotenv import load_dotenv
import os


app = Flask(__name__)


load_dotenv() 

@app.route('/')
def home():
    api_key = os.getenv('NEWS_API_KEY')  
    url = 'https://newsapi.org/v2/everything'
    query = 'Agriculture'

    params = {
        'q': query,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'relevancy',

        'pageSize': 10
    }

    response = requests.get(url, params=params)
    print(response.status_code)  # Print the status code
    print(response.json())  # Print the JSON response

    news_data = response.json()

    if news_data['status'] == 'ok':
        articles = news_data['articles']
    else:
        articles = []

    return render_template('news.html', articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
