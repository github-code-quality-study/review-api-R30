import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import pandas as pd
import json
from typing import Callable, Any
from datetime import datetime
import os
import uuid
from wsgiref.simple_server import make_server

# Download necessary NLTK data if not already present
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load reviews from CSV
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Valid locations
valid_locations = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", "Colorado Springs, Colorado",
    "Denver, Colorado", "El Cajon, California", "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California", "Phoenix, Arizona",
    "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        return sia.polarity_scores(review_body)

    def filter_and_sort_reviews(self, location=None, start_date=None, end_date=None):
        filtered_reviews = [
            review for review in reviews
            if (not location or review['Location'] == location) and
               (not start_date or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date) and
               (not end_date or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date)
        ]
        for review in filtered_reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
        return sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        try:
            if environ["REQUEST_METHOD"] == "GET":
                query_params = parse_qs(environ.get('QUERY_STRING', ''))
                location = query_params.get('location', [None])[0]
                start_date_str = query_params.get('start_date', [None])[0]
                end_date_str = query_params.get('end_date', [None])[0]

                start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None

                filtered_reviews = self.filter_and_sort_reviews(location, start_date, end_date)
                response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

                start_response("200 OK", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            elif environ["REQUEST_METHOD"] == "POST":
                request_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_size).decode('utf-8')
                post_data = parse_qs(request_body)
                review_body = post_data.get('ReviewBody', [None])[0]
                location = post_data.get('Location', [None])[0]

                if not review_body or not location:
                    raise ValueError("Missing ReviewBody or Location")

                if location not in valid_locations:
                    raise ValueError("Invalid location")

                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "Location": location,
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "ReviewBody": review_body,
                    "sentiment": self.analyze_sentiment(review_body)
                }

                reviews.append(new_review)
                response_body = json.dumps(new_review, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

        except Exception as e:
            response_body = json.dumps({"error": str(e)}).encode('utf-8')
            start_response("400 Bad Request", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()