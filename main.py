# Imports the Google Cloud client library + other necessary libraries
from google.cloud import language
from google.cloud.language import types
from google.cloud.language import enums
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
# instantiate a client
client = language.LanguageServiceClient.from_service_account_json \
    ("/Users/phoenix/Downloads/My First Project-4361e522b85a.json")


class TextObject:
    def __init__(self,url,language): # initialize our TextObject
        self.url = url
        self.language = language # support will be added later for this feature -> multilingual sentiment analysis

    def extractText(self):
        r1 = requests.get(self.url)
        coverpage = r1.content

        text_data = BeautifulSoup(coverpage,'html.parser')

        paragraph_text = [p.text for p in text_data.find_all('p')]
        df = pd.DataFrame({"text": paragraph_text,"score": np.nan,"magnitude": np.nan})

        return df


    def findSentiment(self,df):
        dfString = [d for d in df["text"]]
        # iterate through each paragraph of text
        for index,s in enumerate(dfString):
            document = types.Document(
                content = s,
                type = enums.Document.Type.PLAIN_TEXT)

            # Detects the sentiment of the text
            sentiment = client.analyze_sentiment(document = document).document_sentiment
            df["score"][index] = sentiment.score
            df["magnitude"][index] = sentiment.magnitude
        return df


if __name__ == "__main__":
    urls = ["https://www.nytimes.com/2016/05/01/magazine/what-happened-to-worcester.html",
            "https://clarknow.clarku.edu/2020/09/29/will-mahan-23-crafts-history-lessons-for-a-new-generation/",
            "https://www.theguardian.com/world/2020/sep/30/north-carolina-university-duke-coronavirus",
            "https://cooking.nytimes.com/recipes/1021483-salt-baked-new-potatoes-with-pink-peppercorn-butter?action"
            "=click&region=Sam%20Sifton%27s%20Suggestions&rank=1 "
            ]

    plt.figure(figsize = (10,6))
    for index,website in enumerate(urls): # iterate through the sentiment data for each webpage
        t = TextObject(website,'english')
        text = t.extractText()
        text = t.findSentiment(text)
        print("The average sentiment(positive/negative, +1/-1) is : {}".format(np.mean(text["score"])))
        print("The average magnitude(0/+inf) is : {}".format(np.mean(text["magnitude"])))
        print("\n")

        plt.figure(figsize = (10,6))
        plt.ylim(-1, 8)

        x_vals_scores = list(range(len(text["score"])))
        y_vals_scores = text["score"]
        plt.plot(x_vals_scores, y_vals_scores, label="positive/negative sentiment")

        y_vals_magnitude = text["magnitude"]
        plt.plot(x_vals_scores, y_vals_magnitude, label="magnitude of sentiment")

        plt.xlabel("From beginning to end of article")
        plt.ylabel("Magnitude & Score")

        plt.legend('best', fontsize = 15)

        plt.show()
