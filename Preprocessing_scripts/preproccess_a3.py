import pandas as pd
import time

# first lets run clean_text on the 'content' column
from cleantext import clean
def clean_text(s):
    return clean(s,lower=True,                     # lowercase text
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_numbers=True,               # replace all numbers with a special token
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_number="<NUM>",
        lang="en"                   
    )

# clean the text
def clean_column(series):
    return series.apply(clean_text)

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def tokenize_column(series):
    return series.apply(word_tokenize)
    
state = False

from nltk.corpus import stopwords
# removing generic stopwords
def remove_stopwords(series):
    stop_words = set(stopwords.words('english'))
    return series.apply(lambda x: [w for w in x if not w in stop_words])

# stemming the text
from nltk.stem import PorterStemmer
def stem_column(series):
    ps = PorterStemmer()
    return series.apply(lambda x: [ps.stem(w) for w in x])
    

# remove punctiuation
import string
def remove_punctuation(series):
    return series.apply(lambda x: [w for w in x if w not in string.punctuation])
    

import dask.dataframe as dd
import dask.bag as db
from dask.diagnostics import ProgressBar


def main(input, output):


    # read csv
    ddf = dd.read_csv(input, parse_dates=['scraped_at', 'inserted_at', 'updated_at'])

    # Drop the unnecessary columns
    ddf = ddf.drop(['Unnamed: 0', 'summary', 'source'], axis=1)

    # Fill the missing values in the 'authors' column with 'Unknown'
    ddf['authors'] = ddf['authors'].fillna('Unknown')

    # clean the content
    ddf['content'] = ddf['content'].map_partitions(clean_column)
    ddf['content'] = ddf['content'].map_partitions(tokenize_column)
    ddf['content'] = ddf['content'].map_partitions(remove_stopwords)
    #ddf['content'] = ddf['content'].map_partitions(stem_column)
    ddf['content'] = ddf['content'].map_partitions(remove_punctuation)
    with ProgressBar():
        ddf = ddf.compute()

    # save data to csv
    ddf.to_csv(output, index=False)
    


from os import cpu_count
print('cores available:', cpu_count())
# run from the command line with input and output files as arguments
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file')
    parser.add_argument('--output', help='output file', default='output.csv')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    
    main(input_file, output_file)