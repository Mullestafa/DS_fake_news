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

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
    
state = False

from nltk.corpus import stopwords
# removing generic stopwords
def remove_stopwords(string):
    stop_words = set(stopwords.words('english'))
    return [w for w in string if not w in stop_words]

# lemmatizing the text
from nltk.stem import WordNetLemmatizer
def lemmatize(string):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in string]

    

# remove punctiuation
import string
def remove_punctuation(string):
    return [w for w in string if w not in string.punctuation]
    

import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def main(input, output):


    # read csv
    ddf = dd.read_csv(input, parse_dates=['scraped_at', 'inserted_at', 'updated_at'], nrows=100000)
    ddf = ddf.repartition(npartitions='auto', partition_size=1000)
    # Drop the unnecessary columns
    ddf = ddf.drop(['Unnamed: 0', 'summary', 'source'], axis=1)

    # Fill the missing values in the 'authors' column with 'Unknown'
    ddf['authors'] = ddf['authors'].fillna('Unknown')

    # clean the content
    ddf['content'] = ddf['content'].map_partitions(clean_text)
    ddf['content'] = ddf['content'].map_partitions(word_tokenize)
    ddf['content'] = ddf['content'].map_partitions(remove_stopwords)
    ddf['content'] = ddf['content'].map_partitions(lemmatize)
    ddf['content'] = ddf['content'].map_partitions(remove_punctuation)
    with ProgressBar():
        ddf = ddf.compute()

    # save data to csv
    ddf.to_parquet(output, index=False)
    


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