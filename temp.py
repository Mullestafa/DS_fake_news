import pandas as pd
# load csv reader
import csv
import swifter



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
    return [w for w in series if not w in stop_words]
    

# remove punctiuation
import string
def remove_punctuation(series):
    return [w for w in series if w not in string.punctuation]

def main(input, output):

    # open pandas csv file in chunks
    df = pd.read_csv(input, parse_dates=['scraped_at', 'inserted_at', 'updated_at'], chunksize=1000)

    # process file in chunks
    for i, chunk in enumerate(df):
        # clean the text with parallel processing
        chunk['content'] = chunk['content'].swifter.apply(clean_text)
        # tokenize the text with parallel processing
        chunk['content'] = chunk['content'].swifter.apply(word_tokenize)
        # remove stopwords with parallel processing
        chunk['content'] = chunk['content'].swifter.apply(remove_stopwords)
        # remove punctuation with parallel processing
        chunk['content'] = chunk['content'].swifter.apply(remove_punctuation)

        # write to csv file
        if i == 0:
            chunk.to_csv(output, index=False)
        else:
            chunk.to_csv(output, mode='a', header=False, index=False)

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
