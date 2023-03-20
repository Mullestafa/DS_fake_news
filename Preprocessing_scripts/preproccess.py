import pandas as pd

# split the data in chunks and run in parallel
from joblib import Parallel, delayed
from os import cpu_count

# run in parallel
def run_parallel(df, n_jobs, func):
    # call every element in the chunks in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(func)(element) for element in df)
    return results

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
def clean_column(df):
    n_jobs = cpu_count()
    results = run_parallel(df, n_jobs, clean_text)
    # replace column with cleaned text
    return results

from nltk.tokenize import word_tokenize
# tokenize the text. run in parallel
def tokenize_column(df):
    # run the function on the data
    n_jobs = cpu_count()
    results = run_parallel(df['content'], n_jobs, word_tokenize)
    
    return results

from nltk.corpus import stopwords
# removing generic stopwords
def remove_stopwords(df):
    stop_words = set(stopwords.words('english'))

    # remove stopwords from the text
    def r(s):
        return [w for w in s if not w in stop_words]

    # run the function on the df
    n_jobs = cpu_count()
    results = run_parallel(df, n_jobs, r)

    return results

# stemming the text
from nltk.stem import PorterStemmer
def stem_column(df):
    # create a stemmer
    ps = PorterStemmer()

    # stem the text
    def stem(s):
        return [ps.stem(w) for w in s]

    # run the function on the df
    n_jobs = cpu_count()
    results = run_parallel(df, n_jobs, stem)
    return results

# remove punctiuation
import string
def remove_punctuation(df):
    # remove punctuation
    def remove_punct(s):
        return [w for w in s if w not in string.punctuation]

    # run the function on the df
    n_jobs = cpu_count()
    results = run_parallel(df, n_jobs, remove_punct)
    return results

def main(input, output):
    # read the data
    df = pd.read_csv(input)
    # clean the text
    df['content'] = clean_column(df['content'])
    # tokenize the text
    df['content'] = tokenize_column(df)
    # remove stopwords
    df['content'] = remove_stopwords(df['content'])
    # stem the text
    #df['content'] = stem_column(df['content'])
    # remove punctuation
    df['content'] = remove_punctuation(df['content'])
    # save the data
    df.to_csv(output, index=False)

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