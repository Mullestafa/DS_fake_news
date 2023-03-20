import pandas as pd
import swifter
from joblib import Parallel, delayed
import time


# run in parallel
def run_parallel(df, n_jobs, func):
    # call every element in the chunks in parallel
    results = Parallel(n_jobs=32)(delayed(func)(element) for element in df)
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
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# tokenize the text. run in parallel
def tokenize_column(df):
    # run the function on the data
    n_jobs = cpu_count()
    results = run_parallel(df['content'], n_jobs, word_tokenize)
    
    return results

from nltk.corpus import stopwords
# removing generic stopwords
def remove_stopwords(row):
    stop_words = set(stopwords.words('english'))

    # remove stopwords from the text
    return [w for w in row if not w in stop_words]


# stemming the text
from nltk.stem import PorterStemmer
def stem_column(row):
    # create a stemmer
    ps = PorterStemmer()
    return [ps.stem(w) for w in row]

# remove punctiuation
import string
def remove_punctuation(row):
    # remove punctuation
    
    return [w for w in row if w not in string.punctuation]

def main(input, output):
    print('starting to load data')
    # read the data in chunks
    start_time = time.time()
    chunks = pd.read_csv(input, chunksize=10000,error_bad_lines=False, warn_bad_lines=True)
    end_time = time.time()

    print('data loaded')
    print("Execution time:", end_time - start_time, "seconds")

    # process the data in chunks
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}")
        # clean the text
        start_time = time.time()
        chunk['content'] = clean_column(chunk['content'])
        end_time = time.time()

        print('finished cleaning')
        print("Execution time:", end_time - start_time, "seconds")

        # tokenize the text
        start_time = time.time()
        chunk['content'] = tokenize_column(chunk)
        end_time = time.time()
        print('finished tokenizing')
        print("Execution time:", end_time - start_time, "seconds")

        # remove stopwords
        chunk['content'] = chunk['content'].swifter.apply(remove_stopwords)

        print('finished removing stopwords')

        # stem the text
        #chunk['content'] = chunk['content'].swifter.apply(stem_column)

        # remove punctuation
        chunk['content'] = chunk['content'].swifter.apply(remove_punctuation)

        print('finished removing punctuation')
        chunk = chunk.drop('Unnamed: 0', axis=1)


        # save the data
        if i == 0:
            chunk.to_parquet(output, index=None, engine='fastparquet')
        elif i == 11:
            chunk.to_csv('temp.csv')
        else:
            chunk.to_parquet(output, index=None, engine='fastparquet', append=True)

    print('finished processing data')
    
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