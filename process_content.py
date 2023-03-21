from os import cpu_count
from joblib import Parallel, delayed
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
    # parallelized operation
    return Parallel(n_jobs=cpu_count())(delayed(clean_text)(s) for s in series)

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def tokenize_column(series):
    # parallelized operation
    return Parallel(n_jobs=cpu_count())(delayed(word_tokenize)(s) for s in series)
    
state = False

from nltk.corpus import stopwords
# removing generic stopwords
def remove_stopwords(series):
    stop_words = set(stopwords.words('english'))
    # parallelized operation
    return Parallel(n_jobs=cpu_count())(delayed(lambda x: [w for w in x if not w in stop_words])(s) for s in series)

# lemmatizing the text
from nltk.stem import WordNetLemmatizer
def lemmatize_column(series):
    lemmatizer = WordNetLemmatizer()
    # parallelized operation
    return Parallel(n_jobs=cpu_count())(delayed(lambda x: [lemmatizer.lemmatize(w) for w in x])(s) for s in series)
    

# remove punctiuation
import string
def remove_punctuation(series):
    # parallelized operation
    return Parallel(n_jobs=cpu_count())(delayed(lambda x: [w for w in x if w not in string.punctuation])(s) for s in series)




def main(input, output):
    #csv file in chunks
    import pandas as pd

    chunk_size = 10000
    df = pd.read_csv(input, parse_dates=['scraped_at', 'inserted_at', 'updated_at'], chunksize=chunk_size)

    for i, chunk in enumerate(df):
    # Drop the unnecessary columns
        chunk = chunk['content']

        # clean the text
        chunk = clean_column(chunk)
        chunk = tokenize_column(chunk)
        chunk = remove_stopwords(chunk)
        chunk = lemmatize_column(chunk)
        chunk = remove_punctuation(chunk)

        # convert list back to series
        chunk = pd.Series(chunk)
        
        # save to file. Append if file exists, otherwise create new file
        if i == 0:
            chunk.to_csv(output, mode='w', index=False)
        else:
            chunk.to_csv(output, mode='a', header=False, index=False)

    # print progress
    print('processed {} rows'.format((i+1)*chunk_size))


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