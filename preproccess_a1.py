import pandas as pd
import swifter

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
    # read the data
    df = pd.read_csv(input)

    # clean the text
    df['content'] = df['content'].swifter.apply(clean_text)

    # tokenize the text
    df['content'] = df['content'].swifter.apply(word_tokenize)

    # remove stopwords
    df['content'] = df['content'].swifter.apply(remove_stopwords)

    # stem the text
    #df['content'] = df['content'].swifter.apply(stem_column)

    # remove punctuation
    df['content'] = df['content'].swifter.apply(remove_punctuation)

    # save the data with parquet
    df.to_parquet(output)

# run from the command line with input and output files as arguments
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file')
    parser.add_argument('--output', help='output file', default='output.parquet')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    
    main(input_file, output_file)