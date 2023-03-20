
#csv file in chunks
import pandas as pd

df = pd.read_csv('fake_news_cleaned.csv', parse_dates=['scraped_at', 'inserted_at', 'updated_at'], chunksize=100000)

# save each chunk to hdf file
for i, chunk in enumerate(df):

    # Drop the unnecessary columns
    chunk = chunk.drop(['Unnamed: 0', 'summary', 'source'], axis=1)

    # Fill the missing values in the 'authors' column with 'Unknown'
    chunk['authors'] = chunk['authors'].fillna('Unknown')

    # Use the str accessor to vectorize string operations on 'url' and 'content' columns
    chunk['url_length'] = chunk['url'].str.len()
    chunk['content_length'] = chunk['content'].str.len()

    # save to file. Append if file exists, otherwise create new file using format='table'
    if i == 0:
        chunk.to_hdf('fake_news_cleaned.hdf', key='df', mode='w')
    else:
        chunk.to_hdf('fake_news_cleaned.hdf', key='df', mode='a')

    # print progress
    print('processed {} rows'.format((i+1)*100000))