import pandas as pd
# check csv file for bad lines
chunksize = 10000
for i, chunk in enumerate(pd.read_csv('E:/ML/fake_news_data/news.csv/news_cleaned_2018_02_13.csv', chunksize=chunksize, on_bad_lines='warn', engine='python')):
    # try saving chunk to temp.csv
    print('row {}'.format(i*chunksize))
    try:
        # save to file. Append if file exists, otherwise create new file
        if i == 0:
            chunk.to_csv('temp.csv', index=False)
        else:
            chunk.to_csv('temp.csv', index=False, mode='a')
            
    # print error if chunk has bad lines
    except ValueError as e:
        print(e)
        print(chunk.info())
        print('bad chunk at row {}'.format(i*chunksize))
        break