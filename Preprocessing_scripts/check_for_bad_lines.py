bad_rows = []

# load in csv file with csv reader
import csv
with open('E:/ML/fake_news_data/news.csv/news_cleaned_2018_02_13.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        # check if first column is int and if there are 17 columns
        if not row[0].isdigit() or len(row) != 17:
            print('bad row at row {}'.format(i))
            bad_rows.append(i)

# save bad rows to parquet file
import pandas as pd
df = pd.DataFrame(bad_rows, columns=['bad_row_index'])
df.to_parquet('skip_list.parquet')
            
