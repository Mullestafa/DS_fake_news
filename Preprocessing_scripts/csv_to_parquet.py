import pandas as pd
import time


import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def main(input, output):
    # read first 20 rows of csv file
    df = pd.read_csv(input, nrows=20)
    # print line 17
    print(df.iloc[17])


    
    ddf = dd.read_csv(input, low_memory=False)
    print(ddf)

    # process the data in chunks
    with ProgressBar():
        # do something with the data just to check it works
        ddf.map_partitions(lambda df: print(df.info())).compute()

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




""" import pandas as pd
import time


import dask.dataframe as dd
from dask.diagnostics import ProgressBar
def main(input, output):
    import pandas as pd
    # check csv file for bad lines
    chunksize = 10000
    for i, chunk in enumerate(pd.read_csv(input, chunksize=chunksize, on_bad_lines='warn', engine='python')):
        # try saving chunk to temp.csv
        print('row {}'.format(i*chunksize))
        try:
            # save to file. Append if file exists, otherwise create new file
            if i == 0:
                chunk.to_csv(output, index=False)
            else:
                chunk.to_csv(output, index=False, mode='a')
                
        # print error if chunk has bad lines
        except ValueError as e:
            print(e)
            print(chunk.info())
            print('bad chunk at row {}'.format(i*chunksize))
            break

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
    
    main(input_file, output_file) """