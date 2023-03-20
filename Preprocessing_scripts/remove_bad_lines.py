import pandas as pd
# load csv reader
import csv

def main(input, output):
    # read skiprows from parquet file
    skiprows = pd.read_parquet('skip_list.parquet')
    # convert skiprows to list
    skiprows = skiprows['bad_row_index'].tolist()

    print(skiprows)
    

    # read csv file
    df = pd.read_csv(input, skiprows=skiprows, chunksize=100000)

    # save each chunk to csv file
    for i, chunk in enumerate(df):

        # remove where id is not int
        chunk = chunk[chunk['id'].apply(lambda x: isinstance(x, int))]

        # save to file. Append if file exists, otherwise create new file
        if i == 0:
            chunk.to_csv(output, index=False)
        else:
            chunk.to_csv(output, index=False, mode='a')
        print('processed {} rows'.format((i+1)*100000))
    




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
