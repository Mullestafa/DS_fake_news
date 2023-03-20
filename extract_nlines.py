from pyarrow.parquet import ParquetFile
import pyarrow as pa 

pf = ParquetFile('fake_news_output.parquet') 
first_n_rows = next(pf.iter_batches(batch_size = 1,)) 
df = pa.Table.from_batches([first_n_rows]).to_pandas()

df.info()