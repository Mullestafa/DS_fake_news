# DS_fake_news
This is the repository for the Fake News classifier Project, part of the Data Science course at University of Copehagen in 2023, conducted by group 57.
The different scripts have been seperated into their relevant directories. The preproccessing pipeline, will be found in the preproccessing folder,
meanwhile all the models, whether baseline, or advanced will be found in the models folder.


# necesary dependencies
The repository includes all code that is needed to reproduce the results from the project. Due to size constraints, there are several large files,
which are not included in the repository. This will often result in a file-not-found error when running the different notebooks.

Therefore we provide links to download several large files that where used during the project:

- Vocab.pkl: a pickled list of tuples each with a word and its frequency in the fake news corpus
  - https://sid.erda.dk/share_redirect/gLpZPm84dp
- fake_news_cleaned.csv: similar to the orginal corpus, but where corrupted rows have been removed
  - https://sid.erda.dk/share_redirect/CwrLUB0wA4
- fake_news_cleaned_filtered.csv: similar to the previous file on the list, but the data preproccesing has been applied to it.
  - https://sid.erda.dk/share_redirect/aCZi5cUPlm
  
  
Python Dependencies:
  - clean-text
  - scikit-learn
  - unicode
  - nltk
  - numpy
  - pandas
  - dask
  - dask[diagnostics]
  - matplotlib
  - gensim
  - pytorch
  - keras

NLTK Data Dependencies:
  - stopwords
  - punkt
  - wordnet
  
How to run:
Either run the preprocessing code under the preprocessing folder.
Run the baseline-models.ipynb under models with the three files from preprocessing in the folder: Vocab.pkl named as file.pkl, fake_news_cleaned.csv, and fake_news_cleaned_filtered.csv.
Run the advanced model comparing_embedding_and_CNN_model.ipynb under models with fake_news_cleaned.csv and fake_news_cleaned_filtered.csv.
