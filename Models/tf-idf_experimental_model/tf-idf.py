# %%
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import pandas as pd
from torch import nn

# %%
def data_iterator(csv_file='E:/ML/DS_fake_news/fake_news_cleaned.csv'):
    data = pd.read_csv(csv_file, usecols=['content', 'type'], chunksize=2000)
    label_map = {'bias': 0,
                        'clickbait': 0,
                        'conspiracy': 0,
                        'fake': 1,
                        'hate': 1,
                        'junksci': 1,
                        'political': 0,
                        'reliable': 0,
                        'rumor': 0,
                        'satire': 0,
                        'unreliable': 1}
    for chunk in data:
        # throw away rows with missing type
        chunk = chunk.dropna(subset=['type'])
        # drop rows with 'unknown' type
        chunk = chunk[chunk['type'].isin(label_map.keys())]
        chunk['type'] = chunk['type'].map(label_map)
        yield chunk

# %%
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(n_features=2**20, stop_words='english')



# %%
data = data_iterator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
first = True
chunk_num = 0
for chunk in data:
    # transform the text to tf-idf
    try:
        tfidf = vectorizer.transform(chunk['content'])
    except:
        chunk_num += 1
        continue
    labels = chunk['type']
    # convert labels to tensor
    labels = torch.tensor(labels.values)


    # %%
    X_data = tfidf
    y_data = labels

    # %%
    # count each label
    label_count = Counter(y_data.numpy())

    # remove labels with less than 2 samples
    for label in label_count:
        if label_count[label] < 2:
            X_data = X_data[y_data != label]
            y_data = y_data[y_data != label]
    label_count = Counter(y_data.numpy())
    # check if any of the labels have 0 samples
    if len(label_count) < 2:
        chunk_num += 1
        continue
    # if the two classes are not balanced, skip chunk
    label_count = Counter(y_data.numpy())
    print(label_count)
    if label_count[0]/label_count[1] < 0.5 or label_count[0]/ label_count[1] > 2:
        chunk_num += 1
        continue


    print(Counter(y_data.numpy()))

    # %%
    # split the dataset using sklearn
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # %%
    # convert to coo matrix
    X_train = X_train.tocoo()
    X_test = X_test.tocoo()
    X_val = X_val.tocoo()

    # %%

    # convert to torch tensors
    X_train = torch.sparse_coo_tensor(torch.LongTensor([X_train.row, X_train.col]), torch.FloatTensor(X_train.data), X_train.shape).to(device)
    X_test = torch.sparse_coo_tensor(torch.LongTensor([X_test.row, X_test.col]), torch.FloatTensor(X_test.data), X_test.shape).to(device)
    X_val = torch.sparse_coo_tensor(torch.LongTensor([X_val.row, X_val.col]), torch.FloatTensor(X_val.data), X_val.shape).to(device)

    y_train = y_train.to(device)
    y_test = y_test.to(device)
    y_val = y_val.to(device)

    # %%
    # run once only

    if first:
        model = nn.Sequential(
                    nn.Linear(X_train.shape[1], 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, len(set(y_train))),
                    nn.LogSoftmax(dim=1)).to(device)
        # Define the loss
        criterion = nn.NLLLoss()
        # Forward pass, log  
        logps = model(X_train)
        # Calculate the loss with the logits and the labels
        loss = criterion(logps, y_train)
        loss.backward()
        # Optimizers need parameters to optimize and a learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        first = False

    # %%
    epochs = 20
    for e in range(epochs):
        optimizer.zero_grad()
        output = model.forward(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            log_ps = model(X_test)
            test_loss = criterion(log_ps, y_test)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y_test.view(*top_class.shape)
            test_accuracy = torch.mean(equals.float())
        
        print(f"Epoch {e+1}/{epochs}.. ",
                f"Train loss: {loss:.3f}.. ",
                f"Test loss: {test_loss:.3f}.. ",
                f"Test accuracy: {test_accuracy:.3f}")
    chunk_num += 1
    print(f'Finished chunk {chunk_num}')
    # save model every 10 chunks
    if chunk_num % 100 == 0:
        torch.save(model.state_dict(), f'./tf-idf-{chunk_num}.pth')
        print(f'Saved model at chunk {chunk_num}')
torch.save(model.state_dict(), f'./tf-idf-{finished}.pth')
print(f'Saved model at chunk {finished}')
