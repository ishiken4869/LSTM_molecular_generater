
# %%
from LSTM_pramater import MAX_EPOCHS
from LSTM_pramater import BATCH_SIZE
from LSTM_pramater import SEQUENCE_LENGTH
from LSTM_pramater import LEARNING_RATE
from LSTM_pramater import NROWS
from LSTM_pramater import URL


# %%
import torch
import pandas as pd
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, url, smiles_col, sequence_length=4):
        self.url = url
        self.smiles_col = smiles_col
        self.sequence_length = sequence_length
        self.smiles = []
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_table(self.url, nrows=NROWS)
        self.smiles = list(train_df[self.smiles_col])
        text = train_df[self.smiles_col].str.cat(sep=' ')
        text = "".join(text.split(' '))
        return [text[i] for i in range(len(text))]

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )

# %%
import torch

class LSTM_Generator(torch.nn.Module):
    def __init__(self, dataset):
        super(LSTM_Generator, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        self.embedding = torch.nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = torch.nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


# %%
import numpy as np

def predict(dataset, model, text, next_words=50):
    words = [text[i] for i in range(len(text))]
    model.eval()

    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return trim_smiles("".join(words))

# %%
from rdkit import Chem

def trim_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        mol = False
    while not mol:
        if len(smile) == 0: break
        smile = smile[:-1]
        try:
            mol = Chem.MolFromSmiles(smile)
        except:
            mol = False
    return smile




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: " + str(device) + '\n')

dataset = Dataset(url=URL, smiles_col="SMILES_COL")
model = LSTM_Generator(dataset)
model.load_state_dict(torch.load('model' + '_' + 'M' + str(MAX_EPOCHS) + '_' + 'B' + str(BATCH_SIZE) + '_' + 'L' + str(LEARNING_RATE) + '_' + 'N' + str(NROWS) + '.pt', map_location=torch.device('cpu')))
#model = model.to(device)

print("\ntext='C'")
print(predict(dataset, model, text='C'))
print("\ntext='N'")
print(predict(dataset, model, text='N'))
print("\ntext='O'")
print(predict(dataset, model, text='O'))

# %%
'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
axes[0].plot(losses, label="Loss")
axes[0].grid()
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[1].plot(losses, label="Loss")
axes[1].grid()
axes[1].set_yscale('log')
axes[1].set_xlabel("epoch")
axes[1].legend()
plt.show()
'''



