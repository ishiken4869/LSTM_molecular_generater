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
import torch
import numpy as np

def train(dataset, model):
    losses = []
    model.train()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        state_h, state_c = model.init_state(SEQUENCE_LENGTH)
        state_h = state_h.to('cuda:0')
        state_c = state_c.to('cuda:0')
        total_loss = 0

        for batch, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x.to('cuda:0'), (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y.to('cuda:0'))
            total_loss += loss.item()

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
        '''
        print("Epoch: {}, Loss: {:.3f}, Generated SMILES: {}".format(
            epoch+1, 
            total_loss,
            get_best_smiles(dataset, model)
            )
        )
        '''

        print("Epoch: {}, Loss: {:.3f}".format(
            epoch+1, 
            total_loss / (NROWS/BATCH_SIZE),
            #get_best_smiles(dataset, model)
            )
        )

        total_loss = total_loss
        losses.append(total_loss)
    return losses

# %%
import random 

def get_best_smiles(dataset, model, next_words=100, max_trial=10, start_length=3):
    best_smile = ""
    for trial in range(max_trial):
        starting_text = random.choice(dataset.smiles)[:start_length]
        smile = predict(dataset, model, text=starting_text, next_words=next_words)
        if len(best_smile) < len(smile):
            best_smile = smile
    return best_smile

# %%
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
    mol = Chem.MolFromSmiles(smile)
    while not mol:
        if len(smile) == 0: break
        smile = smile[:-1]
        mol = Chem.MolFromSmiles(smile)
    return smile

# %%
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: " + str(device) + '\n')

print("MAX_EPOCHS:" + str(MAX_EPOCHS))
print("BATCH_SIZE:" + str(BATCH_SIZE))
print("SEQUENCE_LENGTH:" + str(SEQUENCE_LENGTH))
print("LEARNING_RATE:" + str(LEARNING_RATE))
print("NROWS:" + str(NROWS))

start = time.time()


dataset = Dataset(url=URL, smiles_col="SMILES_COL")
model = LSTM_Generator(dataset)
model = model.to(device)
losses = train(dataset, model)
losses = losses

end = time.time()
print("time:" + str(end - start))

model = model.to('cpu')
torch.save(model.state_dict(), 'model' + '_' + str(BATCH_SIZE) + '_' + str(NROWS) + '.pt')



# %%
import matplotlib.pyplot as plt

losses = [n / (NROWS/BATCH_SIZE) for n in losses]

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
#plt.show()

plt.savefig("losses_graph" + '_' + "MAX_EPOCHS" + str(MAX_EPOCHS) + '_' + "BATCH_SIZE" + str(BATCH_SIZE) + '_' + "NROWS" + str(NROWS) + ".png")


