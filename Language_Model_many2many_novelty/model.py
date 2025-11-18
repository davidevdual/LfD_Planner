import torch
from torch import nn
torch.device('cuda')

class Model(nn.Module):
    def __init__(self, dataset,_lstm_size,_embedding_dim,_num_layers,_dropout):
        super(Model, self).__init__();
        self.lstm_size = _lstm_size; #original: 128
        self.embedding_dim = _embedding_dim; #original: 128
        self.num_layers = _num_layers;
        self.dropout = _dropout;#dropout of 0.2 is good
        self.softmax = nn.Softmax(dim=1);# Softmax layer. 
        self.nllloss = nn.NLLLoss();# Cross-entropy Loss without the Softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1);# Log Softmax layer

        n_vocab = len(dataset.uniq_words);
        self.n_vocab = n_vocab;
        self.embedding = nn.Embedding(
            num_embeddings=self.n_vocab,
            embedding_dim=self.embedding_dim,
        );
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=self.dropout,#original"0.2
            batch_first=False,
        );
        self.fc = nn.Linear(self.lstm_size,self.n_vocab);

    def forward(self, x, prev_state):
        embed = self.embedding(x);
        output, state = self.lstm(embed,prev_state);# it should be: self.lstm(embed,prev_state)
        logits = self.fc(output);
        return logits,state;

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.lstm_size),
                torch.zeros(self.num_layers, batch_size, self.lstm_size));