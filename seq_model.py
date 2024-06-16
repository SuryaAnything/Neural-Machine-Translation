import torch
import torch.nn as nn
import random
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional = True)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        encoder_states, (hidden, cell) = self.rnn(embedded)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers)
        self.energy = nn.Linear(hidden_size*3, 1) 
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0) 

    def forward(self, src, encoder_states, hidden, cell): 
        src = src.unsqueeze(0)
        embedded = self.dropout(self.embedding(src))
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2))) 
        attention = self.softmax(energy) 
        attention = attention.permute(1, 2 ,0) 
        encoder_states = encoder_states.permute(1, 0, 2) 
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2) 
        rnn_input = torch.cat((context_vector, embedded), dim = 2) 

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell)) 
        predictions = self.fc(outputs.squeeze(0))
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_states, hidden, cell = self.encoder(src)
        x = trg[0]
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            if random.random() < teacher_forcing_ratio:
                x = trg[t]
            else:
                x = output.argmax(1)
        return outputs