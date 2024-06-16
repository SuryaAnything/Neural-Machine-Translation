import torch
import spacy
import torch.nn as nn
import torch.optim as optim
import os
from dataloder import create_dataset
from torchtext.data import Field, Example, Dataset, TabularDataset, BucketIterator
import seq_model
from utils import calculate_bleu, convert_to_sentences, translate_sentence


SRC_LANG = "de_core_news_sm"  # German model
TRG_LANG = "en_core_web_sm"   # English model

spacy_src = spacy.load(SRC_LANG)
spacy_eng = spacy.load(TRG_LANG)

# Load Spacy tokenizers
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Define tokenization functions
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Define the fields
SRC_FIELD = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG_FIELD = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# Paths to the dataset files (assuming they are in the current working directory)
<<<<<<< HEAD
train_de_path = './data/en-de/train.de'
train_en_path = './data/en-de/train.en'
val_de_path = './data/en-de/val.de'
val_en_path = './data/en-de/val.en'
test_de_path = './data/en-de/test.de'
test_en_path = './data/en-de/test.en'
=======
train_de_path = 'train.de'
train_en_path = 'train.en'
val_de_path = 'val.de'
val_en_path = 'val.en'
test_de_path = 'test.de'
test_en_path = 'test.en'
>>>>>>> f68d07a6a34fb64101c8e0196536d92300038c94

# Create the dataset fields
fields = [('src', SRC_FIELD), ('trg', TRG_FIELD)]

# Create examples for train, validation, and test datasets
train_examples = create_dataset(train_de_path, train_en_path, fields)
valid_examples = create_dataset(val_de_path, val_en_path, fields)
test_examples = create_dataset(test_de_path, test_en_path, fields)

# Initialize the datasets
train_data = Dataset(train_examples, fields)
valid_data = Dataset(valid_examples, fields)
test_data = Dataset(test_examples, fields)

# Build vocabulary
SRC_FIELD.build_vocab(train_data, min_freq=2)
TRG_FIELD.build_vocab(train_data, min_freq=2)


# Create iterators for the datasets
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=32,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Verify the dataset
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(vars(train_data.examples[0]))

print("SRC vocabulary:", SRC_FIELD.vocab.itos)
print("TRG vocabulary:", TRG_FIELD.vocab.itos)

num_epochs = 100
learning_rate = 3e-4
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(SRC_FIELD.vocab)
output_size = len(TRG_FIELD.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1

enc_dropout = 0.5
dec_dropout = 0.5

step = 0
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

encoder_net = seq_model.Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout)
decoder_net = seq_model.Decoder(output_size, decoder_embedding_size, hidden_size, num_layers, dec_dropout)
pad_idx = TRG_FIELD.vocab.stoi[TRG_FIELD.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
model = seq_model.Seq2Seq(encoder_net, decoder_net, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
checkpoint_path = "model_checkpoint.pth"
load = True
if os.path.exists(checkpoint_path) and load:
    print("loaded")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    losses = checkpoint["losses"]

len(losses)


# Define the initial value for teacher forcing ratio
initial_teacher_forcing_ratio = 0.6
scheduled_sampling_decay = 0.01
for epoch in range(num_epochs):
    # Calculate the current teacher forcing ratio for this epoch
    teacher_forcing_ratio = initial_teacher_forcing_ratio / (1 + scheduled_sampling_decay * epoch)
    teacher_forcing_ratio = max(teacher_forcing_ratio, 0.0)  # Ensure the ratio doesn't go negative

    print(f"Epoch {epoch+1} / {num_epochs}, Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}")

    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses": losses
    }
    torch.save(checkpoint, checkpoint_path)

    model.train()
    for batch_idx, batch in enumerate(test_iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        optimizer.zero_grad()

        # Pass the teacher forcing ratio to the model
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        output_dim = output.shape[2]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if step % 10 == 0:
            print(f"Loss = {loss.item():.4f}")
        step = step + 1
        
        
import matplotlib.pyplot as plt
plt.plot(losses)

print(len(losses))

# Test the model
model.eval()
bleu_score = calculate_bleu(test_iterator, model, TRG_FIELD, device)
print(f'BLEU score: {bleu_score*100:.2f}')

# Convert test_data to normal German sentences
german_sentences = convert_to_sentences(test_data, 'src')
english_sentences = convert_to_sentences(test_data, 'trg')
print(english_sentences[1])


i = 0
for sentence in german_sentences:
  translated = translate_sentence(model, sentence, SRC_FIELD, TRG_FIELD, device)
  filtered_output = [word for word in translated if word not in ["[", "]", "<eos>", "'", ",", "."]]
  output_sentence = " ".join(filtered_output)
  print("German : ", sentence)
  print("English : ", english_sentences[i])
  i+=1
  print("Translated :", output_sentence)
  print()
  if(i==25):
    break

