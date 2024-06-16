from torchtext.data.metrics import bleu_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import spacy

def calculate_bleu(data, model, trg_field, device):
    targets = []
    outputs = []

    for example in data:
        src = example.src.to(device)
        trg = example.trg.to(device)

        output = model(src, trg, teacher_forcing_ratio=0) # No teacher forcing during testing
        output = output.argmax(dim=-1)

        # Convert token indices to tokens
        trg_tokens = [trg_field.vocab.itos[idx.item()] for idx in trg[1:].view(-1)]
        output_tokens = [trg_field.vocab.itos[idx.item()] for idx in output.squeeze(0)[1:].view(-1)]

        # Cut off <eos> token
        trg_tokens = trg_tokens[:-1]
        output_tokens = output_tokens[:-1]

        targets.append([trg_tokens])
        outputs.append(output_tokens)

    bleu = bleu_score(outputs, targets)
    return bleu

# Function to convert tokenized examples to normal sentences
def convert_to_sentences(dataset, field):
    sentences = []
    for example in dataset.examples:
        tokens = getattr(example, field)
        sentence = " ".join(tokens)
        sentences.append(sentence)
    return sentences

def translate_sentence(model, sentence, german, english, device, max_length=50):


    # Load german tokenizer
    spacy_ger = spacy.load("de_core_news_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_states, hidden, cell)
            best_guess = output.argmax(1).item()



        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

        outputs.append(best_guess)

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]