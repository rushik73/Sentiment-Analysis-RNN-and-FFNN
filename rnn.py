import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # Step 1: Obtain hidden layer representation
        output, hidden = self.rnn(inputs)
        # 'output' shape: (seq_len, batch_size, hidden_size)

        # Step 2: Pass all outputs through the linear layer
        output = self.W(output)  # Shape: (seq_len, batch_size, num_classes)

        # Step 3: Sum over the sequence length
        output = torch.sum(output, dim=0)  # Shape: (batch_size, num_classes)

        # Step 4: Obtain probability distribution
        predicted_vector = self.softmax(output)  # Shape: (batch_size, num_classes)

        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    # Load pre-trained word embeddings (e.g., GloVe, Word2Vec)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    # Define the model
    input_dim = next(iter(word_embedding.values())).shape[0]  # Get embedding dimension
    model = RNN(input_dim, args.hidden_dim)

    # Choose an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop parameters
    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]

                if len(vectors) == 0:
                    continue  # Skip empty inputs

                # Convert list of NumPy arrays to a single NumPy array
                vectors = np.stack(vectors)  # Shape: (seq_len, embedding_dim)

                # Convert the NumPy array to a PyTorch tensor
                vectors = torch.tensor(vectors, dtype=torch.float32)

                # Reshape the tensor
                vectors = vectors.view(len(vectors), 1, -1)  # Shape: (seq_len, batch_size=1, input_dim)

                # Move tensors to the configured device
                vectors = vectors.to(device)
                gold_label_tensor = torch.tensor([gold_label], device=device)

                # Forward pass
                output = model(vectors)

                # Compute loss
                example_loss = model.compute_Loss(output, gold_label_tensor)

                # Get predicted label
                predicted_label = torch.argmax(output, dim=1).item()

                correct += int(predicted_label == gold_label)
                total += 1

                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            if loss is not None:
                loss = loss / minibatch_size
                loss_total += loss.item()
                loss_count += 1
                loss.backward()
                optimizer.step()

        if loss_count > 0:
            average_loss = loss_total / loss_count
        else:
            average_loss = 0

        print(f"Average Training Loss: {average_loss:.4f}")
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {:.4f}".format(epoch + 1, correct / total))
        training_accuracy = correct / total

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        print("Validation started for epoch {}".format(epoch + 1))

        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]

                if len(vectors) == 0:
                    continue  # Skip empty inputs

                vectors = np.stack(vectors)
                vectors = torch.tensor(vectors, dtype=torch.float32)
                vectors = vectors.view(len(vectors), 1, -1)

                vectors = vectors.to(device)
                gold_label_tensor = torch.tensor([gold_label], device=device)

                output = model(vectors)
                predicted_label = torch.argmax(output, dim=1).item()
                correct += int(predicted_label == gold_label)
                total += 1

        validation_accuracy = correct / total if total > 0 else 0
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {:.4f}".format(epoch + 1, validation_accuracy))

        # Early stopping condition
        if validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Training stopped to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy

        epoch += 1

    print("Training finished.")

    # Save the model if needed
    # torch.save(model.state_dict(), 'rnn_model.pth')
