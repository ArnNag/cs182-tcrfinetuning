import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

pairs = pd.read_csv('TCREpitopePairs.csv')
binding_labels = torch.tensor(pairs["binding"], dtype=torch.float)
epitope_embeddings = np.load("embeddings_epitope_no_finetuning_duplicated.npy")

class BindingClassifier(nn.Module):
    def __init__(self, input_size_A=1024, input_size_B=1024, hidden_size_one=2048, hidden_size_two=1024):
        super().__init__()

        def make_fc(input_size, hidden_size):
            fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3),
                nn.SiLU()
            )
            return fc 

        self.fc_A = make_fc(input_size_A, hidden_size_one)
        self.fc_B = make_fc(input_size_B, hidden_size_one)
        self.fc_combined = make_fc(hidden_size_one * 2, hidden_size_two)

        self.fc_output = nn.Linear(hidden_size_two, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_A, input_B):
        x_A = self.fc_A(input_A)
        x_B = self.fc_B(input_B)

        # Concatenate the outputs of fc_A and fc_B
        x_combined = torch.cat((x_A, x_B), axis=1)

        # Pass through the combined layer
        x_combined = self.fc_combined(x_combined)

        # Final output layer
        output = self.fc_output(x_combined)
        output = self.sigmoid(output)
        output = output.squeeze(-1)

        return output

train_val_idxs, test_idxs = train_test_split(torch.arange(len(tcr_embeddings)), test_size=0.1, random_state=1984)
train_idxs, val_idxs = train_test_split(torch.arange(len(tcr_embeddings)), test_size=0.111111, random_state=1984)


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

class PairDataset(Dataset):

    def __init__(self, epitope_embeddings, tcr_embeddings, binding_labels):
        assert len(epitope_embeddings) == len(tcr_embeddings) == len(binding_labels)
        self.epitope_embeddings = torch.tensor(epitope_embeddings)
        self.tcr_embeddings = torch.tensor(tcr_embeddings)
        self.binding_labels = binding_labels

    def __len__(self):
        return len(epitope_embeddings)

    def __getitem__(self, idx):
        return (epitope_embeddings[idx], tcr_embeddings[idx]), binding_labels[idx]

def run_experiment(tcr_embedding_path):
    model = BindingClassifier(input_size_A, input_size_B)
    tcr_embeddings = np.load(tcr_embedding_path)

    input_size_A = tcr_embeddings.shape[1] 
    input_size_B = epitope_embeddings.shape[1]
    assert(len(tcr_embeddings) == len(epitope_embeddings))

            
    training_data = PairDataset(epitope_embeddings[train_idxs], tcr_embeddings[train_idxs], binding_labels[train_idxs])
    training_loader = DataLoader(training_data, batch_size=int(2.5e3), shuffle=True)

    validation_data = PairDataset(epitope_embeddings[val_idxs], tcr_embeddings[val_idxs], binding_labels[val_idxs])
    validation_loader = DataLoader(training_data, batch_size=int(2.5e3), shuffle=True)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(tqdm(training_loader)):
            # Every data instance is an input + label pair
            inputs, labels = data
            labels = labels.type(torch.float)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(*inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{tcr_embedding_path}/{timestamp}')
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vlabels = vlabels.type(torch.float)
                voutputs = model(*vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}_{}'.format(tcr_embedding_path, timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

if __name__ == "__main__":
    # model_names = ["Rostlab_prot_t5_xl_uniref50", "checkpoint-100000", "checkpoint-200000"]
    model_names = ["Rostlab_prot_t5_xl_uniref50"]
    for name in model_names:
        tcr_embedding_path = f"tcr_embeddings_{name}.npy"
        run_experiment(tcr_embedding_path)
