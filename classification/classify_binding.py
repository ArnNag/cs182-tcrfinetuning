import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

pairs = pd.read_csv('TCREpitopePairs.csv')
tcr_embeddings = np.load("tcr_embeddings_no_finetune.npy")
epitope_embeddings = np.load("embeddings_epitope_no_finetuning_duplicated.npy")

class BindingClassifier(nn.Module):
    def __init__(self, input_size_A=1024, input_size_B=1024, hidden_size=2048):
        super().__init__()

        def make_fc(input_size, hidden_size):
            fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3),
                nn.SiLU()
            )
            return fc 

        self.fc_A = make_fc(input_size_A)
        self.fc_B = make_fc(input_size_B)
        self.fc_combined = make_fc(hidden_size * 2)

        self.fc_output = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_A, input_B):
        x_A = self.fc_A(input_A)
        x_B = self.fc_B(input_B)

        # Concatenate the outputs of fc_A and fc_B
        x_combined = torch.cat((x_A, x_B))

        # Pass through the combined layer
        x_combined = self.fc_combined(x_combined)

        # Final output layer
        output = self.fc_output(x_combined)
        output = self.sigmoid(output)

        return output

# Assuming X1_train and X2_train are your training data for inputA and inputB
input_size_A = tcr_embeddings.size()[1] 
input_size_B = epitope_embedings.size()[1]

model = BindingClassifier(input_size_A, input_size_B)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

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
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
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
            voutputs = model(vinputs)
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
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
