import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from dataset import bAbIDataset, bAbIDataLoader
from model import GGNN

# Task ID to use out of the bAbI tasks. Task 19 is path finding and according
# to the paper is "arguably the hardest task".
task_id = 19

# Batch size for training
batch_size = 10

# Some of the bAbI tasks (all of the ones we have here) have multiple question
# types.
question_id = 0

# Use the GPU?
use_cuda = False

# If we should log training loss to output.
should_log_train = False

# Learning rate for training
lr = 0.01

# Number of epochs for training
n_epochs = 10

# GGNN hidden state size
state_dim = 4

# Number of propogation steps
n_steps = 5

# Annotation dimension. For the bAbi tasks we have one hot encoding per node.
annotation_dim = 1

# One fold of our preprocessed dataset.
dataset_path = 'babi_data/processed_1/train/%d_graphs.txt' % task_id

train_dataset = bAbIDataset(dataset_path, question_id=0, is_train=True)
train_data_loader = bAbIDataLoader(train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2)

test_dataset = bAbIDataset(dataset_path, question_id=0, is_train=False)
test_data_loader = bAbIDataLoader(test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2)

n_edge_types = train_dataset.n_edge_types
n_nodes = train_dataset.n_node

# The dataset has the form: [(adjacency matrix, annotation, target), ...]
ggnn = GGNN(state_dim, annotation_dim, n_edge_types, n_nodes, n_steps)

# The dataset is all doubles so convert the model to be double
ggnn = ggnn.double()

crit = nn.CrossEntropyLoss()

if use_cuda:
    net.use_cuda()
    crit.use_cuda()

opt = optim.Adam(ggnn.parameters(), lr=lr)

def model_inference(ggnn, adj_matrix, annotation, target):
    padding = torch.zeros(len(annotation), n_nodes, state_dim -
            annotation_dim).double()

    # See section 3.1 of the paper for how we create the node annotations.
    init_input = torch.cat((annotation, padding), 2)

    if use_cuda:
        init_input = init_input.use_cuda()
        adj_matrix = adj_matrix.use_cuda()
        annotation = annotation.use_cuda()
        target = target.use_cuda()

    output = ggnn(init_input, annotation, adj_matrix)

    return output, target


for epoch in range(n_epochs):
    # Train
    ggnn.train()
    for i, (adj_matrix, annotation, target) in enumerate(train_data_loader):
        # Adjency matrix will have shape [batch_size, n_nodes, 2 * n_nodes * n_edge_types]
        ggnn.zero_grad()

        output, target = model_inference(ggnn, adj_matrix, annotation, target)
        loss = crit(output, target)

        loss.backward()
        opt.step()

        if should_log_train:
            print('[%i / %i], [%i / %i] Loss: %.4f' % (epoch, n_epochs, i,
                len(train_data_loader), loss.data))

    # Evaluate performance over validation dataset.
    ggnn.eval()
    test_loss = 0
    correct = 0
    for adj_matrix, annotation, target in test_data_loader:
        output, target = model_inference(ggnn, adj_matrix, annotation, target)

        test_loss += crit(output, target).data
        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_dataset)

    print('[%i, %i] Val: Avg Loss %.4f, Accuracy %i/%i' % (epoch, n_epochs, test_loss,
        correct, len(test_dataset)))

