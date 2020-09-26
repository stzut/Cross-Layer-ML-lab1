import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse

# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch-size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--enable-cuda', action="store_true", help='Flag that enables cuda')
args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='./data',
        train = True,
        transform = transforms.ToTensor(),
        download = True)

test_dataset = dsets.MNIST(root ='./data',
        train = False,
        transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False)

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

if args.enable_cuda:
    model = LogisticRegression(input_size, num_classes).cuda()
else:
    model = LogisticRegression(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
reg = 0.005
# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if args.enable_cuda:    
            images = Variable(images.view(-1, 28 * 28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # (1)
        # implement the L1 regularization here
        loss += reg * model.linear.weight.norm(p=1)

        loss.backward()
        # (2)
        optimizer.step()
        # (3)

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))

# output the L1 norm of weights
print(model.linear.weight.norm(p=1))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    if args.enable_cuda:
        images = Variable(images.view(-1, 28 * 28)).cuda()
        labels = Variable(labels).cuda()
    else:
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
    
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (100 * correct / total))

