import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
input_size = 784
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

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

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__() 
        self.features = nn.Sequential()
        self.features.add_module("conv1", nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1))
        #self.features.add_module("relu1", nn.ReLU())
        #self.features.add_module("Sig1", nn.Sigmoid())
        self.features.add_module("tanh1", nn.Tanh())
        self.features.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module("conv2", nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1))
        #self.features.add_module("relu2", nn.ReLU())
        #self.features.add_module("Sig2", nn.Sigmoid())
        self.features.add_module("tanh2", nn.Tanh())
        self.features.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.lin1 = nn.Linear(7 * 7 * 16, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1) 
        out = self.lin1(out)
        return out

model = SimpleNet().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #images = Variable(images.view(-1, 28 * 28))
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # (1)
        loss.backward()
        # (2)
        optimizer.step()
        # (3)

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))

#train_loader_2 = torch.utils.data.DataLoader(dataset = train_dataset,
#        batch_size = batch_size,
#        shuffle = False)
    correct_train = 0
    total_train = 0
    loss_train = 0
    for images, labels in train_loader:
        #images = Variable(images.view(-1, 28 * 28))
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        loss_train += criterion(outputs, labels).data.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum()

    #print('Train Accuracy in epoch % d: % d %%' % (epoch, 100 * correct_train / total_train))
    #print('Train Loss: %.4f' % (loss_train / total_train * batch_size))
    train_accuracy.append(100 * correct_train / total_train)
    train_loss.append(loss_train / total_train * batch_size)

# Test the Model
    correct_test = 0
    total_test = 0
    loss_test = 0
    for images, labels in test_loader:
        #images = Variable(images.view(-1, 28 * 28))
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        loss_test += criterion(outputs, labels).data.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum()

    #print('Test Accuracy: % d %%' % (100 * correct_test / total_test))
    #print('Test Loss: %.4f' % (loss_test / total_test * batch_size))
    test_accuracy.append(100 * correct_test / total_test)
    test_loss.append(loss_test / total_test * batch_size)

plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('./part3_4_tanh_acc.png')

plt.clf()
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./part3_4_tanh_loss.png')


