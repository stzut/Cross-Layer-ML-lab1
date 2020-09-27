import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        self.features.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module("conv2", nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1))
        self.features.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.lin1 = nn.Linear(7 * 7 * 16, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1) 
        out = self.lin1(out)
        return out

model = SimpleNet().cuda()
train_count = 0
y = []
def getnorm(self, input, output):
    if train_count % 600 == 0:
    	y.append(self.weight.data.norm())

model.features.conv2.register_backward_hook(getnorm)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #images = Variable(images.view(-1, 28 * 28))
        train_count += 1
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


# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    #images = Variable(images.view(-1, 28 * 28))
    images = Variable(images).cuda()
    labels = Variable(labels).cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (100 * correct / total))
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x_axis,y)
plt.ylabel('Norm of Weight')
plt.xlabel('Number of Epochs')
plt.savefig('./part3_2.png')
