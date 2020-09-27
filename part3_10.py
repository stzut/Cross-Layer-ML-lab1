import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
from torchsummary import summary
input_size = 784
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.01

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='./data',
        train = True,
        transform = transforms.ToTensor(),
        download = True)
train_dataset_1 = []
for i in range(int(len(train_dataset) / 2)):
    train_dataset_1.append(train_dataset[i])


test_dataset = dsets.MNIST(root ='./data',
        train = False,
        transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset_1,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__() 

        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4608,1152)
        self.fc2 = nn.Linear(1152,144)
        self.fc3 = nn.Linear(144, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(out)))
        out = self.relu(self.bn6(self.conv6(out)))
        out = self.pool(out)

        out = self.relu(self.bn7(self.conv7(out)))
        out = self.relu(self.bn8(self.conv8(out)))
        out = self.relu(self.bn9(self.conv9(out)))
        out = self.relu(self.bn9(self.conv10(out)))
        out = self.relu(self.bn9(self.conv11(out)))
        out = self.relu(self.bn9(self.conv12(out)))
        out = self.pool(out)

        out = self.relu(self.bn9(self.conv13(out)))
        out = self.relu(self.bn9(self.conv14(out)))
        out = self.relu(self.bn9(self.conv15(out)))
        out = self.pool(out)
        out = out.view(out.size(0), -1) 
        
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

model = SimpleNet().cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
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
                       len(train_dataset_1) // batch_size, loss.data.item()))

    correct_train = 0
    total_train = 0
    loss_train = 0
    for images, labels in train_loader:
        #images = Variable(images.view(-1, 28 * 28))
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        #loss_train += criterion(outputs, labels).data.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum()

    #print('Train Accuracy in epoch % d: % d %%' % (epoch, 100 * correct_train / total_train))
    #print('Train Loss: %.4f' % (loss_train / total_train * batch_size))
    train_accuracy.append(100 * correct_train / total_train)
    #train_loss.append(loss_train / total_train * batch_size)



# Test the Model
    correct_test = 0
    total_test = 0
    loss_test = 0
    for images, labels in test_loader:
        #images = Variable(images.view(-1, 28 * 28))
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        #loss_test += criterion(outputs, labels).data.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum()

    #print('Test Accuracy: % d %%' % (100 * correct_test / total_test))
    #print('Test Loss: %.4f' % (loss_test / total_test * batch_size))
    test_accuracy.append(100 * correct_test / total_test)
    #test_loss.append(loss_test / total_test * batch_size)

print('The training accuracy is: % d %%' % train_accuracy[9])
print('The test accuracy is: % d %%' % test_accuracy[9])

plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('./part3_10_Adam.png')

