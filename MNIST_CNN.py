import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch import optim

# -------------------------------------
# Data Downloading and Preprocessing
# -------------------------------------

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),                # Convert images to PyTorch tensors (scale tha values to a range between 0 and 1)
    transforms.Normalize((0.5,), (0.5,))  # Normalize tensors to have a mean of 0 and a range between -1 and 1
])

# Download and load the MNIST dataset for training
trainset = torchvision.datasets.MNIST(
    root='./data',          # Where the data will be saved
    train=True,             # Get the training set
    download=True,          # Download the data if not present
    transform=transform     # Apply the transformations
)

# Create a data loader for the training set
trainloader = torch.utils.data.DataLoader(
    trainset,               # Dataset
    batch_size=64,          # Load 64 samples per batch
    shuffle=True            # Shuffle the data
)

# Download and load the MNIST dataset for testing
testset = torchvision.datasets.MNIST(
    root='./data',          # Directory where the data will be saved
    train=False,            # Get the test set
    download=True,          # Download the data if not present
    transform=transform     # Apply the transformations
)

# Create a data loader for the test set
testloader = torch.utils.data.DataLoader(
    testset,                # Dataset
    batch_size=64,          # Load 64 samples per batch
    shuffle=False           # Don't schuffle the data
)

# -------------------------------------
# Define the Model
# -------------------------------------

class MNIST_CNN_Model(nn.Module):
    def __init__(self):
        """Constructor method to define the layers of the CNN"""
        super().__init__()
        # Define the first convolutional layer: 
        # Input: 1 channel, Output: 32 channels, Kernel size: 3x3, Padding: 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # Define the second convolutional layer:
        # Input: 32 channels, Output: 64 channels, Kernel size: 3x3, Padding: 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Define the first fully connected layer:
        # Input: 64*7*7 (flattened image from convolution layers), Output: 128 neurons
        self.fc1 = nn.Linear(64*7*7, 128)
        
        # Define the second fully connected layer:
        # Input: 128 neurons, Output: 10 (one for each digit 0-9)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        """Method that returns the predictions"""
        # Apply first convolutional layer and ReLU
        x = torch.relu(self.conv1(x))
        
        # Apply max pooling
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        
        # Apply second convolutional layer and ReLU
        x = torch.relu(self.conv2(x))
        
        # Apply max pooling
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output
        x = x.view(-1, 64*7*7)
        
        # Apply first fully connected layer with ReLU
        x = torch.relu(self.fc1(x))
        
        # Apply second fully connected layer
        out = self.fc2(x)
        
        return out

# -------------------------------------
# Model Initialization
# -------------------------------------

# Initialize the model
model = MNIST_CNN_Model()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the number of epochs
num_epochs = 5

# -------------------------------------
# Training the Model
# -------------------------------------

# Loop over the dataset
for epoch in range(num_epochs):
    curr_loss = 0.0  # Initialize the current loss
    
    # Loop over the data
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data 
        
        # Zero out the gradient
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss (difference between predicted and actual values)
        loss = criterion(outputs, labels)
        
        # Backward pass (calculate the gradient)
        loss.backward()
        
        # Optimize (make a step in the opposite direction)
        optimizer.step()
        
        # Add the loss of the current predictio to the running loss
        curr_loss += loss.item()
        
        # Print statistics every 100 mini-batches
        if (i % 100 == 0 and i != 0):
            # print the epoch, batch and the avarage loss of each prediction in that batch
            print('epoch: %d - %d loss: %.5f' %(epoch, i, curr_loss/100))

            # Reset the running loss
            curr_loss = 0.0 

print('Finished Training')

# -------------------------------------
# Testing the Model
# -------------------------------------

# Initialize counters
correct = 0
total = 0

with torch.no_grad():
    # Loop over the test data
    for data in testloader:
        images, labels = data  # Get the inputs and labels
        
        # Forward pass
        outputs = model(images)
        
        # Get the prediction
        _, predicted = torch.max(outputs.data, 1)
        
        # Increase the total
        total += labels.size(0)
        
        # Increase the correct predictions
        correct += (predicted == labels).sum().item()

# Print the accuracy
print('Accuracy: %.2f%%' % (100 * correct / total))
