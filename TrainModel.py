from LineModel import *

"""
Consistenly scary accurate models with this algorithm.
using two pass training with different learning rates and similiar datasets to refine the model
loss function is L1Loss (Mean Absolute Error)
optimizer is Adam (Adaptive Moment Estimation) with learning rates of 0.01 and 0.001
Adam is a derivative of Stochastic Gradient Descend (SGD) and is more efficient, like way more efficient
"""


# make sure all tensors are made on cuda
torch.set_default_device("cuda")

# creating parameters for the model
weight = 0.7
bias = 0.3

# creating data for the model
x = torch.arange(0, 1, 0.002).reshape(-1, 1)
y = weight * x + bias

z = torch.randn_like(x) * 0.2
v = weight * z + bias


split = int(0.8 * len(x))

# separating the data into training and testing in 80% training and 20% testing
xTrain = x[:split]
yTrain = y[:split]

xTest = x[split:]
yTest = y[split:]


# separating the secondary test data same ratio
zTrain = z[:split]
vTrain = v[:split]

zTest = z[split:]
vTest = v[split:]

# initializing the model
modelZero = LinearModel()

# setting the learning rate and manual seed
torch.manual_seed(20)


# setting up the loss function, optimizer and learning rate scheduler
lossFunction = neural.L1Loss()
optim = torch.optim.Adam(modelZero.parameters(), lr=0.01)

# training the model using xy data
epC, ep, train, test = trainLoop(
    modelZero, xTrain, yTrain, xTest, yTest, lossFunction, optim
)

# training the model using zv data
optim = torch.optim.Adam(modelZero.parameters(), lr=0.001)
epC, ep, train, test = trainLoop(
    modelZero,
    zTrain,
    vTrain,
    zTest,
    vTest,
    lossFunction,
    optim,
    ep,
    epC,
    train,
    test,
)

plotLoss(epC, train, test)

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(modelZero.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# plot the graph for testing data
TestModel(xTest, yTest, modelZero, "Test Model - xy", x, y)
TestModel(zTest, vTest, modelZero, "Test Model - zv", z, v)

# makings new data for testing
z = torch.arange(0, 1, 0.3).reshape(-1, 1)
v = weight * z + bias
zTrain = z[split:]
vTrain = v[split:]
zTest = z[:split]
vTest = v[:split]


TestModel(zTest, vTest, modelZero, "Test Model - zv(new)", z, v)

torch.save(modelZero.state_dict(), "models/modelTest.pth")

# modelZero.load_state_dict(torch.load("models/modelZero.pth"))
# TestModel(zTest, vTest, modelZero, "Model Zero", z, v)

# modelZero.load_state_dict(torch.load("models/modelOne.pth"))
# TestModel(zTest, vTest, modelZero, "Model One", z, v)
