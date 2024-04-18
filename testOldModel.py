from LineModel import *

# make sure all tensors are made on cuda
torch.set_default_device("cuda")

modelTest0 = LinearModel()
modelTest1 = LinearModel()
modelTest2 = LinearModel()

modelTest0.load_state_dict(torch.load("models/modelZero.pth"))
modelTest1.load_state_dict(torch.load("models/modelOne.pth"))
modelTest2.load_state_dict(torch.load("models/modelTwo.pth"))

print(modelTest0.state_dict())
print()
print(modelTest1.state_dict())
print()
print(modelTest2.state_dict())

weight = 0.7
bias = 0.3

x = torch.arange(0, 1, 0.01).reshape(-1, 1)
y = weight * x + bias

split = int(0.8 * len(x))

xTest = x[:split]
yTest = y[:split]


# mPlot.scatter(
#     x.detach().to("cpu"), y.detach().to("cpu"), label="Data", c="b", s=40, marker="1"
# )
# mPlot.title("Dataset")
# mPlot.show()

TestModel(xTest, yTest, modelTest0, "Loaded Model - Model Zero", x, y)

TestModel(xTest, yTest, modelTest1, "Loaded Model - Model One", x, y)

TestModel(xTest, yTest, modelTest2, "Loaded Model - Model Two", x, y)


x = torch.arange(0, 1, 0.2).reshape(-1, 1)
y = weight * x + bias

xTest = x[:split]
yTest = y[:split]

TestModel(xTest, yTest, modelTest0, "Loaded Model - Model Zero", x, y)

TestModel(xTest, yTest, modelTest1, "Loaded Model - Model One", x, y)

TestModel(xTest, yTest, modelTest2, "Loaded Model - Model Two", x, y)
