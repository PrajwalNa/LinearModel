import torch
from torch import nn as neural
import matplotlib.pyplot as mPlot


# creating the model
class LinearModel(neural.Module):
    def __init__(self):
        super().__init__()
        self.weights = neural.Parameter(
            torch.rand(1, dtype=torch.float32), requires_grad=True
        )
        self.bias = neural.Parameter(
            torch.rand(1, dtype=torch.float32), requires_grad=True
        )

    def linear(self, x):
        return self.weights * x + self.bias  # y = wx + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def getWeights(self):
        return self.weights

    def getBias(self):
        return self.bias


def graph(
    test,
    testLabel,
    train,
    trainLabel,
    predicted=None,
    title="",
    training=False,
):
    mPlot.clf()  # clear the plot

    # move all tensors to cpu for plotting
    test = test.to("cpu")
    testLabel = testLabel.to("cpu")

    if train != None:
        train = train.to("cpu")
        trainLabel = trainLabel.to("cpu")
        mPlot.scatter(
            train, trainLabel, label="True Data", c="b", s=40, marker="1"
        )  # plot training data

    mPlot.scatter(
        test, testLabel, label="Test Data", c="r", s=40, marker="+"
    )  # plot testing data

    if predicted is not None:
        predicted = predicted.to("cpu")
        mPlot.scatter(
            test, predicted, label="Predicted Data", c="g", s=40, marker="+"
        )  # plot prediction

    mPlot.title(title)
    mPlot.legend(prop={"size": 14})

    if training == True:
        mPlot.draw()  # update the figure
        mPlot.pause(0.5)  # pause a bit so that plots are updated
    else:
        mPlot.show()  # show the plot


def trainLoop(
    model,
    xTrain,
    yTrain,
    xTest,
    yTest,
    lossFunction,
    optim,
    epoch=0,
    epochCount=[],
    trainLoss=[],
    testLoss=[],
):

    loss = 1
    testLossValue = 1
    # train while the loss is greater than 0.002
    while loss > 0.002 and testLossValue > 0.002:
        epoch += 1
        # set model to training mode
        model.train()

        # forward pass
        predicted = model(xTrain)
        loss = lossFunction(predicted, yTrain)

        # backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        # testing the model
        model.eval()

        with torch.inference_mode():
            testPredicted = model(xTest)
            testLossValue = lossFunction(testPredicted, yTest)

        # appending values to the lists
        trainLoss.append(loss.detach().to("cpu").numpy())
        testLoss.append(testLossValue.detach().to("cpu").numpy())
        epochCount.append(epoch)

        # printing the loss
        if epoch % 6 == 0:
            print(
                f"Epoch: {epoch}, Training Loss: {loss.item()}, Testing Loss: {testLossValue.item()}"
            )
            graph(xTest, yTest, xTrain, yTrain,
                  testPredicted, "Training Model", True)

    return epochCount, epoch, trainLoss, testLoss


def TestModel(
    xTest,
    yTest,
    model,
    labl,
    xtrue=None,
    ytrue=None,
):
    model.eval()

    with torch.inference_mode():
        predictions = model(xTest)

    graph(xTest, yTest, xtrue, ytrue, predictions, labl)


def plotLoss(epochCount, trainLoss, testLoss):
    mPlot.figure(figsize=(10, 7))
    mPlot.plot(epochCount, trainLoss, label="Training Loss")
    mPlot.plot(epochCount, testLoss, label="Testing Loss")
    mPlot.xlabel("Epoch")
    mPlot.ylabel("Loss")
    mPlot.legend()
    mPlot.show()
