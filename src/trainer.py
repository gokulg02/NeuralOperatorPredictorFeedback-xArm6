import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import deepxde as dde

def model_trainer(model, trainData, testData, num_epochs, batch_size, gamma, learning_rate, weight_decay, model_type, model_filename):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss = torch.nn.MSELoss()
    best_test_loss = np.inf
    train_lossArr = []
    test_lossArr = []
    time_Arr = []
    match model_type:
        case "DeepONet":
            grid = model.grid
        case "GRU" | "LSTM" | "FNO+GRU":
            nD = model.nD
            dof = model.dof
        case "FNO":
            pass
        case _:
            raise Exception("Model type not supported. Please use GRU, FNO, DeepONet, LSTM, DeepONet+GRU, FNO+GRU.")
    print("Epoch", "Time", "Train Loss", "Test Loss")
    print("Total epochs", num_epochs)
    for ep in range(num_epochs):
        model.train()
        t1 = time.time()
        train_loss = 0
        for x_vals, y_vals in trainData:
            optimizer.zero_grad()
            match model_type:
                case "DeepONet":
                    out = model((x_vals, grid))
                case "GRU" | "LSTM":
                    x_vals = x_vals.reshape(x_vals.shape[0], nD, 3*dof)
                    y_vals = y_vals.reshape(y_vals.shape[0], nD, 2*dof)
                    out = model(x_vals)
                case "FNO":
                    out = model(x_vals)
                case "FNO+GRU":
                    y_vals = y_vals.reshape(y_vals.shape[0], nD, 2*dof)
                    out = model(x_vals)
                case _:
                    raise Exception("Model type not supported. Please use GRU, FNO, DeepONet, LSTM, DeepONet+GRU, FNO+GRU.")
    
            lp = loss(out, y_vals)
            lp.backward()
            
            optimizer.step()
            train_loss += lp.item()
            
        scheduler.step()
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_vals, y_vals in testData:
                match model_type:
                    case "DeepONet":
                        out = model((x_vals, grid))
                    case "GRU" | "LSTM":
                        x_vals = x_vals.reshape(x_vals.shape[0], nD, 3*dof)
                        y_vals = y_vals.reshape(y_vals.shape[0], nD, 2*dof)
                        out = model(x_vals)
                    case "FNO":
                        out = model(x_vals)
                    case "FNO+GRU":
                        y_vals = y_vals.reshape(y_vals.shape[0], nD, 2*dof)
                        out = model(x_vals)
                    case _:
                        raise Exception("Model type not supported. Please use GRU, FNO, DeepONet, LSTM, DeepONet+GRU, FNO+GRU.")
                test_loss += loss(out, y_vals).item()
                
        train_loss /= len(trainData)
        test_loss /= len(testData)
        
        train_lossArr.append(train_loss)
        test_lossArr.append(test_loss)
        
        t2 = time.time()
        time_Arr.append(t2-t1)
        if ep%5 == 0:
            print(ep, t2-t1, train_loss, test_loss)

        if test_loss <= best_test_loss:
            bestModelDict = model.state_dict()
            best_test_loss = test_loss
    torch.save(bestModelDict,"../models/" + model_filename)
    return model, train_lossArr, test_lossArr
    
def evaluate_model(model, name, train_lossArr, test_lossArr):
    print("Evaluating model", name)
    # Display Model Details
    plt.figure()
    plt.title("Loss function "+name)
    plt.plot(train_lossArr, label="Train Loss")
    plt.plot(test_lossArr, label="Test Loss")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.savefig("losscurve.pdf")
        
    print("Final Testing Loss:", test_lossArr[-1])
    print("Final Training Loss:", train_lossArr[-1])
