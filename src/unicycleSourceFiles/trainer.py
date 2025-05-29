import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt

def model_trainer(model, trainData, testData, num_epochs, batch_size, gamma, learning_rate, weight_decay, file_save_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss = torch.nn.MSELoss(reduction="mean")
    best_test_loss = np.inf
    train_lossArr = []
    test_lossArr = []
    time_Arr = []
    print("Epoch", "Time", "Train Loss", "Test Loss")
    print("Total epochs", num_epochs)
    for ep in range(num_epochs):
        model.train()
        t1 = time.time()
        train_loss = 0
        for x_vals, y_vals in trainData:
            optimizer.zero_grad()
            out = model(x_vals)
        
            lp = loss(out, y_vals) 
            lp.backward()
            
            optimizer.step()
            train_loss += lp.item()
            
        scheduler.step()
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_vals, y_vals in testData:
                out = model(x_vals)
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
    torch.save(bestModelDict,file_save_path)
    return model, train_lossArr, test_lossArr
    
def evaluate_train_performance(model, name, train_lossArr, test_lossArr):
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
    
def evaluate_model(model, model_type, trainData, testData):
    loss = torch.nn.MSELoss()
    train_loss = 0
    test_loss= 0
    with torch.no_grad():
        for x_vals, y_vals in trainData:
            out = model(x_vals)
            train_loss += loss(out, y_vals).item()
            
    with torch.no_grad():
        for x_vals, y_vals in testData:
            out = model(x_vals)
            test_loss += loss(out, y_vals).item()
                
        train_loss /= len(trainData)
        test_loss /= len(testData)
    return train_loss, test_loss
