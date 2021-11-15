from ARMA import ARModel
import torch
import torch.nn as nn
from tqdm import tqdm
from Data import COVID19Daily
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--p', type = float, default = 10)
parser.add_argument('--epochs', type = int, default = 100)

args = vars(parser.parse_args())
torch.cuda.manual_seed(10)

loss_function = nn.MSELoss()

def train_model(model, epochs, train_data, val_data=None):
    acc=0.0
    best_acc= 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for epoch in range(epochs):
        epoch_loss = 0.
        model.train()
        for past, present in train_data:
            pred = model(past)
            loss = loss_function(present, pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            # error = torch.abs(present - pred)
            # acc = torch.mean(error/(present+1e-8))
            # print(f'current accuracy: {acc}%')

        print(f"epoch loss: {epoch_loss}")
        test_model(model, val_data)

def test_model(model, val_data):
    mean_error = 0
    model.eval()
    
    for past, present in val_data:
        pred  = model(past)
        error = torch.abs(present - pred)
        error = torch.mean(error/(present+1e-8))
        mean_error += error

    mean_error = mean_error/len(val_data)
    print(f"epoch error: {mean_error*100}%")
    return error

def main():
    fullfile = "data/COVID-19_aantallen_nationale_per_dag.csv"
    trainset = COVID19Daily(fullfile, args["p"])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    model = ARModel(p = args["p"])

    train_model(model, args['epochs'], trainloader, trainloader)

    error = test_model(model, trainloader)
    print(f"average error after training is: {error*100}%")

if __name__ == "__main__":
    main()