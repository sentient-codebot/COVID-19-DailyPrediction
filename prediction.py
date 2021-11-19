'''
load model and make predictions
'''
import torch
from torch.utils.data import DataLoader
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Data import COVID19Daily
from NNModel import RIMModel, MLPModel
from ARMA import ARModel

parser = argparse.ArgumentParser()

parser.add_argument('--p', type=int, default=10)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--model', type=str, default='AR')
parser.add_argument('--hidden_size', type=int, default=10)
parser.add_argument('--num_units', type=int, default=4)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=0)

args = vars(parser.parse_args())

load_dir = f'logs/current_{args["model"]}_model.pt'

def predict(model, data, days_ahead=10):
    model.eval()
    all_present = torch.zeros(1,1)
    all_pred = torch.zeros(1,1)
    for past, present in data:
        pred = model(past)
        all_present = torch.cat((all_present, present), 0)
        all_pred = torch.cat((all_pred, pred), 0)
    
    # predict future
    past = torch.cat((past[:, :-1], present.reshape(-1,1)), 1)
    for day in range(days_ahead):
        pred = model(past)
        past = torch.cat((past[:, :-1], pred.reshape(-1,1)), 1)
        all_pred = torch.cat((all_pred, pred), 0)

    return all_present[1:], all_pred[1:]



def main():
    fullfile = "data/COVID-19_aantallen_nationale_per_dag.csv"
    trainset = COVID19Daily(fullfile, args["p"])
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False)

    if args['model'] == 'AR':
        model = ARModel(p = args["p"])
    elif args['model'] == 'MLP':
        model = MLPModel(torch.device('cpu'), 
                        args['p'], args['hidden_size'])
    else:
        model = RIMModel(torch.device('cpu'), 
                        args['hidden_size'], args['num_units'], args['k'], 'LSTM')

    saved = torch.load(load_dir, map_location=torch.device('cpu'))
    model.load_state_dict(saved['net'])

    days_ahead = 7
    actual, predicted = predict(model, trainloader, days_ahead=days_ahead)
    actual = actual.numpy()
    predicted = predicted.detach().numpy()

    fig, axes = plt.subplots(2,1)

    fig.set_figwidth(16)
    fig.set_figheight(8)

    actual_all = axes[0].plot(actual, 'r', label="actual")
    predicted_all = axes[0].plot(predicted, '.', color='dodgerblue', label="predicted") # why i have to detach while model.eval() is on?

    actual_recent = axes[1].plot(actual[-30:], 'r', label="actual")
    predicted_recent = axes[1].plot(predicted[-30-days_ahead:], ':', color='dodgerblue', label="predicted")

    for ax in axes:
        ax.legend(loc='upper left')
    

    plt.show()



    # plt.plot(actual.numpy(), 'r')
    # plt.plot(predicted.detach().numpy(), 'b.') # why i have to detach while model.eval() is on?

    print(f"last seven days: {actual[-7:]}")
    print(f"predictions last seven days: {predicted[-7-days_ahead:-days_ahead]}")

    print(f"predictions coming days: {predicted[-days_ahead:]}")
    
    pass
    

if __name__ == "__main__":
    main()