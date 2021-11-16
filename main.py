from ARMA import ARModel
from NNModel import RIMModel, MLPModel
import torch
import torch.nn as nn
from tqdm import tqdm
from Data import COVID19Daily
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--p', type=int, default=10)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--model', type=str, default='AR')
parser.add_argument('--hidden_size', type=int, default=10)
parser.add_argument('--num_units', type=int, default=4)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--weight_decay', type=float, default=0)


args = vars(parser.parse_args())
torch.cuda.manual_seed(10)

loss_function = nn.MSELoss()

writer = SummaryWriter()

def train_model(model, epochs, train_data, val_data=None):
    acc=0.0
    best_acc= 0.0

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args['lr'], weight_decay=args['weight_decay'])

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

        # print(f"epoch loss: {epoch_loss}")
        mean_error = test_model(model, val_data)
        print(f"{epoch}: epoch error: {mean_error*100: .2f}%")
        writer.add_scalar('Error/train', mean_error*100, epoch)

def test_model(model, val_data):
    mean_error = 0
    model.eval()
    
    for past, present in val_data:
        pred  = model(past)
        error = torch.abs(present - pred)
        error = torch.mean(error/(present+1e-8))
        mean_error += error

    mean_error = mean_error/len(val_data)
    return mean_error

def main():
    fullfile = "data/COVID-19_aantallen_nationale_per_dag.csv"
    trainset = COVID19Daily(fullfile, args["p"])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    if args['model'] == 'AR':
        model = ARModel(p = args["p"])
    elif args['model'] == 'MLP':
        model = MLPModel(torch.device('cpu'), 
                        args['p'], args['hidden_size'])
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"aantal parameters: {pytorch_total_params}")
    else:
        model = RIMModel(torch.device('cpu'), 
                        args['hidden_size'], args['num_units'], args['k'], 'LSTM')
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"aantal parameters: {pytorch_total_params}")
    
    train_model(model, args['epochs'], trainloader, trainloader)

    error = test_model(model, trainloader)
    print(f"evaluation error: {error*100}%")

    model.eval()
    df = pd.read_csv(fullfile)
    past = torch.tensor(df.tail(args['p']+1).iloc[:,-1].to_numpy()).reshape(1,-1)
    pred_today = model(past[:, :-1].float())
    pred_tmr = model(past[:, 1:].float())
    print(f"pred today: {pred_today}, actual today: {past[:, -1]}")
    print(f"pred tomorrow: {pred_tmr}")

if __name__ == "__main__":
    main()