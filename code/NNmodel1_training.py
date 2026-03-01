# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 21:06:46 2025

@author: jzf
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import optuna

def draw_scatter(x,y,filename):
    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Prediction')
    ax1.scatter(x, y, s=10, c='k', marker='.')
    a = np.arange(0, 1, 0.00001)
    b = a
    plt.plot(a,b)
    plt.text(0.8,0.1,'r2=%.3f'%r2, fontsize=12, color='k')
    plt.text(0.8,0,'RMSE=%.3f'%rmse, fontsize=12, color='k')
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.show()

class FNNRegressor1(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_units, dropout_rate, activation):
        super(FNNRegressor1, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_units))
        
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
        
        self.output_layer = nn.Linear(hidden_units, 5)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.layer0 = nn.Linear(hidden_units, hidden_units)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.LeakyReLU()
            
        self.dropout = nn.Dropout(dropout_rate)
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_units) for _ in range(hidden_layers)])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def optuna_opt(data):
    X = data[['x1','x2','x3','x4','x5']]
    Y = data[['x6','x7','x8','x9','x10']]
    x = X.to_numpy()
    y = Y.to_numpy()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    def create_optimizer(model, optimizer_name, lr):
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            return optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            return optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model.parameters(), lr=lr)
        else:
            return optim.Adam(model.parameters(), lr=lr)
    
    def func(hidden_layers,hidden_units,dropout_rate,activation,lr,optimizer_name,epoches):
        model = FNNRegressor1(x_train_tensor.shape[1],hidden_layers,hidden_units,dropout_rate,activation)
        optimizer = create_optimizer(model, optimizer_name, lr)
        criterion = nn.SmoothL1Loss()
        epoches = epoches
        for i in range(epoches):
            img = Variable(x_train_tensor)
            label = Variable(y_train_tensor)
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        y_test_pred = model(x_test_tensor).squeeze().detach().numpy()
        y_train_pred = model(x_train_tensor).squeeze().detach().numpy()
        y_test_real = y_test.squeeze()
        y_train_real = y_train.squeeze()
        r2_test = r2_score(y_test_real, y_test_pred)
        r2_train = r2_score(y_train_real, y_train_pred)
        r2 = r2_test
        return r2
    
    def objective_func1(trial):
        params = {
            'hidden_layers':trial.suggest_int('hidden_layers',1,5),
            'hidden_units':trial.suggest_int('hidden_units',32,512),
            'dropout_rate':trial.suggest_float('dropout_rate',0.1,0.7),
            'activation':trial.suggest_categorical('activation',['gelu','relu','tanh','leaky_relu','elu']),
            'lr':trial.suggest_float('lr',1e-5, 1e-2),
            'optimizer_name':trial.suggest_categorical('optimizer_name',['Adam', 'SGD', 'RMSprop', 'AdamW']),
            'epoches':trial.suggest_int('epoches',100, 500)
            }
        score = func(**params)
        return score
    
    def optimize_hyperparameters(objective_func, n_trials, study_name='optimization_study', seed=1):
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.HyperbandPruner(),
            sampler = optuna.samplers.TPESampler(n_startup_trials=500,seed=seed)
            )
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)
        return study
    
    def visualize_results(study):
        print(f"best trial number: {study.best_trial.number}")
        print(f"best value: {study.best_value:.4f}")
        print(f"best params: {study.best_params}")
    
    def export_all_results(study, filename_prefix="optuna_results"):
        trials = study.trials
        results = []
        for trial in trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            result = {}
            for param_name, param_value in trial.params.items():
                result[f'{param_name}'] = param_value
            result['score'] = trial.value
            results.append(result)
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
        return df
    
    def model_train(para):
        best_para = para.iloc[0].values.tolist()
        model = FNNRegressor1(x_train_tensor.shape[1],best_para[0],best_para[1],best_para[2],best_para[3])
        optimizer = create_optimizer(model, best_para[5], best_para[4])
        criterion = nn.SmoothL1Loss()
        epoches = best_para[6]
        for i in range(epoches):
            img = Variable(x_train_tensor)
            label = Variable(y_train_tensor)
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model, 'NN_net_1.pkl')
        model.eval()
        y_test_pred = model(x_test_tensor).squeeze().detach().numpy()
        y_train_pred = model(x_train_tensor).squeeze().detach().numpy()
        y_test_real = y_test.squeeze()
        y_train_real = y_train.squeeze()
        draw_scatter(y_train_real,y_train_pred,'Training set of neural network model 1')
        draw_scatter(y_test_real,y_test_pred,'Testing set of neural network model 1')
    
    study1 = optimize_hyperparameters(
        objective_func=objective_func1,
        n_trials=300,
        study_name='optimization_study',
        seed=42
    )
    visualize_results(study1)
    result = export_all_results(study1, filename_prefix="optuna_results")
    model_train(result)
    
    return result.loc[0]
    
if __name__ == "__main__":
    input_name = 'data_8.csv'
    data = pd.read_csv(input_name)
    para = optuna_opt(data)
    para.to_csv('NN_best_para_1.csv', index=False)