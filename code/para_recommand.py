# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 22:00:18 2025

@author: jzf
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
import optuna

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

class FNNRegressor2(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_units, dropout_rate, activation):
        super(FNNRegressor2, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_units))
        
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
        
        self.output_layer = nn.Linear(hidden_units, 1)
        
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
        return x

def optuna_opt(NNmodel1,NNmodel2):
    def func1(x1,x2,x3,x4):
        NNmodel1.eval()
        NNmodel2.eval()
        x5 = 1-x1-x2-x3-x4
        if x5 > 0 and x5 < 0.5:
            inp = np.array([x1,x2,x3,x4,x5]).reshape(1, -1)
            oup1 = NNmodel1(torch.tensor(inp, dtype=torch.float32))
            oup2 = NNmodel2(torch.tensor(oup1, dtype=torch.float32))
            oup = oup2.squeeze().tolist()
            score = oup
        else:
            score = -100
        return score
    
    def objective_func1(trial):
        params = {
            'x1':trial.suggest_float('x1',0.01,0.5),
            'x2':trial.suggest_float('x2',0.01,0.5),
            'x3':trial.suggest_float('x3',0.01,0.5),
            'x4':trial.suggest_float('x4',0.01,0.5)
            }
        score = func1(**params)
        return score
    
    def optimize_hyperparameters(objective_func, n_trials, study_name='optimization_study', seed=1):
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.HyperbandPruner(),
            sampler = optuna.samplers.TPESampler(gamma=lambda n: int(0.5 * n),
                                                 seed=seed)
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
    
    seedlist = range(0,100)
    result = []
    for i in seedlist:
        study1 = optimize_hyperparameters(
            objective_func=objective_func1,
            n_trials=200,
            study_name='optimization_study',
            seed=i
        )
        visualize_results(study1)
        result1 = export_all_results(study1, filename_prefix="optuna_results")
        result1 = result1.iloc[0].values.tolist()
        result1.insert(4,1-result1[0]-result1[1]-result1[2]-result1[3])
        NNmodel1.eval()
        result2 = NNmodel1(torch.tensor([result1[0:5]], dtype=torch.float32)).squeeze().tolist()
        result1[5:5] = result2
        result.append(result1)
        
    df_result = pd.DataFrame(result,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','score'])
    df_result_sorted = df_result.sort_values(by=['score'],ascending=False)
    return df_result_sorted.iloc[0:5]

if __name__ == "__main__":
    NNmodel1 = torch.load('NN_net_1.pkl',weights_only=False)
    NNmodel2 = torch.load('NN_net_2.pkl',weights_only=False)
    optuna_result = optuna_opt(NNmodel1,NNmodel2)
    optuna_result.to_csv('para_recomand_8.csv', index=False)