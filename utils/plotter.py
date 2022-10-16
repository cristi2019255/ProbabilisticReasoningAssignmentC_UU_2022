# Copyright 2022 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_data(x: List[float], y: List[float], title="Temperature vs d18_O", x_label= "d18_O_c - d18_O_w", y_label = "Temperature T", file_name = "data.png", folder = "Q1"):
    os.makedirs(os.path.join("plots", folder), exist_ok=True)    
    plt.figure(figsize=(15, 10))
    plt.cla()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    file_name = os.path.join("plots", folder, file_name)
    plt.savefig(file_name)
    
def plot_predictions(x: List[float], y: List[float], pred: pd.DataFrame, another_pred: pd.DataFrame = None, title="Temperature vs d18_O", x_label= "d18_O_c - d18_O_w", y_label = "Temperature T", file_name = "data.png", folder = "Q1"):
    os.makedirs(os.path.join("plots", folder), exist_ok=True)    
    plt.figure(figsize=(15, 10))
    plt.cla()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    means = pred["mean"].to_list()
    stds = pred["std"].to_list()
    plt.errorbar(x, means, yerr=stds, color="red", label="Predicted temperature", fmt=".", capsize=5, capthick=2, ecolor="red", elinewidth=2)
    
    if another_pred is not None:
        plt.errorbar(x, another_pred["mean"].to_list(), yerr=another_pred["std"].to_list(), color="green", label="Predicted temperature without considering uncertainty", fmt=".", capsize=5, capthick=2, ecolor="green", elinewidth=2, alpha=0.5)
    
    plt.legend()
    file_name = os.path.join("plots", folder, file_name)
    plt.savefig(file_name)
    
    
def plot_data_and_fit(x: List[float], y: List[float], fit:pd.DataFrame, model_function, model_name = "Linear regression", title="Temperature vs d18_O", x_label= "d18_O_c - d18_O_w", y_label = "Temperature T", file_name = "data_fit.png", cols = ["a", "b", "sigma"], folder = "Q1"):
    os.makedirs(os.path.join("plots", folder), exist_ok=True)    
    plt.figure(figsize=(15, 10))
    plt.cla()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    
    # plotting the mean of the posterior distribution
    means = fit[cols].mean()
    stds = fit[cols].std()
    label = f"{model_name}"
    for col in cols:
        label += f", {col}_mean = {means[col]:.2f} (+/- {stds[col]:.2f})"
                
    plt.plot(x, model_function(x, means), color="red", label=label)
    
    # plotting samples from the posterior distribution to show the uncertainty cloud
    for index, params in fit[cols].iterrows():
        plt.plot(x, model_function(x, params), color="green", alpha=0.05)
    
    plt.legend()
    file_name = os.path.join("plots", folder, file_name)
    plt.savefig(file_name)
    
def plot_data_and_fit_no_pooling_and_mix_pooling(x: List[float], y: List[float], fit_no_pooling:pd.DataFrame, fit_mix_pooling: pd.DataFrame, model_function, model_name = "Linear regression", title="Temperature vs d18_O", x_label= "d18_O_c - d18_O_w", y_label = "Temperature T", file_name = "data_fit.png", cols = ["a", "b", "sigma"], folder = "Q1"):
    os.makedirs(os.path.join("plots", folder), exist_ok=True)    
    plt.figure(figsize=(15, 10))
    plt.cla()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    
    # plotting no pooling
    
    # plotting the mean of the posterior distribution
    means = fit_no_pooling[cols].mean()
    stds = fit_no_pooling[cols].std()
    label = f"{model_name} no pooling"
    for col in cols:
        label += f", {col}_mean = {means[col]:.2f} (+/- {stds[col]:.2f})"
                
    plt.plot(x, model_function(x, means), color="red", label=label)
    
    # plotting samples from the posterior distribution to show the uncertainty cloud
    for index, params in fit_no_pooling[cols].iterrows():
        plt.plot(x, model_function(x, params), color="green", alpha=0.05)
    
    
    # plotting mix pooling
    means = fit_mix_pooling[cols].mean()
    stds = fit_mix_pooling[cols].std()
    label = f"{model_name} mix pooling"
    for col in cols:
        label += f", {col}_mean = {means[col]:.2f} (+/- {stds[col]:.2f})"
                
    plt.plot(x, model_function(x, means), color="blue", label=label)
    
    # plotting samples from the posterior distribution to show the uncertainty cloud
    for index, params in fit_mix_pooling[cols].iterrows():
        plt.plot(x, model_function(x, params), color="orange", alpha=0.05)
   
    
    plt.legend()
    file_name = os.path.join("plots", folder, file_name)
    plt.savefig(file_name)