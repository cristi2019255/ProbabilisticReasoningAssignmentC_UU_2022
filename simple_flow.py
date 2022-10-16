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

from model_function import construct_model_function
from utils.plotter import plot_data
from utils.read import read_data
import numpy as np
import stan
from utils.stan_models import get_stan_code
from utils.plotter import plot_data_and_fit
from utils.write import write_results

def simple_flow(question = "Q1"):
    stan_code = get_stan_code(question=question)
    
    data = read_data("data/merged_data.csv", cols = ["d18_O_w", "d18_O", "temperature"])
    
    data = {
        "N": len(data),
        "d18_O_w": data["d18_O_w"].to_list(),
        "d18_O_c": data["d18_O"].to_list(),
        "y": data["temperature"].to_list()
    }
    
    
    # visualizing the data 
    x = np.array(data["d18_O_c"]) - np.array(data["d18_O_w"])
    y = data["y"]
    
    # plotting the data
    plot_data(x, y, folder = question)
    
    # fitting the model
    posterior = stan.build(stan_code, data=data, random_seed=1)
    
    fit = posterior.sample(num_chains=4, num_samples=1000)
    df = fit.to_frame()
    print(df)
    print(df.describe().T)    
       
    # getting the parameters from the posterior   
    write_results(df, file_name = "results.txt", cols = ["a", "b", "sigma"], folder=question)
    
    plot_data_and_fit(x, y, df, construct_model_function(), folder = question)
