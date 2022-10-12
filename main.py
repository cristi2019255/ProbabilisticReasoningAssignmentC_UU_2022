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


from utils.read import read_data


def main():
    stan_code = """
        data {
            int J;
            int n[J];
            vector[J] x;
            int y[J];
        }
        
        parameters {
            real a;
            real b;
        }
        
        model {
            y ~ binomial_logit(n, a + b*x);
        }
    """
    
    data = read_data("data/merged_data.csv")
    
    """
    # visualize the data
    x = golf_data["x"]
    y = []
    ticks = []
    plt.title("Golf data")
    plt.xlabel("Distance from the hole")
    plt.ylabel("Probability of success")
    
    for i in range(golf_data["J"]):
        p_hat = golf_data["y"][i]/golf_data["n"][i]
        error = sqrt(p_hat * (1 - p_hat) / golf_data["n"][i])
        ticks.append(f'{golf_data["y"][i]}/{golf_data["n"][i]}')
        y.append(p_hat)
        
    #plt.errorbar(x=x, y=y, yerr=error, fmt="o", capsize=5, capthick=2, ecolor="red", elinewidth=2, color="blue")
    #plt.show()
    
    fit_logistic = stan.build(stan_code, data=golf_data)    
    fit = fit_logistic.sample(num_chains=1, num_samples=200)
    df = fit.to_frame()
    
    print(df)
    
    print(df.describe().T)
    
    x_line = np.linspace(2, 20, 1000)
    a_mean = df["a"].mean()
    b_mean = df["b"].mean()
    a_std = df["a"].std()
    b_std = df["b"].std()
    a_values = np.random.normal(a_mean, a_std, 100)
    b_values = np.random.normal(b_mean, b_std, 100)
    
    
    plt.title("Golf data")
    plt.xlabel("Distance from the hole")
    plt.ylabel("Probability of success")
    plt.errorbar(x=x, y=y, yerr=error, fmt="o", capsize=5, capthick=2, ecolor="red", elinewidth=2, color="blue")
    
    plt.plot(x_line, 1/(1+np.exp(-(a_mean + b_mean * x_line))), color="black", label="Logistic regression")
    for a, b in zip(a_values, b_values):
        plt.plot(x_line, 1/(1+np.exp(-(a + b*x_line))), color="green", alpha=0.1)
         
    #plt.plot(x_line, a_mean + b_mean*x_line, title = f"Linear, a = {a_mean}, b = {b_mean}", color = "red") 
    
    plt.show()
    """
    
if __name__ == "__main__":
    main()