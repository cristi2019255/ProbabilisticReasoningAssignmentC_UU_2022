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


def get_stan_code(question = "Q1"):
    stan_code_q1 = """
        data {
            int<lower=0> N;
            real d18_O_w[N];
            real d18_O_c[N];
            real y[N]; // temperature
        }
        
        parameters {
            real a;
            real b;
            real sigma;
        }
        
        model {
            for (i in 1:N)
                y[i] ~ normal(a + b * (d18_O_c[i] - d18_O_w[i]), sigma);                    
        }
    """
    
    stan_code_q2 = """
        data {
            int<lower=0> N;
            real d18_O_w[N];
            real d18_O_c[N];
            real<lower=-2, upper=50> y[N]; // temperature
        }
        
        parameters {
            real a;
            real b;
            real sigma;
        }
        
        transformed parameters {
            real<lower=-4, upper=5> diff[N];  
            for (i in 1:N)
                diff[i] = d18_O_c[i] - d18_O_w[i];        // difference between the two d18_O values
        }
            
        model {
            for (i in 1:N)
                y[i] ~ normal(a + b * diff[i], sigma - 1);                    
        }
        
        generated quantities {
            vector[N] y_new;
            
            for (k in 1:N)
                y_new[k] = normal_rng(a + b * (d18_O_c[k] - d18_O_w[k]), sigma - 1);
        }
    """
    
    stan_code_q3 = """
    data {            
            int<lower=0> N; // number of observations
            real d18_O_w[N]; // d18_O of water
            real d18_O_c[N]; // d18_O of carbon
            real<lower=-2, upper=50> y[N]; // temperature
        }
        
        parameters {
            real a; // intercept
            real b; // slope
            real sigma; // standard deviation
        }
            
        model {
            for (i in 1:N)
                y[i] ~ normal(a + b * (d18_O_c[i] - d18_O_w[i]), sigma);                    
        }
    """
    
    stan_code_q3_mix_pooling = """
        data {            
            int<lower=0> N; // number of observations
            real d18_O_w[N]; // d18_O of water
            real d18_O_c[N]; // d18_O of carbon
            real<lower=-2, upper=50> y[N]; // temperature
            real a_m; // intercept_mean
            real b_m; // slope_mean
            real sigma_a; // intercept_std
            real sigma_b; // slope_std
        }
        
        parameters {
            real a; // intercept
            real b; // slope
            real sigma; // standard deviation
        }
        
        transformed parameters {
            real theta[N];
            
            for (i in 1:N)
                theta[i] = a + b * (d18_O_c[i] - d18_O_w[i]);        // difference between the two observed d18_O values
        }
            
        model {
            target += normal_lpdf(a | a_m, sigma_a);
            target += normal_lpdf(b | b_m, sigma_b);
            target += normal_lpdf(y | theta, sigma);
        }
    """
    
    stan_code_q4 = """
        data {            
            int<lower=0> N; // number of observations
            int<lower=0> K; // number of new observations
            
            real d18_O_w[N]; // d18_O of water
            real d18_O_c[N]; // d18_O of carbon
            real<lower=-2, upper=50> y[N]; // temperature
            real a_m; // intercept_mean
            real b_m; // slope_mean
            real sigma_a; // intercept_std
            real sigma_b; // slope_std
            real d18_O_w_new[K]; // new data
            real d18_O_c_new[K]; // new data
        }
        
        
        transformed data {
            real x_new[K];
            for (i in 1:K)
                x_new[i] = (d18_O_c_new[i] - d18_O_w_new[i]);   // difference between the two new d18_O values
        }
        
        
        parameters {
            real a; // intercept
            real b; // slope
            real sigma; // standard deviation
        }
        
        transformed parameters {
            real theta[N];  
            
            for (i in 1:N)
                theta[i] = a + b * (d18_O_c[i] - d18_O_w[i]);        // difference between the two observed d18_O values
        
        }
        
            
        model {
            
            target += normal_lpdf(a | a_m, sigma_a);
            target += normal_lpdf(b | b_m, sigma_b);
            target += normal_lpdf(y | theta, sigma);
        }
        
        generated quantities {
            vector[K] y_new;
            for (k in 1:K)
                y_new[k] = normal_rng(a + b * x_new[k], sigma);
        }
    """

    stan_code_q4_b = """
          data {            
            int<lower=0> N; // number of observations
            int<lower=0> K; // number of new observations
            
            real d18_O_w[N]; // d18_O of water
            real d18_O_c[N]; // d18_O of carbon
            real<lower=-2, upper=50> y[N]; // temperature
            real a_m; // intercept_mean
            real b_m; // slope_mean
            real sigma_a; // intercept_std
            real sigma_b; // slope_std
            real d18_O_w_new[K]; // new data
            real d18_O_c_new[K]; // new data
        
            real d18_O_c_std; // new data std
            real d18_O_w_std; // new data std
        }
        
        
        parameters {
            real a; // intercept
            real b; // slope
            real sigma; // standard deviation
            real d18_O_c_s[K];
            real d18_O_w_s[K]; 
        }
        
        transformed parameters {
            real theta[N];  
            
            for (i in 1:N)
                theta[i] = a + b * (d18_O_c[i] - d18_O_w[i]);        // difference between the two observed d18_O values
        
        }
        
            
        model {
            
            target += normal_lpdf(a | a_m, sigma_a);
            target += normal_lpdf(b | b_m, sigma_b);
            target += normal_lpdf(y | theta, sigma);

            d18_O_c_s ~ normal(d18_O_c_new, d18_O_c_std);
            d18_O_w_s ~ normal(d18_O_w_new, d18_O_w_std);
            
        }
        
        generated quantities {
            vector[K] y_new;
            
            for (k in 1:K)
                y_new[k] = normal_rng(a + b * (d18_O_c_s[k] - d18_O_w_s[k]), sigma);
        }
    """
    
    
    stan_codes = {
       "Q1": stan_code_q1,
       "Q2": stan_code_q2,
       "Q3_A": stan_code_q3,
       "Q3_B": stan_code_q3_mix_pooling,
       "Q4_A": stan_code_q4,
       "Q4_B": stan_code_q4_b,
    }

    return stan_codes[question]
