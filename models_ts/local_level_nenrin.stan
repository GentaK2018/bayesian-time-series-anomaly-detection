data {
    int<lower=0> N_tr;  // 学習データ数
    vector[N_tr] y_tr;
}

parameters {
    vector[N_tr] alpha;
    vector<lower=0>[2] sigma;
    // vector<lower=0>[2] tau;
}

model {
    // 事前分布
    alpha[1] ~ normal(0,10);
    for (i in 1:2){
        sigma[i] ~ cauchy(0,5);
        // tau[i] ~ normal(0, sigma[i]);
    }

    // 観測モデル
    y_tr ~ normal(alpha, sigma[1]);
    
    // システムモデル
    // for (i in 2:N_tr){
    //     alpha[i] ~ normal(alpha[i-1], tau[2]);
    // }
    alpha[2:N_tr] ~ normal(alpha[1:(N_tr-1)], sigma[2]);
}

generated quantities {
    array[N_tr] real y_tr_hat;

    y_tr_hat = normal_rng(alpha, sigma[1]);
}