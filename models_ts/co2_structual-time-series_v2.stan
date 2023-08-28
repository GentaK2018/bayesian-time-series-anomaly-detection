data {
    int<lower=0> T_tr;
    vector[T_tr] y_tr;
    int<lower=0> T_te;
    vector[T_te] y_te;
    int<lower=0> S;
}

parameters {
    real<lower=0> s_z;  // ドリフト成分の変動の大きさを表す標準偏差
    real<lower=0> s_v;  // 観測誤差の標準偏差
    real<lower=0> s_s;  // 季節成分の大きさを表す標準偏差

    vector[T_tr] mu;  // 水準＋ドリフト成分の推定値
    vector[T_tr] gamma;  // 季節成分の推定値
}

transformed parameters {
    vector[T_tr] alpha;
    // for(i in 1:T_tr){
    //     alpha[i] = mu[i] + gamma[i];
    // }
    alpha = mu + gamma;
}

model {
    // 水準＋ドリフト成分
    for(i in 3:T_tr){
        mu[i] ~ normal(2*mu[i-1]-mu[i-2], s_z);
    }

    // 季節成分
    for(i in S:T_tr){
        gamma[i] ~ normal(-sum(gamma[(i-S+1):(i-1)]), s_s);
    }

    // 観測方程式に従い、観測値が得られる
    y_tr ~ normal(alpha, s_v);
}

generated quantities {
    vector[T_tr] y_tr_hat;
    vector[T_te] y_te_hat;
    // vector[T_tr+T_te] seasonal_raw_hat;
    vector[T_tr+T_te] mu_hat;
    vector[T_tr+T_te] gamma_hat;

    // 学習データの期間は推定済みのパラメータをそのまま使う
    mu_hat[1:T_tr] = mu[1:T_tr];
    gamma_hat[1:T_tr] = gamma[1:T_tr];

    for (t in 2:(T_tr+T_te)){
        if (t <= T_tr){
            // y_tr_hat[t] = normal_rng(trend_hat[t] + seasonal_hat[t] + irregular_hat[t], sigma);
            y_tr_hat[t] = normal_rng(mu_hat[t] + gamma_hat[t], s_v);
        }
        else {
            mu_hat[t] = normal_rng(2*mu_hat[t-1]-mu_hat[t-2], s_z);
            gamma_hat[t] = normal_rng(-1 * sum(gamma_hat[(t-S+1):(t-1)]), s_s);
            y_te_hat[t-T_tr] = normal_rng(mu_hat[t] + gamma_hat[t], s_v);
        }
    }
}