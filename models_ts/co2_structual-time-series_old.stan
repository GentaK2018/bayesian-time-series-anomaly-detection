data {
    int<lower=0> T_tr;
    vector[T_tr] y_tr;
    int<lower=0> T_te;
    vector[T_te] y_te;
    int<lower=0> S;
}

parameters {
    real<lower=0> sigma_mu;  // トレンド成分の水準成分ノイズパラメータ
    real<lower=0> sigma_delta;  // トレンド成分の傾き成分ノイズパラメータ
    real<lower=0> sigma_seasonal;  // 季節成分のノイズパラメータ
    real<lower=0> sigma; // データ生成過程のノイズ

    vector[T_tr] mu;  // トレンド成分のμ
    vector[T_tr] delta;  // トレンド成分のδ
    vector[T_tr] seasonal;  // 季節成分
}

model {
    // トレンド成分の事前分布
    mu[2:T_tr] ~ normal(mu[1:(T_tr-1)] + delta[1:(T_tr-1)], sigma_mu);
    delta[2:T_tr] ~ normal(delta[1:(T_tr-1)], sigma_delta);
    
    // 季節成分の事前分布
    for (t in S:T_tr){
        seasonal[t] ~ normal(-1 * sum(seasonal[(t-S+1):(t-1)]), sigma_seasonal);
    }

    // データの生成過程
    y_tr ~ normal(mu + seasonal, sigma);

}

generated quantities {
    vector[T_tr] y_tr_hat;
    vector[T_te] y_te_hat;
    vector[T_tr+T_te] seasonal_hat;
    vector[T_tr+T_te] mu_hat;
    vector[T_tr+T_te] delta_hat;

    // 学習データの期間は推定済みのパラメータをそのまま使う
    mu_hat[1:T_tr] = mu[1:T_tr];
    delta_hat[1:T_tr] = delta[1:T_tr];
    seasonal_hat[1:T_tr] = seasonal[1:T_tr];

    for (t in 2:(T_tr+T_te)){
        if (t <= T_tr){
            y_tr_hat[t] = normal_rng(mu_hat[t] + seasonal_hat[t], sigma);
        }
        else {
            mu_hat[t] = normal_rng(mu_hat[t-1] + delta_hat[t-1], sigma_mu);
            delta_hat[t] = normal_rng(delta_hat[t-1], sigma_delta);
            seasonal_hat[t] = normal_rng(-1 * sum(seasonal_hat[(t-S+1):(t-1)]), sigma_seasonal);
            y_te_hat[t-T_tr] = normal_rng(mu_hat[t] + seasonal_hat[t], sigma);
        }
    }
}