data {
   int<lower=0> N_tr;
   int<lower=0> N_te;
   vector[N_tr] acc_tr;
   vector[N_tr] rot_tr;
   vector[N_te] acc_te;
   vector[N_te] rot_te;
}
parameters {
    real alpha0;
    real alpha;
    real beta0;
    real beta;
    real<lower=0> sigma0;
    real<lower=0> sigma[N_tr];
}
model {
    // 学習データだけでモデリング
    // 初期値の推定
    sigma[1] ~ cauchy(0, sigma0);
    acc_tr[1] ~ normal(0, sigma[1]);
    // 初期値以外の推定
    for (n in 2:N_tr){
        sigma[n] ~ cauchy(alpha0 + beta0 * rot_tr[n-1], sigma0);
        acc_tr[n] ~ normal(alpha + beta * acc_tr[n-1], sigma[n]);
    }
}
generated quantities {
    vector[N_tr] acc_tr_hat;
    vector[N_te] acc_te_hat;
    vector[N_tr] log_lik_tr;
    vector[N_te] log_lik_te;
    real sigma_tr_hat[N_tr];
    real sigma_te_hat[N_te];

    // 学習データについて予測
    // 初期値の推定
    sigma_tr_hat[1] = cauchy_rng(0, sigma0);
    acc_tr_hat[1] = normal_rng(0, sigma_tr_hat[1]);
    // 初期値以外の推定
    for (n in 2:N_tr){
        sigma_tr_hat[n] = normal_rng(alpha0 + beta0 * rot_tr[n-1], sigma0);
        acc_tr_hat[n] = normal_rng(alpha + beta * acc_tr[n-1], sigma_tr_hat[n]);
    }
    
    // テストデータについて予測
    // 初期値の推定
    sigma_te_hat[1] = cauchy_rng(0, sigma0);
    acc_te_hat[1] = normal_rng(0, sigma_te_hat[1]);
    // 初期値以外の推定
    for (n in 2:N_te){
        sigma_te_hat[n] = normal_rng(alpha0 + beta0 * rot_te[n-1], sigma0);
        acc_te_hat[n] = normal_rng(alpha + beta * acc_te[n-1], sigma_te_hat[n]);
    }
    
    // 学習データについて対数尤度を計算
    for (n in 1:N_tr){
        log_lik_tr[n] = normal_lpdf(acc_tr[n] | acc_tr_hat[n], sigma);
    }
    // テストデータについて対数尤度を計算
    for (n in 1:N_te){
        log_lik_te[n] = normal_lpdf(acc_te[n] | acc_te_hat[n], sigma);
    }

}
