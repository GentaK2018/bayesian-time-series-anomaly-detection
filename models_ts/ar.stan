data {
   int<lower=0> N_tr;
   int<lower=0> N_te;
   vector[N_tr] y_tr;
   vector[N_te] y_te;
}
parameters {
    real alpha0;
    real alpha;
    real beta;
    // real<lower=0, upper=2> sigma0;
    real<lower=0> sigma;
}
model {
    //学習データだけでモデリング
    // 初期値の推定
    y_tr[1] ~ normal(alpha0, sigma);
    // 初期値以外の推定
    for (n in 2:N_tr){
        y_tr[n] ~ normal(alpha + beta*y_tr[n-1], sigma);
    }
}
generated quantities {
    vector[N_tr] y_tr_hat;
    vector[N_te] y_te_hat;
    vector[N_tr] log_lik_tr;
    vector[N_te] log_lik_te;

    // 学習データについて予測
    // 初期値の推定
    y_tr_hat[1] = normal_rng(alpha0, sigma);
    // 初期値以外の推定
    for (n in 2:N_tr){
        y_tr_hat[n] = normal_rng(alpha + beta*y_tr[n-1], sigma);
    }
    // テストデータについて予測
    // テストデータの初期値を学習データの次に来るのでこのように推定
    y_te_hat[1] = normal_rng(alpha + beta*y_tr[N_tr], sigma);
    for (n in 2:N_te){
        y_te_hat[n] = normal_rng(alpha + beta*y_te[n-1], sigma);
    }
    // 学習データについて対数尤度を計算
    for (n in 1:N_tr){
        log_lik_tr[n] = normal_lpdf(y_tr[n] | y_tr_hat[n], sigma);
    }
    // テストデータについて対数尤度を計算
    for (n in 1:N_te){
        log_lik_te[n] = normal_lpdf(y_te[n] | y_te_hat[n], sigma);
    }

}
