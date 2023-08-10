data {
   int<lower=0> N;
   int<lower=0> K;
   matrix<lower=0, upper=1>[N,K] X;
   vector<lower=0, upper=1>[N] y;
}

parameters {
   real alpha;
   vector[K] beta;
   real<lower=0> sigma;
}

transformed parameters {
   vector[N] mu;
   mu = alpha + X*beta;
}

model {
   y ~ normal(mu, sigma);
}

generated quantities {
   array[N] real y_pred;
   y_pred = normal_rng(mu, sigma);
}