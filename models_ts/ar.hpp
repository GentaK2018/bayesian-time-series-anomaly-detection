// Code generated by stanc v2.32.2
#include <stan/model/model_header.hpp>
namespace ar_model_namespace {
using stan::model::model_base_crtp;
using namespace stan::math;
stan::math::profile_map profiles__;
static constexpr std::array<const char*, 37> locations_array__ =
  {" (found before start of program)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 8, column 4 to column 16)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 9, column 4 to column 15)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 10, column 4 to column 14)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 12, column 4 to column 24)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 24, column 4 to column 26)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 25, column 4 to column 26)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 26, column 4 to column 28)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 27, column 4 to column 28)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 31, column 4 to column 44)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 34, column 8 to column 64)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 33, column 21 to line 35, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 33, column 4 to line 35, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 38, column 4 to column 61)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 40, column 8 to column 64)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 39, column 21 to line 41, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 39, column 4 to line 41, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 44, column 8 to column 66)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 43, column 21 to line 45, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 43, column 4 to line 45, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 48, column 8 to column 66)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 47, column 21 to line 49, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 47, column 4 to line 49, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 17, column 4 to column 36)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 20, column 8 to column 56)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 19, column 21 to line 21, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 19, column 4 to line 21, column 5)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 2, column 3 to column 21)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 3, column 3 to column 21)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 4, column 10 to column 14)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 4, column 3 to column 21)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 5, column 10 to column 14)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 5, column 3 to column 21)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 24, column 11 to column 15)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 25, column 11 to column 15)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 26, column 11 to column 15)",
  " (in '/home/karinog/work/study/bayesian-time-series-anomaly-detection/models_ts/ar.stan', line 27, column 11 to column 15)"};
class ar_model final : public model_base_crtp<ar_model> {
 private:
  int N_tr;
  int N_te;
  Eigen::Matrix<double,-1,1> y_tr_data__;
  Eigen::Matrix<double,-1,1> y_te_data__;
  Eigen::Map<Eigen::Matrix<double,-1,1>> y_tr{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double,-1,1>> y_te{nullptr, 0};
 public:
  ~ar_model() {}
  ar_model(stan::io::var_context& context__, unsigned int random_seed__ = 0,
           std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double;
    boost::ecuyer1988 base_rng__ =
      stan::services::util::create_rng(random_seed__, 0);
    // suppress unused var warning
    (void) base_rng__;
    static constexpr const char* function__ = "ar_model_namespace::ar_model";
    // suppress unused var warning
    (void) function__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 27;
      context__.validate_dims("data initialization", "N_tr", "int",
        std::vector<size_t>{});
      N_tr = std::numeric_limits<int>::min();
      current_statement__ = 27;
      N_tr = context__.vals_i("N_tr")[(1 - 1)];
      current_statement__ = 27;
      stan::math::check_greater_or_equal(function__, "N_tr", N_tr, 0);
      current_statement__ = 28;
      context__.validate_dims("data initialization", "N_te", "int",
        std::vector<size_t>{});
      N_te = std::numeric_limits<int>::min();
      current_statement__ = 28;
      N_te = context__.vals_i("N_te")[(1 - 1)];
      current_statement__ = 28;
      stan::math::check_greater_or_equal(function__, "N_te", N_te, 0);
      current_statement__ = 29;
      stan::math::validate_non_negative_index("y_tr", "N_tr", N_tr);
      current_statement__ = 30;
      context__.validate_dims("data initialization", "y_tr", "double",
        std::vector<size_t>{static_cast<size_t>(N_tr)});
      y_tr_data__ = Eigen::Matrix<double,-1,1>::Constant(N_tr,
                      std::numeric_limits<double>::quiet_NaN());
      new (&y_tr) Eigen::Map<Eigen::Matrix<double,-1,1>>(y_tr_data__.data(),
        N_tr);
      {
        std::vector<local_scalar_t__> y_tr_flat__;
        current_statement__ = 30;
        y_tr_flat__ = context__.vals_r("y_tr");
        current_statement__ = 30;
        pos__ = 1;
        current_statement__ = 30;
        for (int sym1__ = 1; sym1__ <= N_tr; ++sym1__) {
          current_statement__ = 30;
          stan::model::assign(y_tr, y_tr_flat__[(pos__ - 1)],
            "assigning variable y_tr", stan::model::index_uni(sym1__));
          current_statement__ = 30;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 31;
      stan::math::validate_non_negative_index("y_te", "N_te", N_te);
      current_statement__ = 32;
      context__.validate_dims("data initialization", "y_te", "double",
        std::vector<size_t>{static_cast<size_t>(N_te)});
      y_te_data__ = Eigen::Matrix<double,-1,1>::Constant(N_te,
                      std::numeric_limits<double>::quiet_NaN());
      new (&y_te) Eigen::Map<Eigen::Matrix<double,-1,1>>(y_te_data__.data(),
        N_te);
      {
        std::vector<local_scalar_t__> y_te_flat__;
        current_statement__ = 32;
        y_te_flat__ = context__.vals_r("y_te");
        current_statement__ = 32;
        pos__ = 1;
        current_statement__ = 32;
        for (int sym1__ = 1; sym1__ <= N_te; ++sym1__) {
          current_statement__ = 32;
          stan::model::assign(y_te, y_te_flat__[(pos__ - 1)],
            "assigning variable y_te", stan::model::index_uni(sym1__));
          current_statement__ = 32;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 33;
      stan::math::validate_non_negative_index("y_tr_hat", "N_tr", N_tr);
      current_statement__ = 34;
      stan::math::validate_non_negative_index("y_te_hat", "N_te", N_te);
      current_statement__ = 35;
      stan::math::validate_non_negative_index("log_lik_tr", "N_tr", N_tr);
      current_statement__ = 36;
      stan::math::validate_non_negative_index("log_lik_te", "N_te", N_te);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = 1 + 1 + 1 + 1;
  }
  inline std::string model_name() const final {
    return "ar_model";
  }
  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.32.2",
             "stancflags = "};
  }
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ = "ar_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      local_scalar_t__ alpha0 = DUMMY_VAR__;
      current_statement__ = 1;
      alpha0 = in__.template read<local_scalar_t__>();
      local_scalar_t__ alpha = DUMMY_VAR__;
      current_statement__ = 2;
      alpha = in__.template read<local_scalar_t__>();
      local_scalar_t__ beta = DUMMY_VAR__;
      current_statement__ = 3;
      beta = in__.template read<local_scalar_t__>();
      local_scalar_t__ sigma = DUMMY_VAR__;
      current_statement__ = 4;
      sigma = in__.template read_constrain_lb<local_scalar_t__,
                jacobian__>(0, lp__);
      {
        current_statement__ = 23;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(
                         stan::model::rvalue(y_tr, "y_tr",
                           stan::model::index_uni(1)), alpha0, sigma));
        current_statement__ = 26;
        for (int n = 2; n <= N_tr; ++n) {
          current_statement__ = 24;
          lp_accum__.add(stan::math::normal_lpdf<propto__>(
                           stan::model::rvalue(y_tr, "y_tr",
                             stan::model::index_uni(n)), (alpha + (beta *
                           stan::model::rvalue(y_tr, "y_tr",
                             stan::model::index_uni((n - 1))))), sigma));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  template <typename RNG, typename VecR, typename VecI, typename VecVar,
            stan::require_vector_like_vt<std::is_floating_point,
            VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral,
            VecI>* = nullptr, stan::require_vector_vt<std::is_floating_point,
            VecVar>* = nullptr>
  inline void
  write_array_impl(RNG& base_rng__, VecR& params_r__, VecI& params_i__,
                   VecVar& vars__, const bool
                   emit_transformed_parameters__ = true, const bool
                   emit_generated_quantities__ = true, std::ostream*
                   pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    // suppress unused var warning
    (void) propto__;
    double lp__ = 0.0;
    // suppress unused var warning
    (void) lp__;
    int current_statement__ = 0;
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    constexpr bool jacobian__ = false;
    static constexpr const char* function__ =
      "ar_model_namespace::write_array";
    // suppress unused var warning
    (void) function__;
    try {
      double alpha0 = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 1;
      alpha0 = in__.template read<local_scalar_t__>();
      double alpha = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 2;
      alpha = in__.template read<local_scalar_t__>();
      double beta = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 3;
      beta = in__.template read<local_scalar_t__>();
      double sigma = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 4;
      sigma = in__.template read_constrain_lb<local_scalar_t__,
                jacobian__>(0, lp__);
      out__.write(alpha0);
      out__.write(alpha);
      out__.write(beta);
      out__.write(sigma);
      if (stan::math::logical_negation(
            (stan::math::primitive_value(emit_transformed_parameters__) ||
            stan::math::primitive_value(emit_generated_quantities__)))) {
        return ;
      }
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      }
      Eigen::Matrix<double,-1,1> y_tr_hat =
        Eigen::Matrix<double,-1,1>::Constant(N_tr,
          std::numeric_limits<double>::quiet_NaN());
      Eigen::Matrix<double,-1,1> y_te_hat =
        Eigen::Matrix<double,-1,1>::Constant(N_te,
          std::numeric_limits<double>::quiet_NaN());
      Eigen::Matrix<double,-1,1> log_lik_tr =
        Eigen::Matrix<double,-1,1>::Constant(N_tr,
          std::numeric_limits<double>::quiet_NaN());
      Eigen::Matrix<double,-1,1> log_lik_te =
        Eigen::Matrix<double,-1,1>::Constant(N_te,
          std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 9;
      stan::model::assign(y_tr_hat,
        stan::math::normal_rng(alpha0, sigma, base_rng__),
        "assigning variable y_tr_hat", stan::model::index_uni(1));
      current_statement__ = 12;
      for (int n = 2; n <= N_tr; ++n) {
        current_statement__ = 10;
        stan::model::assign(y_tr_hat,
          stan::math::normal_rng((alpha + (beta *
            stan::model::rvalue(y_tr, "y_tr", stan::model::index_uni((n - 1))))),
            sigma, base_rng__), "assigning variable y_tr_hat",
          stan::model::index_uni(n));
      }
      current_statement__ = 13;
      stan::model::assign(y_te_hat,
        stan::math::normal_rng((alpha + (beta *
          stan::model::rvalue(y_tr, "y_tr", stan::model::index_uni(N_tr)))),
          sigma, base_rng__), "assigning variable y_te_hat",
        stan::model::index_uni(1));
      current_statement__ = 16;
      for (int n = 2; n <= N_te; ++n) {
        current_statement__ = 14;
        stan::model::assign(y_te_hat,
          stan::math::normal_rng((alpha + (beta *
            stan::model::rvalue(y_te, "y_te", stan::model::index_uni((n - 1))))),
            sigma, base_rng__), "assigning variable y_te_hat",
          stan::model::index_uni(n));
      }
      current_statement__ = 19;
      for (int n = 1; n <= N_tr; ++n) {
        current_statement__ = 17;
        stan::model::assign(log_lik_tr,
          stan::math::normal_lpdf<false>(
            stan::model::rvalue(y_tr, "y_tr", stan::model::index_uni(n)),
            stan::model::rvalue(y_tr_hat, "y_tr_hat",
              stan::model::index_uni(n)), sigma),
          "assigning variable log_lik_tr", stan::model::index_uni(n));
      }
      current_statement__ = 22;
      for (int n = 1; n <= N_te; ++n) {
        current_statement__ = 20;
        stan::model::assign(log_lik_te,
          stan::math::normal_lpdf<false>(
            stan::model::rvalue(y_te, "y_te", stan::model::index_uni(n)),
            stan::model::rvalue(y_te_hat, "y_te_hat",
              stan::model::index_uni(n)), sigma),
          "assigning variable log_lik_te", stan::model::index_uni(n));
      }
      out__.write(y_tr_hat);
      out__.write(y_te_hat);
      out__.write(log_lik_tr);
      out__.write(log_lik_te);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, typename VecI,
            stan::require_vector_t<VecVar>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void
  unconstrain_array_impl(const VecVar& params_r__, const VecI& params_i__,
                         VecVar& vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ alpha0 = DUMMY_VAR__;
      current_statement__ = 1;
      alpha0 = in__.read<local_scalar_t__>();
      out__.write(alpha0);
      local_scalar_t__ alpha = DUMMY_VAR__;
      current_statement__ = 2;
      alpha = in__.read<local_scalar_t__>();
      out__.write(alpha);
      local_scalar_t__ beta = DUMMY_VAR__;
      current_statement__ = 3;
      beta = in__.read<local_scalar_t__>();
      out__.write(beta);
      local_scalar_t__ sigma = DUMMY_VAR__;
      current_statement__ = 4;
      sigma = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, sigma);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, stan::require_vector_t<VecVar>* = nullptr>
  inline void
  transform_inits_impl(const stan::io::var_context& context__, VecVar&
                       vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 1;
      context__.validate_dims("parameter initialization", "alpha0", "double",
        std::vector<size_t>{});
      current_statement__ = 2;
      context__.validate_dims("parameter initialization", "alpha", "double",
        std::vector<size_t>{});
      current_statement__ = 3;
      context__.validate_dims("parameter initialization", "beta", "double",
        std::vector<size_t>{});
      current_statement__ = 4;
      context__.validate_dims("parameter initialization", "sigma", "double",
        std::vector<size_t>{});
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ alpha0 = DUMMY_VAR__;
      current_statement__ = 1;
      alpha0 = context__.vals_r("alpha0")[(1 - 1)];
      out__.write(alpha0);
      local_scalar_t__ alpha = DUMMY_VAR__;
      current_statement__ = 2;
      alpha = context__.vals_r("alpha")[(1 - 1)];
      out__.write(alpha);
      local_scalar_t__ beta = DUMMY_VAR__;
      current_statement__ = 3;
      beta = context__.vals_r("beta")[(1 - 1)];
      out__.write(beta);
      local_scalar_t__ sigma = DUMMY_VAR__;
      current_statement__ = 4;
      sigma = context__.vals_r("sigma")[(1 - 1)];
      out__.write_free_lb(0, sigma);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  inline void
  get_param_names(std::vector<std::string>& names__, const bool
                  emit_transformed_parameters__ = true, const bool
                  emit_generated_quantities__ = true) const {
    names__ = std::vector<std::string>{"alpha0", "alpha", "beta", "sigma"};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      std::vector<std::string>
        temp{"y_tr_hat", "y_te_hat", "log_lik_tr", "log_lik_te"};
      names__.reserve(names__.size() + temp.size());
      names__.insert(names__.end(), temp.begin(), temp.end());
    }
  }
  inline void
  get_dims(std::vector<std::vector<size_t>>& dimss__, const bool
           emit_transformed_parameters__ = true, const bool
           emit_generated_quantities__ = true) const {
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
                std::vector<size_t>{}, std::vector<size_t>{},
                std::vector<size_t>{}};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      std::vector<std::vector<size_t>>
        temp{std::vector<size_t>{static_cast<size_t>(N_tr)},
             std::vector<size_t>{static_cast<size_t>(N_te)},
             std::vector<size_t>{static_cast<size_t>(N_tr)},
             std::vector<size_t>{static_cast<size_t>(N_te)}};
      dimss__.reserve(dimss__.size() + temp.size());
      dimss__.insert(dimss__.end(), temp.begin(), temp.end());
    }
  }
  inline void
  constrained_param_names(std::vector<std::string>& param_names__, bool
                          emit_transformed_parameters__ = true, bool
                          emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "alpha0");
    param_names__.emplace_back(std::string() + "alpha");
    param_names__.emplace_back(std::string() + "beta");
    param_names__.emplace_back(std::string() + "sigma");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N_tr; ++sym1__) {
        param_names__.emplace_back(std::string() + "y_tr_hat" + '.' +
          std::to_string(sym1__));
      }
      for (int sym1__ = 1; sym1__ <= N_te; ++sym1__) {
        param_names__.emplace_back(std::string() + "y_te_hat" + '.' +
          std::to_string(sym1__));
      }
      for (int sym1__ = 1; sym1__ <= N_tr; ++sym1__) {
        param_names__.emplace_back(std::string() + "log_lik_tr" + '.' +
          std::to_string(sym1__));
      }
      for (int sym1__ = 1; sym1__ <= N_te; ++sym1__) {
        param_names__.emplace_back(std::string() + "log_lik_te" + '.' +
          std::to_string(sym1__));
      }
    }
  }
  inline void
  unconstrained_param_names(std::vector<std::string>& param_names__, bool
                            emit_transformed_parameters__ = true, bool
                            emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "alpha0");
    param_names__.emplace_back(std::string() + "alpha");
    param_names__.emplace_back(std::string() + "beta");
    param_names__.emplace_back(std::string() + "sigma");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N_tr; ++sym1__) {
        param_names__.emplace_back(std::string() + "y_tr_hat" + '.' +
          std::to_string(sym1__));
      }
      for (int sym1__ = 1; sym1__ <= N_te; ++sym1__) {
        param_names__.emplace_back(std::string() + "y_te_hat" + '.' +
          std::to_string(sym1__));
      }
      for (int sym1__ = 1; sym1__ <= N_tr; ++sym1__) {
        param_names__.emplace_back(std::string() + "log_lik_tr" + '.' +
          std::to_string(sym1__));
      }
      for (int sym1__ = 1; sym1__ <= N_te; ++sym1__) {
        param_names__.emplace_back(std::string() + "log_lik_te" + '.' +
          std::to_string(sym1__));
      }
    }
  }
  inline std::string get_constrained_sizedtypes() const {
    return std::string("[{\"name\":\"alpha0\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"alpha\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"y_tr_hat\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_tr) + "},\"block\":\"generated_quantities\"},{\"name\":\"y_te_hat\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_te) + "},\"block\":\"generated_quantities\"},{\"name\":\"log_lik_tr\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_tr) + "},\"block\":\"generated_quantities\"},{\"name\":\"log_lik_te\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_te) + "},\"block\":\"generated_quantities\"}]");
  }
  inline std::string get_unconstrained_sizedtypes() const {
    return std::string("[{\"name\":\"alpha0\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"alpha\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"y_tr_hat\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_tr) + "},\"block\":\"generated_quantities\"},{\"name\":\"y_te_hat\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_te) + "},\"block\":\"generated_quantities\"},{\"name\":\"log_lik_tr\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_tr) + "},\"block\":\"generated_quantities\"},{\"name\":\"log_lik_te\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_te) + "},\"block\":\"generated_quantities\"}]");
  }
  // Begin method overload boilerplate
  template <typename RNG> inline void
  write_array(RNG& base_rng, Eigen::Matrix<double,-1,1>& params_r,
              Eigen::Matrix<double,-1,1>& vars, const bool
              emit_transformed_parameters = true, const bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (((1 + 1) + 1) + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * ((((N_tr +
      N_te) + N_tr) + N_te));
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    std::vector<int> params_i;
    vars = Eigen::Matrix<double,-1,1>::Constant(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <typename RNG> inline void
  write_array(RNG& base_rng, std::vector<double>& params_r, std::vector<int>&
              params_i, std::vector<double>& vars, bool
              emit_transformed_parameters = true, bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (((1 + 1) + 1) + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * ((((N_tr +
      N_te) + N_tr) + N_te));
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    vars = std::vector<double>(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(Eigen::Matrix<T_,-1,1>& params_r, std::ostream* pstream = nullptr) const {
    Eigen::Matrix<int,-1,1> params_i;
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(std::vector<T_>& params_r, std::vector<int>& params_i,
           std::ostream* pstream = nullptr) const {
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  inline void
  transform_inits(const stan::io::var_context& context,
                  Eigen::Matrix<double,-1,1>& params_r, std::ostream*
                  pstream = nullptr) const final {
    std::vector<double> params_r_vec(params_r.size());
    std::vector<int> params_i;
    transform_inits(context, params_i, params_r_vec, pstream);
    params_r = Eigen::Map<Eigen::Matrix<double,-1,1>>(params_r_vec.data(),
                 params_r_vec.size());
  }
  inline void
  transform_inits(const stan::io::var_context& context, std::vector<int>&
                  params_i, std::vector<double>& vars, std::ostream*
                  pstream__ = nullptr) const {
    vars.resize(num_params_r__);
    transform_inits_impl(context, vars, pstream__);
  }
  inline void
  unconstrain_array(const std::vector<double>& params_constrained,
                    std::vector<double>& params_unconstrained, std::ostream*
                    pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = std::vector<double>(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
  inline void
  unconstrain_array(const Eigen::Matrix<double,-1,1>& params_constrained,
                    Eigen::Matrix<double,-1,1>& params_unconstrained,
                    std::ostream* pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = Eigen::Matrix<double,-1,1>::Constant(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
};
}
using stan_model = ar_model_namespace::ar_model;
#ifndef USING_R
// Boilerplate
stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
stan::math::profile_map& get_stan_profile_data() {
  return ar_model_namespace::profiles__;
}
#endif