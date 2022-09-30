#ifndef __HORIZON__ILQR__H__
#define __HORIZON__ILQR__H__

#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <memory>
#include <variant>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>

#include "profiling.h"
#include "iterate_filter.h"

namespace horizon
{

typedef Eigen::Ref<const Eigen::VectorXd> VecConstRef;
typedef Eigen::Ref<const Eigen::MatrixXd> MatConstRef;


/**
 * @brief IterativeLQR implements a multiple-shooting variant of the
 * notorious ILQR algorithm, implemented following the paper
 * "A Family of Iterative Gauss-Newton Shooting Methods for Nonlinear
 * Optimal Control" by M. Giftthaler, et al., from which most of the notation
 * is taken.
 *
 * The framework supports arbitrary (differentiable) discrete time dynamics
 * systems as well as arbitrary (twice differentiable) cost functions.
 *
 * Furthermore, arbitrary (differentiable) equality constraints are treated
 * with a projection approach.
 */
class IterativeLQR
{

public:

    /**
     *
     */
    struct ForwardPassResult;

    /**
     * @brief CallbackType
     */
    typedef std::function<bool(const ForwardPassResult& res)> CallbackType;

    typedef std::variant<int, double, bool, std::string> OptionTypes;
    typedef std::map<std::string, OptionTypes> OptionDict;


    /**
     * @brief Class constructor
     * @param fdyn is a function mapping state and control to the integrated state;
     * required signature is (x, u, p) -> (f)
     * @param N is the number of shooting intervals
     */
    IterativeLQR(casadi::Function fdyn,
                 int N,
                 OptionDict opt = OptionDict());


    void setStateBounds(const Eigen::MatrixXd& lb, const Eigen::MatrixXd& ub);

    void setInputBounds(const Eigen::MatrixXd& lb, const Eigen::MatrixXd& ub);

    /**
     * @brief set an intermediate cost term for the k-th intermediate state,
     * as specificed by a vector of indices
     * @param indices: the nodes that the cost refers to
     * @param inter_cost: a function with required signature (x, u, p) -> (l)
     */
    void setCost(std::vector<int> indices, const casadi::Function& inter_cost);

    void setResidual(std::vector<int> indices, const casadi::Function& inter_cost);

    /**
     * @brief set the final cost
     * @param final_cost: a function with required signature (x, u) -> (l),
     * even though the input 'u' is not used
     */
    void setFinalCost(const casadi::Function& final_cost);

    /**
     * @brief  set an intermediate constraint term for the k-th intermediate state,
     * as specificed by a vector of indices
     * @param indices: the nodes that the cost refers to
     * @param inter_constraint: a function with required signature (x, u, p) -> (h),
     * where the constraint is h(x, u) = 0
     * @param target_values: if specified, the i-th entry is used as target value
     * for the constraint function at the indices[i]
     */
    void setConstraint(std::vector<int> indices,
                       const casadi::Function& inter_constraint,
                       std::vector<Eigen::VectorXd> target_values = std::vector<Eigen::VectorXd>());

    void setFinalConstraint(const casadi::Function& final_constraint);

    void setIndices(const std::string& f_name,
                    const std::vector<int>& indices);

    void updateIndices();

    void setParameterValue(const std::string& pname, const Eigen::MatrixXd& value);

    void setInitialState(const Eigen::VectorXd& x0);

    void setStateInitialGuess(const Eigen::MatrixXd& x0);

    void setInputInitialGuess(const Eigen::MatrixXd& u0);

    void setIterationCallback(const CallbackType& cb);

    bool solve(int max_iter);

    const Eigen::MatrixXd& getStateTrajectory() const;

    const Eigen::MatrixXd& getInputTrajectory() const;

    const utils::ProfilingInfo& getProfilingInfo() const;

    VecConstRef state(int i) const;

    VecConstRef input(int i) const;

    MatConstRef gain(int i) const;

    ~IterativeLQR();

    struct ForwardPassResult
    {
        Eigen::MatrixXd xtrj;
        Eigen::MatrixXd utrj;
        double hxx_reg;
        double rho;
        double alpha;
        double cost;
        double bound_violation;
        double merit;
        double armijo_merit;
        double mu_f;
        double mu_c;
        double mu_b;
        double f_der;
        double merit_der;
        double step_length;
        double constraint_violation;
        double defect_norm;
        int iter;
        bool accepted;

        Eigen::VectorXd constraint_values;
        Eigen::MatrixXd defect_values;

        ForwardPassResult(int nx, int nu, int N);
        void print() const;
    };




protected:

private:

    static constexpr double inf = std::numeric_limits<double>::infinity();

    struct ConstrainedDynamics;
    struct ConstrainedCost;
    struct FeasibleConstraint;
    struct Dynamics;
    struct Constraint;
    struct IntermediateCost;
    struct ConstraintEntity;
    struct CostEntityBase;
    struct BoundAuglagCostEntity;
    struct IntermediateCostEntity;
    struct IntermediateResidualEntity;
    struct Temporaries;
    struct ConstraintToGo;
    struct BackwardPassResult;
    struct ValueFunction;

    typedef std::tuple<int, ConstrainedDynamics, ConstrainedCost>
        HandleConstraintsRetType;

    typedef std::shared_ptr<std::map<std::string, Eigen::MatrixXd>>
        ParameterMapPtr;

    typedef std::map<std::string, std::shared_ptr<CostEntityBase>>
        CostPtrMap;

    typedef std::map<std::string, std::shared_ptr<ConstraintEntity>>
        ConstraintPtrMap;

    void init_thread_pool(int pool_size);

    void add_param_to_map(const casadi::Function& f);

    void linearize_quadratize();

    void linearize_quadratize_inner(int i);

    void linearize_quadratize_mt();

    void linearize_quadratize_thread_main(int istart, int iend);

    void report_result(const ForwardPassResult& fpres);

    void backward_pass();

    void backward_pass_iter(int i);

    void optimize_initial_state();

    void increase_regularization();

    void reduce_regularization();

    FeasibleConstraint handle_constraints(int i);

    void add_bound_constraint(int k);

    bool auglag_update();

    double compute_merit_value(double mu_f,
                               double mu_c,
                               double cost,
                               double defect_norm,
                               double constr_viol);

    double compute_merit_slope(double cost_slope,
                               double mu_f,
                               double mu_c,
                               double defect_norm,
                               double constr_viol);

    double compute_cost_slope();

    std::pair<double, double> compute_merit_weights(double cost_der, double defect_norm, double constr_viol);

    double compute_cost(const Eigen::MatrixXd& xtrj,
                        const Eigen::MatrixXd& utrj);

    double compute_bound_penalty(const Eigen::MatrixXd& xtrj,
                                 const Eigen::MatrixXd& utrj);

    double compute_constr(const Eigen::MatrixXd& xtrj,
                          const Eigen::MatrixXd& utrj);

    double compute_defect(const Eigen::MatrixXd& xtrj,
                          const Eigen::MatrixXd& utrj);

    bool forward_pass(double alpha);

    void forward_pass_iter(int i, double alpha);

    bool line_search(int iter);

    void reset_iterate_filter();

    bool should_stop();

    void set_default_cost();

    bool fixed_initial_state();



    enum DecompositionType
    {
        Ldlt, Qr, Lu, Cod, Svd
    };

    static DecompositionType str_to_decomp_type(const std::string& dt_str);

    bool _verbose;
    bool _log;

    const int _nx;
    const int _nu;
    const int _N;

    double _step_length;
    double _rho_base;
    double _rho;
    double _rho_growth_factor;
    double _hxx_reg;
    double _hxx_reg_base;
    double _hxx_reg_growth_factor;
    double _huu_reg;
    double _kkt_reg;
    double _line_search_accept_ratio;
    double _alpha_min;
    double _svd_threshold;
    double _constraint_violation_threshold;
    double _defect_norm_threshold;
    double _merit_der_threshold;
    double _step_length_threshold;
    bool _enable_line_search;
    bool _enable_auglag;

    bool _closed_loop_forward_pass;
    std::string _codegen_workdir;
    bool _codegen_enabled;
    DecompositionType _kkt_decomp_type;
    DecompositionType _constr_decomp_type;

    ParameterMapPtr _param_map;
    CostPtrMap _cost_map;
    ConstraintPtrMap _constr_map;

    std::vector<std::shared_ptr<BoundAuglagCostEntity>> _auglag_cost;
    std::vector<IntermediateCost> _cost;
    std::vector<Constraint> _constraint;
    Eigen::MatrixXd _x_lb, _x_ub;
    Eigen::MatrixXd _u_lb, _u_ub;
    std::vector<ValueFunction> _value;
    std::vector<Dynamics> _dyn;

    std::vector<BackwardPassResult> _bp_res;
    std::unique_ptr<ConstraintToGo> _constraint_to_go;
    std::unique_ptr<ForwardPassResult> _fp_res;
    int _fp_accepted;

    IterateFilter _it_filt;
    bool _use_it_filter;

    Eigen::MatrixXd _xtrj;
    Eigen::MatrixXd _utrj;
    std::vector<Eigen::VectorXd> _lam_g;
    Eigen::MatrixXd _lam_x;

    Eigen::MatrixXd _lam_bound_x;
    Eigen::MatrixXd _lam_bound_u;

    std::vector<Temporaries> _tmp;

    std::vector<std::thread> _th_pool;
    std::condition_variable _th_work_avail_cond;
    std::mutex _th_work_avail_mtx;
    std::condition_variable _th_work_done_cond;
    std::mutex _th_work_done_mtx;
    int _th_done_flag;
    int _th_work_available_flag;
    std::atomic<bool> _th_exit;

    CallbackType _iter_cb;
    utils::ProfilingInfo _prof_info;
};



}

#endif
