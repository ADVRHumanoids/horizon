#ifndef __HORIZON__ILQR__H__
#define __HORIZON__ILQR__H__

#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <variant>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>

#include "profiling.h"
#include "iterate_filter.h"
#include "typedefs.h"

namespace horizon
{

typedef Eigen::Ref<const VectorXr> VecConstRef;
typedef Eigen::Ref<const MatrixXr> MatConstRef;

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

    typedef std::variant<int, Real, bool, std::string> OptionTypes;
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


    void setStateBounds(const MatrixXr& lb, const MatrixXr& ub);

    void setInputBounds(const MatrixXr& lb, const MatrixXr& ub);

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
                       std::vector<VectorXr> target_values = std::vector<VectorXr>());

    void setFinalConstraint(const casadi::Function& final_constraint);

    void setIndices(const std::string& f_name,
                    const std::vector<int>& indices);

    void updateIndices();

    void setParameterValue(const std::string& pname, const MatrixXr& value);

    void setInitialState(const VectorXr& x0);

    void setStateInitialGuess(const MatrixXr& x0);

    void setInputInitialGuess(const MatrixXr& u0);

    void setIterationCallback(const CallbackType& cb);
    
    void reset();

    bool solve(int max_iter);

    const MatrixXr& getStateTrajectory() const;

    const MatrixXr& getInputTrajectory() const;

    const utils::ProfilingInfo& getProfilingInfo() const;

    const std::vector<ForwardPassResult>& getIterationHistory() const;

    const VectorXr& getCostValOnNodes() const;

    const std::map<std::string, MatrixXr>& getConstraintsValues() const;

    const VectorXr& getConstrValOnNodes() const;

    const std::map<std::string, VectorXr>& getCostsValues() const;

    const float getResidualNorm() const;

    VecConstRef state(int i) const;

    VecConstRef input(int i) const;

    MatConstRef gain(int i) const;

    ~IterativeLQR();

    struct ForwardPassResult
    {
        MatrixXr xtrj;
        MatrixXr utrj;
        Real hxx_reg;
        Real rho;
        Real alpha;
        Real cost;
        Real bound_violation;
        Real merit;
        Real armijo_merit;
        Real mu_f;
        Real mu_c;
        Real mu_b;
        Real f_der;
        Real merit_der;
        Real step_length;
        Real constraint_violation;
        Real defect_norm;
        int iter;
        bool accepted;

        VectorXr cost_values;
        VectorXr constraint_values;
        MatrixXr defect_values;

        ForwardPassResult(int nx, int nu, int N);

        void print(int N = 1) const;
    };


protected:

private:

    static constexpr Real inf = std::numeric_limits<Real>::infinity();

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

    typedef std::shared_ptr<std::map<std::string, MatrixXr>>
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

    void kkt_solve();

    void optimize_initial_state();

    void increase_regularization();

    void reduce_regularization();

    FeasibleConstraint handle_constraints(int i);

    void add_bound_constraint(int k);

    bool auglag_update();

    Real compute_merit_value(Real mu_f,
                               Real mu_c,
                               Real cost,
                               Real defect_norm,
                               Real constr_viol);

    Real compute_merit_slope(Real cost_slope,
                               Real mu_f,
                               Real mu_c,
                               Real defect_norm,
                               Real constr_viol);

    Real compute_cost_slope();

    std::pair<Real, Real> compute_merit_weights(Real cost_der, Real defect_norm, Real constr_viol);

    Real compute_cost(const MatrixXr& xtrj,
                        const MatrixXr& utrj);

    Real compute_bound_penalty(const MatrixXr& xtrj,
                                 const MatrixXr& utrj);

    Real compute_constr(const MatrixXr& xtrj,
                          const MatrixXr& utrj);

    Real compute_defect(const MatrixXr& xtrj,
                          const MatrixXr& utrj);

    bool forward_pass(Real alpha);

    bool line_search(int iter);

    void reset_iterate_filter();

    bool should_stop();

    void set_default_cost();

    bool fixed_initial_state();

    enum DecompositionType
    {
        Ldlt, Qr, Lu, Cod, Svd, ReducedHessian
    };

    static DecompositionType str_to_decomp_type(const std::string& dt_str);

    bool _verbose;
    bool _debug;
    bool _log_iterations;
    bool _log;
    bool _rti;
    bool _codegen_verbose;
    
    const int _nx;
    const int _nu;
    const int _N;

    Real _step_length;
    Real _rho_base;
    Real _rho;
    Real _rho_growth_factor;
    Real _hxx_reg;
    Real _hxx_reg_base;
    Real _hxx_reg_growth_factor;
    Real _huu_reg;
    Real _kkt_reg;
    Real _line_search_accept_ratio;
    Real _alpha_min;
    Real _svd_threshold;
    Real _constraint_violation_threshold;
    Real _defect_norm_threshold;
    Real _merit_der_threshold;
    Real _step_length_threshold;
    bool _enable_line_search;
    bool _enable_auglag;
    bool _use_kkt_solver;

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
    MatrixXr _x_lb, _x_ub;
    MatrixXr _u_lb, _u_ub;
    std::vector<ValueFunction> _value;
    std::vector<Dynamics> _dyn;

    std::vector<BackwardPassResult> _bp_res;
    std::unique_ptr<ConstraintToGo> _constraint_to_go;
    std::unique_ptr<ForwardPassResult> _fp_res;
    int _fp_accepted;

    std::vector<Eigen::Triplet<Real>> _kkt_triplets;
    Eigen::SparseMatrix<Real> _kkt_mat;
    VectorXr _kkt_rhs;
    Eigen::SparseLU<Eigen::SparseMatrix<Real>, Eigen::COLAMDOrdering<int>> _kkt_lu_solver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Real>, Eigen::Lower, Eigen::COLAMDOrdering<int>> _kkt_ldlt_solver;

    IterateFilter _it_filt;
    bool _use_it_filter;

    MatrixXr _xtrj;
    MatrixXr _utrj;
    std::vector<VectorXr> _lam_g;
    MatrixXr _lam_x;

    MatrixXr _lam_bound_x;
    MatrixXr _lam_bound_u;

    Eigen::MatrixXd _dx, _du;

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

    std::vector<ForwardPassResult> _fp_res_history;

    std::map<std::string, MatrixXr> _constr_values;
    std::map<std::string, VectorXr> _cost_values;
};



}

#endif
