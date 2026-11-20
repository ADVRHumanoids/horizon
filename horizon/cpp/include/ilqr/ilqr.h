#ifndef __HORIZON__ILQR__H__
#define __HORIZON__ILQR__H__

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <atomic>
#include <casadi/casadi.hpp>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>

#include "function.h"
#include "horizon/profiling.h"
#include "iterate_filter.h"
#include "state_space.h"

namespace horizon
{

typedef Eigen::Ref<const Eigen::VectorXd> VecConstRef;
typedef Eigen::Ref<const Eigen::MatrixXd> MatConstRef;

/** Horizon-relative node selection, resolved when a term is installed. */
class Nodes
{
    enum class Kind
    {
        Running,
        Terminal,
        All
    };

  public:
    static Nodes running()
    {
        return Nodes(Kind::Running);
    }
    static Nodes terminal()
    {
        return Nodes(Kind::Terminal);
    }
    static Nodes all()
    {
        return Nodes(Kind::All);
    }
    std::vector<int> resolve(int N) const
    {
        if (_kind == Kind::Terminal)
        {
            return {N};
        }
        std::vector<int> indices(N + (_kind == Kind::All));
        for (int i = 0; i < static_cast<int>(indices.size()); ++i)
        {
            indices[i] = i;
        }
        return indices;
    }

  private:
    explicit Nodes(Kind kind) : _kind(kind)
    {
    }
    Kind _kind;
};

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
    const ForwardPassResult &getIterationResult() const;

    /**
     * @brief CallbackType
     */
    typedef std::function<bool(const ForwardPassResult &res)> CallbackType;

    typedef std::variant<int, double, bool, std::string> OptionTypes;
    typedef std::map<std::string, OptionTypes> OptionDict;

    /**
     * @brief Class constructor
     * @param fdyn is a function mapping state and control to the integrated
     * state; required signature is (x, u, p) -> (f)
     * @param N is the number of shooting intervals
     */
    IterativeLQR(casadi::Function fdyn, int N, OptionDict opt = OptionDict(),
                 std::shared_ptr<const ilqr::StateSpace> space = nullptr);

    IterativeLQR(const ilqr::FunctionManager &manager,
                 const std::string &dynamics, int N,
                 OptionDict opt = OptionDict(),
                 std::shared_ptr<const ilqr::StateSpace> space = nullptr);

    void setCost(std::vector<int> indices, const ilqr::FunctionManager &manager,
                 const std::string &name);
    void setResidual(std::vector<int> indices,
                     const ilqr::FunctionManager &manager,
                     const std::string &name);
    void setConstraint(std::vector<int> indices,
                       const ilqr::FunctionManager &manager,
                       const std::string &name,
                       std::vector<Eigen::VectorXd> targets = {});

    void setCost(Nodes nodes, const ilqr::FunctionManager &manager,
                 const std::string &name)
    {
        setCost(nodes.resolve(_N), manager, name);
    }
    void setCost(Nodes nodes, const casadi::Function &function)
    {
        setCost(nodes.resolve(_N), function);
    }
    void setResidual(Nodes nodes, const ilqr::FunctionManager &manager,
                     const std::string &name)
    {
        setResidual(nodes.resolve(_N), manager, name);
    }
    void setResidual(Nodes nodes, const casadi::Function &function)
    {
        setResidual(nodes.resolve(_N), function);
    }
    void setConstraint(Nodes nodes, const ilqr::FunctionManager &manager,
                       const std::string &name)
    {
        setConstraint(nodes.resolve(_N), manager, name);
    }
    void setConstraint(Nodes nodes, const casadi::Function &function)
    {
        setConstraint(nodes.resolve(_N), function);
    }
    void setConstraint(Nodes nodes, const ilqr::FunctionManager &manager,
                       const std::string &name,
                       std::vector<Eigen::VectorXd> targets)
    {
        setConstraint(nodes.resolve(_N), manager, name, std::move(targets));
    }
    void setConstraint(Nodes nodes, const casadi::Function &function,
                       std::vector<Eigen::VectorXd> targets)
    {
        setConstraint(nodes.resolve(_N), function, std::move(targets));
    }
    void setBatchedCost(Nodes nodes, const std::string &name,
                        const ilqr::BatchedCost &batch, int tangent_offset = 0)
    {
        setBatchedCost(nodes.resolve(_N), name, batch, tangent_offset);
    }

    /** Fix node zero through equal bounds and initialize its guess. */
    void fixInitialState(const Eigen::VectorXd &x0);
    void setInputBounds(double lb, double ub)
    {
        setInputBounds(Eigen::MatrixXd::Constant(_nu, _N, lb),
                       Eigen::MatrixXd::Constant(_nu, _N, ub));
    }

    void setStateBounds(const Eigen::MatrixXd &lb, const Eigen::MatrixXd &ub);

    void setInputBounds(const Eigen::MatrixXd &lb, const Eigen::MatrixXd &ub);

    /**
     * @brief set an intermediate cost term for the k-th intermediate state,
     * as specificed by a vector of indices
     * @param indices: the nodes that the cost refers to
     * @param inter_cost: a function with required signature (x, u, p) -> (l)
     */
    void setCost(std::vector<int> indices, const casadi::Function &inter_cost);

    void setResidual(std::vector<int> indices,
                     const casadi::Function &inter_cost);

    /**
     * @brief set the final cost
     * @param final_cost: a function with required signature (x, u) -> (l),
     * even though the input 'u' is not used
     */
    void setFinalCost(const casadi::Function &final_cost);

    /**
     * @brief  set an intermediate constraint term for the k-th intermediate
     * state, as specificed by a vector of indices
     * @param indices: the nodes that the cost refers to
     * @param inter_constraint: a function with required signature (x, u, p) ->
     * (h), where the constraint is h(x, u) = 0
     * @param target_values: if specified, the i-th entry is used as target
     * value for the constraint function at the indices[i]
     */
    void setConstraint(std::vector<int> indices,
                       const casadi::Function &inter_constraint,
                       std::vector<Eigen::VectorXd> target_values =
                           std::vector<Eigen::VectorXd>());

    void setFinalConstraint(const casadi::Function &final_constraint);

    void setIndices(const std::string &f_name, const std::vector<int> &indices);

    void setIndices(const std::string &f_name, Nodes nodes)
    {
        setIndices(f_name, nodes.resolve(_N));
    }

    void updateIndices();

    void printState() const;

    void setParameterValue(const std::string &pname,
                           const Eigen::MatrixXd &value,
                           const std::vector<int> &indices = {});

    /** Set only the node-zero initial guess; use fixInitialState for bounds. */
    void setInitialState(const Eigen::VectorXd &x0);

    void setStateInitialGuess(const Eigen::MatrixXd &x0);

    void setInputInitialGuess(const Eigen::MatrixXd &u0);

    void setIterationCallback(const CallbackType &cb);

    /** Register a cloned horizon-wide evaluator. Setup operation; the batch
     * must cover the nodes on which its associated functions are installed. */
    void addEvaluationBatch(const ilqr::EvaluationBatch &batch);
    /** State-only Gauss-Newton contribution in a contiguous tangent block.
     * The prototype is cloned; node cost objects share that solver-owned clone.
     */
    void setBatchedCost(std::vector<int> indices, const std::string &name,
                        const ilqr::BatchedCost &batch, int tangent_offset = 0);

    bool solve(int max_iter);

    const Eigen::MatrixXd &getStateTrajectory() const;

    const Eigen::MatrixXd &getInputTrajectory() const;

    Eigen::MatrixXd
    getParameterValue(const std::string &pname,
                      const std::vector<int> &indices = {}) const;

    const utils::ProfilingInfo &getProfilingInfo() const;

    const std::vector<ForwardPassResult> &getIterationHistory() const;

    const std::map<std::string, Eigen::MatrixXd> &getConstraintsValues() const;

    const std::map<std::string, Eigen::VectorXd> &getCostsValues() const;

    VecConstRef state(int i) const;

    VecConstRef input(int i) const;

    MatConstRef gain(int i) const;

    ~IterativeLQR();

    struct ForwardPassResult
    {
        Eigen::MatrixXd xtrj;
        Eigen::MatrixXd utrj;
        double damping = 0;
        double predicted_reduction = 0, actual_reduction = 0,
               model_agreement = 0;
        double stationarity = std::numeric_limits<double>::infinity();
        int trial_count = 0, direction_solves = 0;
        int derivative_evaluations = 0, value_evaluations = 0;
        bool used_correction = false, used_restoration = false;
        std::string acceptance_reason;
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

        void print(int N = 1) const;
    };

  protected:
  private:
    friend struct IterativeLQRTestAccess;

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
    struct ManagedCostEntity;
    struct FunctionWorkspace;
    struct Temporaries;
    struct ConstraintToGo;
    struct BackwardPassResult;
    struct ValueFunction;

    typedef std::tuple<int, ConstrainedDynamics, ConstrainedCost>
        HandleConstraintsRetType;

    typedef std::shared_ptr<std::map<std::string, Eigen::MatrixXd>>
        ParameterMapPtr;

    typedef std::map<std::string, std::shared_ptr<CostEntityBase>> CostPtrMap;

    typedef std::map<std::string, std::shared_ptr<ConstraintEntity>>
        ConstraintPtrMap;

    void init_thread_pool(int pool_size);

    void add_param_to_map(const casadi::Function &f);
    void add_param_to_map(const ilqr::FunctionSpec &f);
    void setManagedCost(std::vector<int> indices,
                        const ilqr::FunctionSpec &spec, const std::string &name,
                        bool residual);
    void validateIndices(const std::vector<int> &indices) const;
    void stateEquality(int k, Eigen::MatrixXd &C, Eigen::VectorXd &h) const;

    void linearize_quadratize();
    void prepareEvaluationBatches(const Eigen::MatrixXd &states,
                                  const Eigen::MatrixXd &controls,
                                  bool derivatives);
    std::vector<std::unique_ptr<ilqr::EvaluationBatch>> _evaluation_batches;
    struct BatchedCostRegistration
    {
        std::shared_ptr<ilqr::BatchedCost> batch;
        std::vector<int> indices;
        Eigen::MatrixXd selected_states;
        bool full_horizon = false;
    };
    std::vector<BatchedCostRegistration> _batched_costs;

    void linearize_quadratize_inner(int i);

    void linearize_quadratize_mt();

    void linearize_quadratize_thread_main(int istart, int iend);

    void report_result(const ForwardPassResult &fpres);

    void backward_pass();

    void backward_pass_iter(int i);

    void kkt_solve();

    void optimize_initial_state();

    void increase_regularization();

    void reduce_regularization();

    FeasibleConstraint handle_constraints(int i);

    void add_bound_constraint(int k);

    bool auglag_update();
    bool auglag_inner_converged() const;
    bool auglag_dual_converged() const;

    double compute_merit_value(double mu_f, double mu_c, double cost,
                               double defect_norm, double constr_viol);

    double compute_merit_slope(double cost_slope, double mu_f, double mu_c,
                               double defect_norm, double constr_viol);

    double compute_cost_slope();

    std::pair<double, double> compute_merit_weights(double cost_der,
                                                    double defect_norm,
                                                    double constr_viol);

    double compute_cost(const Eigen::MatrixXd &xtrj,
                        const Eigen::MatrixXd &utrj);

    double compute_bound_penalty(const Eigen::MatrixXd &xtrj,
                                 const Eigen::MatrixXd &utrj);

    double compute_constr(const Eigen::MatrixXd &xtrj,
                          const Eigen::MatrixXd &utrj);

    double compute_defect(const Eigen::MatrixXd &xtrj,
                          const Eigen::MatrixXd &utrj);

    bool forward_pass(double alpha);

    bool line_search(int iter);
    struct GlobalizationModel;
    void prepareGlobalization();
    bool globalizationStep();
    bool restoreFeasibility();
    Eigen::VectorXd globalizationResidual(const Eigen::MatrixXd &,
                                          const Eigen::MatrixXd &);
    Eigen::VectorXd feasibilityStep(const GlobalizationModel &,
                                    const Eigen::VectorXd &, bool correction);
    void evaluateGlobalizationTrial(const Eigen::VectorXd &step);
    void restoreGlobalizationBaseline();
    void increaseGlobalizationDamping();

    void reset_iterate_filter();

    bool should_stop();

    void set_default_cost();

    bool fixed_initial_state();

    enum DecompositionType
    {
        Ldlt,
        Qr,
        Lu,
        Cod,
        Svd,
        ReducedHessian
    };

    static DecompositionType str_to_decomp_type(const std::string &dt_str);

    bool _verbose;
    bool _log;
    bool _rti;

    const std::shared_ptr<const ilqr::StateSpace> _space;
    const int _ns; // state storage dimension
    const int _nx; // tangent dimension
    const int _nu;
    const int _N;

    std::string _globalization = "legacy";
    double _damping = 1e-3, _stationarity_threshold = 1e-6;
    double _filter_violation_limit = 0;
    double _global_mu = 0;
    double _global_inner_stationarity = 0;
    std::vector<std::pair<double, double>> _global_filter;
    std::unique_ptr<GlobalizationModel> _global_model;
    Eigen::MatrixXd _global_xdiag, _global_udiag;
    double _step_length;
    double _bound_multiplier_max = 0;
    double _rho_base;
    double _rho;
    double _rho_growth_factor;
    double _rho_max;
    double _auglag_progress_ratio;
    double _auglag_inner_tolerance;
    double _auglag_previous_violation;
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
    Eigen::MatrixXd _x_lb, _x_ub;
    Eigen::MatrixXd _u_lb, _u_ub;
    std::vector<ValueFunction> _value;
    std::vector<Dynamics> _dyn;
    std::vector<std::shared_ptr<ilqr::EvaluationContext>> _evaluation_contexts;

    std::vector<BackwardPassResult> _bp_res;
    std::unique_ptr<ConstraintToGo> _constraint_to_go;
    std::unique_ptr<ForwardPassResult> _fp_res;
    int _fp_accepted;

    std::vector<Eigen::Triplet<double>> _kkt_triplets;
    Eigen::SparseMatrix<double> _kkt_mat;
    Eigen::VectorXd _kkt_rhs;
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        _kkt_lu_solver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                          Eigen::COLAMDOrdering<int>>
        _kkt_ldlt_solver;

    IterateFilter _it_filt;
    bool _use_it_filter;

    Eigen::MatrixXd _xtrj;
    Eigen::MatrixXd _utrj;
    // Multipliers in original constraint row order, including terminal rows.
    std::vector<Eigen::VectorXd> _lam_g;
    // Column k multiplies A_k dx_k + B_k du_k + d_k - dx_{k+1}.
    Eigen::MatrixXd _lam_x;

    Eigen::MatrixXd _lam_bound_x;
    Eigen::MatrixXd _lam_bound_u;

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
    std::exception_ptr _th_exception;
    std::mutex _prof_mutex;

    CallbackType _iter_cb;
    utils::ProfilingInfo _prof_info;

    std::vector<ForwardPassResult> _fp_res_history;

    std::map<std::string, Eigen::MatrixXd> _constr_values;
    std::map<std::string, Eigen::VectorXd> _cost_values;
};

} // namespace horizon

#endif
