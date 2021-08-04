#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <memory>

#include "wrapped_function.h"

namespace horizon
{

typedef Eigen::Ref<const Eigen::VectorXd> VecConstRef;
typedef Eigen::Ref<const Eigen::MatrixXd> MatConstRef;

typedef std::function<bool(const Eigen::MatrixXd& xtrj,
                           const Eigen::MatrixXd& utrj,
                           double step_length,
                           double total_cost,
                           double defect_norm,
                           double constraint_violation)> CallbackType;


class IterativeLQR
{

public:

    IterativeLQR(casadi::Function fdyn,
                 int N);

    void setIntermediateCost(const std::vector<casadi::Function>& inter_cost);

    void setIntermediateCost(int k, const casadi::Function& inter_cost);

    void setFinalCost(const casadi::Function& final_cost);

    void setIntermediateConstraint(int k, const casadi::Function& inter_constraint);

    void setIntermediateConstraint(const std::vector<casadi::Function>& inter_constraint);

    void setFinalConstraint(const casadi::Function& final_constraint);

    void setInitialState(const Eigen::VectorXd& x0);

    void setIterationCallback(const CallbackType& cb);

    bool solve(int max_iter);

    const Eigen::MatrixXd& getStateTrajectory() const;

    const Eigen::MatrixXd& getInputTrajectory() const;

    VecConstRef state(int i) const;

    VecConstRef input(int i) const;

    ~IterativeLQR();

protected:

private:

    struct ConstrainedDynamics;
    struct ConstrainedCost;
    struct Dynamics;
    struct Constraint;
    struct IntermediateCost;
    struct Temporaries;
    struct ConstraintToGo;
    struct BackwardPassResult;
    struct ForwardPassResult;
    struct ValueFunction;

    typedef std::tuple<int, ConstrainedDynamics, ConstrainedCost> HandleConstraintsRetType;

    void linearize_quadratize();
    void report_result();
    void backward_pass();
    void backward_pass_iter(int i);
    HandleConstraintsRetType handle_constraints(int i);
    bool forward_pass(double alpha);
    void forward_pass_iter(int i, double alpha);
    void set_default_cost();

    int _nx;
    int _nu;
    int _N;

    std::vector<IntermediateCost> _cost;
    std::vector<Constraint> _constraint;
    std::vector<ValueFunction> _value;
    std::vector<Dynamics> _dyn;

    std::vector<BackwardPassResult> _bp_res;
    std::unique_ptr<ConstraintToGo> _constraint_to_go;
    std::unique_ptr<ForwardPassResult> _fp_res;

    Eigen::MatrixXd _xtrj;
    Eigen::MatrixXd _utrj;

    std::vector<Temporaries> _tmp;

    CallbackType _iter_cb;
};



}
