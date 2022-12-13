#ifndef ILQR_IMPL_H
#define ILQR_IMPL_H

#include "ilqr.h"
#include "wrapped_function.h"

using namespace horizon;
using namespace casadi_utils;

namespace cs = casadi;

extern utils::Timer::TocCallback on_timer_toc;

struct IterativeLQR::Dynamics
{

public:

    // dynamics function
    casadi_utils::WrappedFunction f;

    // dynamics jacobian
    casadi_utils::WrappedFunction df;

    // parameters
    ParameterMapPtr param;

    // df/dx
    const Eigen::MatrixXd& A() const;

    // df/du
    const Eigen::MatrixXd& B() const;

    // defect (or gap)
    Eigen::VectorXd d;

    Dynamics(int nx, int nu);

    VecConstRef integrate(VecConstRef x,
                          VecConstRef u,
                          int k);

    void linearize(VecConstRef x,
                   VecConstRef u,
                   int k);

    void computeDefect(VecConstRef x,
                       VecConstRef u,
                       VecConstRef xnext,
                       int k,
                       Eigen::VectorXd& d);

    void setDynamics(casadi::Function f);

    static casadi::Function Jacobian(const casadi::Function& f);

};

struct IterativeLQR::ConstraintEntity
{
    typedef std::shared_ptr<ConstraintEntity> Ptr;

    // constraint function
    casadi_utils::WrappedFunction f;

    // constraint jacobian
    casadi_utils::WrappedFunction df;

    // parameter map
    ParameterMapPtr param;

    // indices
    std::vector<int> indices;

    // dh/dx
    const Eigen::MatrixXd& C() const;

    // dh/du
    const Eigen::MatrixXd& D() const;

    // constraint violation h(x, u) - hdes
    VecConstRef h() const;

    // valid flag
    bool is_valid() const;

    ConstraintEntity();

    void linearize(VecConstRef x, VecConstRef u, int k);

    void evaluate(VecConstRef x, VecConstRef u, int k);

    void setConstraint(casadi::Function h);

    void setConstraint(casadi::Function h, casadi::Function dh);

    void setTargetValue(const Eigen::VectorXd& hdes);

    static casadi::Function Jacobian(const casadi::Function& h);

private:

    // desired value
    Eigen::VectorXd _hdes;

    // computed value
    Eigen::VectorXd _hvalue;

};

struct IterativeLQR::Constraint
{
    // dh/dx
    const Eigen::MatrixXd& C() const;

    // dh/du
    const Eigen::MatrixXd& D() const;

    // constraint violation f(x, u)
    VecConstRef h() const;

    // size getter
    int size() const;

    // valid flag
    bool is_valid() const;

    Constraint(int nx, int nu);

    void linearize(VecConstRef x, VecConstRef u, int k);

    void evaluate(VecConstRef x, VecConstRef u, int k);

    void addConstraint(ConstraintEntity::Ptr h);

    void clear();

private:

    std::vector<ConstraintEntity::Ptr> items;
    Eigen::MatrixXd _C;
    Eigen::MatrixXd _D;
    Eigen::VectorXd _h;

};

struct IterativeLQR::CostEntityBase
{
    typedef std::shared_ptr<CostEntityBase> Ptr;

    // parameters
    ParameterMapPtr param;

    // indices
    std::vector<int> indices;

    /* Quadratized cost */
    virtual VecConstRef q() const { return _q; }

    virtual VecConstRef r() const { return _r; }

    virtual double evaluate(VecConstRef x,
                            VecConstRef u,
                            int k) = 0;

    virtual void quadratize(VecConstRef x,
                            VecConstRef u,
                            int k,
                            Eigen::MatrixXd& Q,
                            Eigen::MatrixXd& R,
                            Eigen::MatrixXd& P) = 0;

    virtual ~CostEntityBase() = default;

protected:

    Eigen::VectorXd _q, _r;
};

struct IterativeLQR::BoundAuglagCostEntity : CostEntityBase
{
    typedef std::shared_ptr<BoundAuglagCostEntity> Ptr;

    BoundAuglagCostEntity(int N,
                          VecConstRef xlb, VecConstRef xub,
                          VecConstRef ulb, VecConstRef uub);

    void setRho(double rho);

    double evaluate(VecConstRef x, VecConstRef u, int k) override;

    void quadratize(VecConstRef x,
                    VecConstRef u,
                    int k,
                    Eigen::MatrixXd& Q,
                    Eigen::MatrixXd& R,
                    Eigen::MatrixXd& P) override;

    void update_lam(VecConstRef x, VecConstRef u, int k);

    VecConstRef getStateMultiplier() const;

    VecConstRef getInputMultiplier() const;

private:

    VecConstRef _xlb, _xub;
    VecConstRef _ulb, _uub;

    Eigen::VectorXd _x_violation;
    Eigen::VectorXd _u_violation;

    Eigen::VectorXd _xlam, _ulam;
    double _rho;

    const int _N;
};


struct IterativeLQR::IntermediateCostEntity : CostEntityBase
{
    typedef std::shared_ptr<IntermediateCostEntity> Ptr;

    // set cost
    void setCost(casadi::Function l,
                 casadi::Function dl,
                 casadi::Function ddl);

    VecConstRef q() const override;
    VecConstRef r() const override;

    double evaluate(VecConstRef x, VecConstRef u, int k) override;

    void quadratize(VecConstRef x,
                    VecConstRef u,
                    int k,
                    Eigen::MatrixXd& Q,
                    Eigen::MatrixXd& R,
                    Eigen::MatrixXd& P) override;

    static casadi::Function Gradient(const casadi::Function& f);
    static casadi::Function Hessian(const casadi::Function& df);

private:

    // original cost
    casadi_utils::WrappedFunction l;

    // cost gradient
    casadi_utils::WrappedFunction dl;

    // cost hessian
    casadi_utils::WrappedFunction ddl;
};

struct IterativeLQR::IntermediateResidualEntity : CostEntityBase
{
    typedef std::shared_ptr<IntermediateResidualEntity> Ptr;

    void setResidual(casadi::Function res,
                     casadi::Function dres);

    double evaluate(VecConstRef x, VecConstRef u, int k) override;

    void quadratize(VecConstRef x,
                    VecConstRef u,
                    int k,
                    Eigen::MatrixXd& Q,
                    Eigen::MatrixXd& R,
                    Eigen::MatrixXd& P) override;

    static casadi::Function Jacobian(const casadi::Function& f);

private:

    // original residual
    casadi_utils::WrappedFunction res;

    // residual jacobian
    casadi_utils::WrappedFunction dres;


};

struct IterativeLQR::IntermediateCost
{

    /* Quadratized cost */
    const Eigen::MatrixXd& Q() const;
    VecConstRef q() const;
    const Eigen::MatrixXd& R() const;
    VecConstRef r() const;
    const Eigen::MatrixXd& P() const;

    IntermediateCost(int nx, int nu);

    void addCost(CostEntityBase::Ptr cost);

    double evaluate(VecConstRef x, VecConstRef u, int k);
    void quadratize(VecConstRef x, VecConstRef u, int k);

    void clear();

private:

    std::vector<CostEntityBase::Ptr> items;
    Eigen::MatrixXd _Q, _R, _P;
    Eigen::VectorXd _q, _r;
};

struct IterativeLQR::Temporaries
{
    /* Backward pass */

    // temporary for s + S*d
    Eigen::MatrixXd s_plus_S_d;

    // temporary for S*A
    Eigen::MatrixXd S_A;

    // feasible constraint
    Eigen::MatrixXd Cf, Df;
    Eigen::VectorXd hf;

    // cod of constraint
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> ccod;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> cqr;
    Eigen::BDCSVD<Eigen::MatrixXd> csvd;
    Eigen::MatrixXd codQ;
    Eigen::PermutationMatrix<Eigen::Dynamic> codP;

    // quadratized value function
    Eigen::MatrixXd Huu;
    Eigen::MatrixXd Hux;
    Eigen::MatrixXd Hxx;
    Eigen::VectorXd hx;
    Eigen::VectorXd hu;

    // temporary for kkt rhs
    Eigen::MatrixXd kkt;
    Eigen::MatrixXd kx0;

    // lu for kkt matrix
    Eigen::PartialPivLU<Eigen::MatrixXd> lu;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;
    Eigen::LDLT<Eigen::MatrixXd> ldlt;

    // kkt solution
    Eigen::MatrixXd u_lam;

    // infeasible component of constraint
    Eigen::MatrixXd Cinf;
    Eigen::MatrixXd Dinf;
    Eigen::VectorXd hinf;

    // optimal state computation
    // (note: only for initial state x[0])
    Eigen::FullPivLU<Eigen::MatrixXd> x_lu;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> x_qr;
    Eigen::LDLT<Eigen::MatrixXd> x_ldlt;
    Eigen::MatrixXd x_kkt;
    Eigen::VectorXd x_k0;
    Eigen::VectorXd dx_lam;

    /* Forward pass */
    Eigen::VectorXd dx;
    Eigen::VectorXd du;
    Eigen::VectorXd defect;

};

struct IterativeLQR::ConstraintToGo
{
    ConstraintToGo(int nx, int nu);

    void set(MatConstRef C, VecConstRef h);

    void set(const Constraint& constr);

    void propagate_backwards(MatConstRef A, MatConstRef B, VecConstRef d);

    void add(const Constraint& constr);

    void add(MatConstRef C, MatConstRef D, VecConstRef h);

    void add(MatConstRef C, VecConstRef h);

    void clear();

    int dim() const;

    MatConstRef C() const;

    MatConstRef D() const;

    VecConstRef h() const;


private:

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> _C;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> _D;
    Eigen::VectorXd _h;
    int _dim;
};

struct IterativeLQR::ValueFunction
{
    Eigen::MatrixXd S;
    Eigen::VectorXd s;

    ValueFunction(int nx);
};

struct IterativeLQR::BackwardPassResult
{
    // real input as function of state
    // (u = Lu*x + lu)
    Eigen::MatrixXd Lu;
    Eigen::VectorXd lu;

    // auxiliary input as function of state
    // (z = Lz*x + lz, where u = lc + Lc*x + Bz*z)
    Eigen::MatrixXd Lz;
    Eigen::VectorXd lz;

    // constraint-to-go size
    int nc;

    // lagrange multipliers
    Eigen::MatrixXd Gu;
    Eigen::MatrixXd Gx;
    Eigen::VectorXd glam;

    // optimal state
    // (this is only filled at i = 0)
    Eigen::VectorXd dx;
    Eigen::VectorXd dx_lam;

    BackwardPassResult(int nx, int nu);
};

struct IterativeLQR::FeasibleConstraint
{
    MatConstRef C;
    MatConstRef D;
    VecConstRef h;
};

static void set_param_inputs(std::shared_ptr<std::map<std::string, Eigen::MatrixXd>> params, int k,
                             casadi_utils::WrappedFunction& f);

#define THROW_NAN(mat) \
    if((mat).hasNaN()) \
    { \
        throw std::runtime_error("[" + std::string(__func__) + "] NaN value detected in " #mat); \
    } \
    if(!mat.allFinite()) \
    { \
        throw std::runtime_error("[" + std::string(__func__) + "] Inf value detected in " #mat); \
    }


#endif // ILQR_IMPL_H
