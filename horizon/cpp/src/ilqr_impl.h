#ifndef ILQR_IMPL_H
#define ILQR_IMPL_H

#include "ilqr.h"
#include "wrapped_function.h"
#include "typedefs.h"

using namespace horizon;
using namespace casadi_utils;

using Real=horizon::Real;
using MatrixXr=horizon::MatrixXr;
using VectorXr=horizon::VectorXr;

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
    const MatrixXr& A() const;

    // df/du
    const MatrixXr& B() const;

    // defect (or gap)
    VectorXr d;

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
                       VectorXr& d);

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
    const MatrixXr& C() const;

    // dh/du
    const MatrixXr& D() const;

    // constraint violation h(x, u) - hdes
    VecConstRef h() const;

    // valid flag
    bool is_valid() const;

    ConstraintEntity();

    void linearize(VecConstRef x, VecConstRef u, int k);

    void evaluate(VecConstRef x, VecConstRef u, int k);

    void setConstraint(casadi::Function h);

    void setConstraint(casadi::Function h, casadi::Function dh);

    void setTargetValue(const VectorXr& hdes);

    static casadi::Function Jacobian(const casadi::Function& h);

private:

    // desired value
    VectorXr _hdes;

    // computed value
    VectorXr _hvalue;

};

struct IterativeLQR::Constraint
{

    // dh/dx
    const MatrixXr& C() const;

    // dh/du
    const MatrixXr& D() const;

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

    std::vector<ConstraintEntity::Ptr> items;

private:

    MatrixXr _C;
    MatrixXr _D;
    VectorXr _h;

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

    virtual Real evaluate(VecConstRef x,
                            VecConstRef u,
                            int k) = 0;

    virtual void quadratize(VecConstRef x,
                            VecConstRef u,
                            int k,
                            MatrixXr& Q,
                            MatrixXr& R,
                            MatrixXr& P) = 0;

    virtual std::string getName() = 0;

    virtual Real getCostEvaluated() const { return _cost_eval; }

    virtual ~CostEntityBase() = default;



protected:

    VectorXr _q, _r;
    Real _cost_eval;
};

struct IterativeLQR::BoundAuglagCostEntity : CostEntityBase
{
    typedef std::shared_ptr<BoundAuglagCostEntity> Ptr;

    BoundAuglagCostEntity(int N,
                          VecConstRef xlb, VecConstRef xub,
                          VecConstRef ulb, VecConstRef uub);

    void setRho(Real rho);

    Real evaluate(VecConstRef x, VecConstRef u, int k) override;

    void quadratize(VecConstRef x,
                    VecConstRef u,
                    int k,
                    MatrixXr& Q,
                    MatrixXr& R,
                    MatrixXr& P) override;

    std::string getName();

    void update_lam(VecConstRef x, VecConstRef u, int k);

    VecConstRef getStateMultiplier() const;

    VecConstRef getInputMultiplier() const;

private:

    VecConstRef _xlb, _xub;
    VecConstRef _ulb, _uub;

    VectorXr _x_violation;
    VectorXr _u_violation;

    VectorXr _xlam, _ulam;
    Real _rho;

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

    Real evaluate(VecConstRef x, VecConstRef u, int k) override;

    void quadratize(VecConstRef x,
                    VecConstRef u,
                    int k,
                    MatrixXr& Q,
                    MatrixXr& R,
                    MatrixXr& P) override;

    std::string getName();

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

    Real evaluate(VecConstRef x, VecConstRef u, int k) override;

    void quadratize(VecConstRef x,
                    VecConstRef u,
                    int k,
                    MatrixXr& Q,
                    MatrixXr& R,
                    MatrixXr& P) override;

    std::string getName();

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
    const MatrixXr& Q() const;
    VecConstRef q() const;
    const MatrixXr& R() const;
    VecConstRef r() const;
    const MatrixXr& P() const;

    IntermediateCost(int nx, int nu);

    void addCost(CostEntityBase::Ptr cost);

    Real evaluate(VecConstRef x, VecConstRef u, int k);
    void quadratize(VecConstRef x, VecConstRef u, int k);

    void clear();

    std::vector<CostEntityBase::Ptr> items;

private:

    MatrixXr _Q, _R, _P;
    VectorXr _q, _r;
};

struct IterativeLQR::Temporaries
{
    /* Backward pass */

    // temporary for s + S*d
    MatrixXr s_plus_S_d;

    // temporary for S*A
    MatrixXr S_A;

    // feasible constraint
    MatrixXr Cf, Df;
    VectorXr hf;

    // cod of constraint
    Eigen::CompleteOrthogonalDecomposition<MatrixXr> ccod;
    Eigen::ColPivHouseholderQR<MatrixXr> cqr;
    Eigen::BDCSVD<MatrixXr> csvd;
    MatrixXr codQ;
    Eigen::PermutationMatrix<Eigen::Dynamic> codP;

    // quadratized value function
    MatrixXr Huu;
    MatrixXr Hux;
    MatrixXr Hxx;
    VectorXr hx;
    VectorXr hu;

    // temporary for kkt rhs
    MatrixXr kkt;
    MatrixXr kx0;

    // lu for kkt matrix
    Eigen::PartialPivLU<MatrixXr> lu;
    Eigen::ColPivHouseholderQR<MatrixXr> qr;
    Eigen::LDLT<MatrixXr> ldlt;

    // kkt solution
    MatrixXr u_lam;

    // infeasible component of constraint
    MatrixXr Cinf;
    MatrixXr Dinf;
    VectorXr hinf;

    // optimal state computation
    // (note: only for initial state x[0])
    Eigen::FullPivLU<MatrixXr> x_lu;
    Eigen::ColPivHouseholderQR<MatrixXr> x_qr;
    Eigen::LDLT<MatrixXr> x_ldlt;
    MatrixXr x_kkt;
    VectorXr x_k0;
    VectorXr dx_lam;

    /* Forward pass */
    VectorXr dx;
    VectorXr du;
    VectorXr defect;

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

    Eigen::Matrix<Real, -1, -1, Eigen::RowMajor> _C;
    Eigen::Matrix<Real, -1, -1, Eigen::RowMajor> _D;
    VectorXr _h;
    int _dim;
};

struct IterativeLQR::ValueFunction
{
    MatrixXr S;
    VectorXr s;

    ValueFunction(int nx);
};

struct IterativeLQR::BackwardPassResult
{
    // real input as function of state
    // (u = Lu*x + lu)
    MatrixXr Lu;
    VectorXr lu;

    // auxiliary input as function of state
    // (z = Lz*x + lz, where u = lc + Lc*x + Bz*z)
    MatrixXr Lz;
    VectorXr lz;

    // constraint-to-go size
    int nc;

    // lagrange multipliers
    MatrixXr Gu;
    MatrixXr Gx;
    VectorXr glam;

    // optimal state
    // (this is only filled at i = 0)
    VectorXr dx;
    VectorXr dx_lam;

    BackwardPassResult(int nx, int nu);
};

struct IterativeLQR::FeasibleConstraint
{
    MatConstRef C;
    MatConstRef D;
    VecConstRef h;
};

static void set_param_inputs(std::shared_ptr<std::map<std::string, MatrixXr>> params, int k,
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
