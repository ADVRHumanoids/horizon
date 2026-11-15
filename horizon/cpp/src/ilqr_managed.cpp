#include "ilqr/ilqr_impl.h"
#include <set>

namespace
{
// Only the signature is used to share legacy solver initialization. Numerical
// calls are replaced by cloned native evaluators before construction returns.
casadi::Function dynamicsSignature(const ilqr::FunctionSpec& spec)
{
    if(!spec.function) throw std::invalid_argument("null dynamics function");
    const int ns = spec.function->outputSize();
    int nu = spec.function->inputSize() - ns;
    for(const auto& p : spec.parameters) nu -= p.size;
    if(nu <= 0) throw std::invalid_argument("dynamics must have state and control inputs");
    auto x = cs::SX::sym("x", ns);
    auto u = cs::SX::sym("u", nu);
    return cs::Function("managed_dynamics", {x, u}, {x}, {"x", "u"}, {"f"});
}
}

IterativeLQR::FunctionWorkspace::FunctionWorkspace(const ilqr::FunctionSpec& s, int ns, int nu):
    spec(s), function(s.function->clone()),
    z(s.function->inputSize()), value(s.function->outputSize()),
    weights(s.function->outputSize()),
    jac(s.function->outputSize(), s.function->inputSize()),
    hess(s.function->hasHessian() ? s.function->inputSize() : 0,
         s.function->hasHessian() ? s.function->inputSize() : 0)
{
    int size = ns + nu;
    for(const auto& p : spec.parameters) size += p.size;
    if(size != z.size()) throw std::invalid_argument("stage input must be [state; control; parameters]");
    z.setZero(); value.setZero(); weights.setOnes(); jac.setZero(); hess.setZero();
}
IterativeLQR::FunctionWorkspace::FunctionWorkspace(const FunctionWorkspace& other):
    spec(other.spec), function(other.function->clone()), z(other.z), value(other.value),
    weights(other.weights), jac(other.jac), hess(other.hess)
{}

void IterativeLQR::FunctionWorkspace::setInput(VecConstRef x, VecConstRef u, int k,
                                              const ParameterMapPtr& params)
{
    z.head(x.size()) = x;
    z.segment(x.size(), u.size()) = u;
    int offset = x.size() + u.size();
    for(const auto& p : spec.parameters)
    {
        z.segment(offset, p.size) = params->at(p.name).col(k);
        offset += p.size;
    }
}

void IterativeLQR::FunctionWorkspace::setDirections(MatConstRef retraction,int nu)
{
    directions.setZero(z.size(),retraction.cols()+nu);
    directions.topLeftCorner(retraction.rows(),retraction.cols())=retraction;
    directions.block(retraction.rows(),retraction.cols(),nu,nu).setIdentity();
    localJac.resize(value.size(),directions.cols());
}

IterativeLQR::IterativeLQR(const ilqr::FunctionManager& manager, const std::string& name,
                          int N, OptionDict opt, std::shared_ptr<const ilqr::StateSpace> space):
    IterativeLQR(dynamicsSignature(manager.at(name)), N, std::move(opt), std::move(space))
{
    const auto& spec = manager.at(name);
    add_param_to_map(spec);
    for(int i=0;i<_N;++i)
    {
        _dyn[i].native=std::make_shared<FunctionWorkspace>(spec,_ns,_nu);
        _dyn[i].native->function->setEvaluationContext(_evaluation_contexts[i]);
    }
}

void IterativeLQR::validateIndices(const std::vector<int>& indices) const
{
    std::set<int> unique;
    for(int k : indices)
        if(k < 0 || k > _N || !unique.insert(k).second)
            throw std::invalid_argument("node indices must be unique and in [0,N]");
}

void IterativeLQR::stateEquality(int k, Eigen::MatrixXd& C, Eigen::VectorXd& h) const
{
    if(_x_lb.col(k).allFinite() && _x_lb.col(k).isApprox(_x_ub.col(k), 0))
    {
        C.resize(_nx, _nx); h.resize(_nx);
        Eigen::MatrixXd unused(_nx, _nx);
        _space->difference(_x_lb.col(k), state(k), h);
        _space->differenceJacobians(_x_lb.col(k), state(k), unused, C);
        return;
    }
    int rows = 0;
    for(int i = 0; i < _ns; ++i) rows += _x_lb(i, k) == _x_ub(i, k);
    C.resize(rows, _nx); h.resize(rows);
    Eigen::MatrixXd Rx(_ns, _nx);
    _space->retractionJacobian(state(k), Rx);
    int row = 0;
    for(int i = 0; i < _ns; ++i)
        if(_x_lb(i, k) == _x_ub(i, k))
        {
            C.row(row) = Rx.row(i);
            h(row++) = state(k)(i) - _x_lb(i, k);
        }
}

void IterativeLQR::add_param_to_map(const ilqr::FunctionSpec& spec)
{
    for(const auto& p : spec.parameters)
    {
        auto it = _param_map->find(p.name);
        if(it != _param_map->end() && it->second.rows() != p.size)
            throw std::invalid_argument("parameter dimension conflict: " + p.name);
    }
    for(const auto& p : spec.parameters)
        if(!_param_map->count(p.name))
            (*_param_map)[p.name].setConstant(p.size, _N+1, std::numeric_limits<double>::quiet_NaN());
}

void IterativeLQR::setCost(std::vector<int> indices, const ilqr::FunctionManager& manager,
                          const std::string& name)
{
    setManagedCost(std::move(indices), manager.at(name), name, false);
}
void IterativeLQR::setResidual(std::vector<int> indices, const ilqr::FunctionManager& manager,
                              const std::string& name)
{
    setManagedCost(std::move(indices), manager.at(name), name, true);
}
void IterativeLQR::setManagedCost(std::vector<int> indices, const ilqr::FunctionSpec& spec,
                                 const std::string& name, bool residual)
{
    validateIndices(indices);
    if(!residual && (spec.function->outputSize() != 1 || !spec.function->hasHessian()))
        throw std::invalid_argument("scalar costs require one output and an exact Hessian; use residuals for Gauss-Newton");
    auto c = std::make_shared<ManagedCostEntity>(spec, name, residual, _ns, _nu);
    add_param_to_map(spec);
    c->param = _param_map;
    c->indices = std::move(indices);
    _cost_map[name] = c;
    _cost_values[name].setConstant(_N+1, std::numeric_limits<double>::quiet_NaN());
    updateIndices();
}

void IterativeLQR::setConstraint(std::vector<int> indices, const ilqr::FunctionManager& manager,
                                const std::string& name, std::vector<Eigen::VectorXd> targets)
{
    validateIndices(indices);
    const auto& spec = manager.at(name);
    if(!targets.empty() && targets.size() != indices.size())
        throw std::invalid_argument("one constraint target is required per node");
    auto c = std::make_shared<ConstraintEntity>();
    c->native = std::make_shared<FunctionWorkspace>(spec, _ns, _nu);
    c->name = name;
    c->nativeC.setZero(spec.function->outputSize(), _ns);
    c->nativeD.setZero(spec.function->outputSize(), _nu);
    c->indices = indices;
    // Initialize default target through the same checked public operation.
    c->setTargetValue(Eigen::VectorXd::Zero(spec.function->outputSize()));
    for(size_t i = 0; i < targets.size(); ++i)
    {
        if(targets[i].size() != spec.function->outputSize() || !targets[i].allFinite())
            throw std::invalid_argument("constraint target size or value invalid");
        c->targets[indices[i]] = targets[i];
    }
    add_param_to_map(spec);
    c->param = _param_map;
    _constr_map[name] = c;
    _constr_values[name].setConstant(spec.function->outputSize(), _N+1,
                                    std::numeric_limits<double>::quiet_NaN());
    updateIndices();
}

IterativeLQR::ManagedCostEntity::ManagedCostEntity(const ilqr::FunctionSpec& spec,
    std::string name, bool residual, int ns, int nu):
    workspace(std::make_shared<FunctionWorkspace>(spec, ns, nu)),
    name(std::move(name)), residual(residual)
{
    _q.setZero(ns); _r.setZero(nu);
}
IterativeLQR::CostEntityBase::Ptr IterativeLQR::ManagedCostEntity::clone() const
{
    auto result = std::make_shared<ManagedCostEntity>(*this);
    result->workspace = std::make_shared<FunctionWorkspace>(*workspace);
    return result;
}
double IterativeLQR::ManagedCostEntity::evaluate(VecConstRef x, VecConstRef u, int k)
{
    auto& w = *workspace;
    w.setInput(x, u, k, param);
    w.function->evaluate(w.z, w.value);
    _cost_eval = residual ? 0.5*w.value.squaredNorm() : w.value(0);
    return _cost_eval;
}
void IterativeLQR::ManagedCostEntity::quadratize(VecConstRef x, VecConstRef u, int k,
    Eigen::MatrixXd& Q, Eigen::MatrixXd& R, Eigen::MatrixXd& P)
{
    auto& w = *workspace;
    w.setInput(x, u, k, param);
    if(residual) w.function->valueAndJacobian(w.z,w.value,w.jac);
    else w.function->jacobian(w.z, w.jac);
    const int ns = x.size(), nu = u.size();
    const auto Jx = w.jac.leftCols(ns);
    const auto Ju = w.jac.middleCols(ns, nu);
    if(residual)
    {
        _q.noalias() = Jx.transpose()*w.value;
        _r.noalias() = Ju.transpose()*w.value;
        Q.noalias() += Jx.transpose()*Jx;
        R.noalias() += Ju.transpose()*Ju;
        P.noalias() += Ju.transpose()*Jx;
    }
    else
    {
        _q = Jx.row(0).transpose();
        _r = Ju.row(0).transpose();
        w.function->weightedHessian(w.z, w.weights, w.hess);
        Q += w.hess.topLeftCorner(ns, ns);
        R += w.hess.block(ns, ns, nu, nu);
        P += w.hess.block(ns, 0, nu, ns);
    }
}

bool IterativeLQR::ManagedCostEntity::quadratizeLocal(VecConstRef x,VecConstRef u,int k,MatConstRef retraction,
    Eigen::MatrixXd& Q,Eigen::MatrixXd& R,Eigen::MatrixXd& P,Eigen::VectorXd& q,Eigen::VectorXd& r)
{
    if(!residual) return false;
    auto& w=*workspace;w.setInput(x,u,k,param);w.setDirections(retraction,u.size());
    w.function->valueAndJacobianProduct(w.z,w.directions,w.value,w.localJac);
    const auto Jx=w.localJac.leftCols(retraction.cols());
    const auto Ju=w.localJac.rightCols(u.size());
    Q.noalias()+=Jx.transpose()*Jx;R.noalias()+=Ju.transpose()*Ju;P.noalias()+=Ju.transpose()*Jx;
    q.noalias()+=Jx.transpose()*w.value;r.noalias()+=Ju.transpose()*w.value;
    _cost_eval=.5*w.value.squaredNorm();
    return true;
}
void IterativeLQR::ConstraintEntity::linearizeLocal(VecConstRef x,VecConstRef u,int k,MatConstRef retraction)
{
    auto& w=*native;w.setInput(x,u,k,param);w.setDirections(retraction,u.size());
    w.function->valueAndJacobianProduct(w.z,w.directions,w.value,w.localJac);
    const auto target=targets.find(k);
    _hvalue=w.value-(target==targets.end()?_hdes:target->second);
    nativeC=w.localJac.leftCols(retraction.cols());nativeD=w.localJac.rightCols(u.size());
}
