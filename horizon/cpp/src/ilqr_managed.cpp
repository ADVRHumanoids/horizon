#include "horizon/codegen_function.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr_impl.h"
#include <set>

void IterativeLQR::addEvaluationBatch(const ilqr::EvaluationBatch &batch)
{
    _evaluation_batches.push_back(batch.clone());
}

void IterativeLQR::setBatchedCost(std::vector<int> indices,
                                  const std::string &name,
                                  const ilqr::BatchedCost &prototype,
                                  int offset)
{
    validateIndices(indices);
    if (indices.empty() || name.empty() || _cost_map.count(name) ||
        prototype.stateSize() != _ns || offset < 0 ||
        prototype.tangentSize() < 1 || offset > _nx - prototype.tangentSize())
    {
        throw std::invalid_argument(
            "invalid batched cost dimensions, name, or indices");
    }
    struct Entity final : CostEntityBase
    {
        std::shared_ptr<ilqr::BatchedCost> batch;
        std::string name;
        int offset;
        std::vector<int> node_to_batch;
        Ptr clone() const override
        {
            return std::make_shared<Entity>(*this);
        }
        std::string getName() override
        {
            return name;
        }
        bool gaussNewton() const override
        {
            return true;
        }
        double evaluate(VecConstRef, VecConstRef, int k) override
        {
            return _cost_eval = batch->value(node_to_batch.at(k));
        }
        void quadratize(VecConstRef, VecConstRef, int, Eigen::MatrixXd &,
                        Eigen::MatrixXd &, Eigen::MatrixXd &) override
        {
            throw std::logic_error(
                "batched cost requires tangent quadratization");
        }
        bool quadratizeLocal(VecConstRef, VecConstRef, int k,
                             const ilqr::StateSpace *, MatConstRef,
                             Eigen::MatrixXd &Q, Eigen::MatrixXd &,
                             Eigen::MatrixXd &, Eigen::VectorXd &q,
                             Eigen::VectorXd &) override
        {
            k = node_to_batch.at(k);
            _cost_eval = batch->value(k);
            const int n = batch->tangentSize();
            q.segment(offset, n) += batch->gradient(k);
            const auto h = batch->packedHessian(k);
            int entry = 0;
            for (int i = 0; i < n; ++i)
            {
                for (int j = i; j < n; ++j)
                {
                    const double v = h[entry++];
                    Q(offset + i, offset + j) += v;
                    if (i != j)
                    {
                        Q(offset + j, offset + i) += v;
                    }
                }
            }
            return true;
        }
    };
    auto entity = std::make_shared<Entity>();
    entity->batch = std::shared_ptr<ilqr::BatchedCost>(prototype.clone());
    entity->name = name;
    entity->offset = offset;
    entity->indices = std::move(indices);
    entity->node_to_batch.assign(_N + 1, -1);
    bool full = entity->indices.size() == std::size_t(_N + 1);
    for (std::size_t i = 0; i < entity->indices.size(); ++i)
    {
        entity->node_to_batch[entity->indices[i]] = i;
        full = full && entity->indices[i] == int(i);
    }
    _batched_costs.push_back({entity->batch, entity->indices, {}, full});
    _cost_map[name] = entity;
    _cost_values[name].setConstant(_N + 1,
                                   std::numeric_limits<double>::quiet_NaN());
    updateIndices();
}

void IterativeLQR::prepareEvaluationBatches(const Eigen::MatrixXd &states,
                                            const Eigen::MatrixXd &controls,
                                            bool derivatives)
{
    TIC(prepare_evaluation_batches);
    // Invalidate once before preparation, so node evaluation can reuse native
    // values/Jacobians computed while gathering the horizon batch.
    for (auto &context : _evaluation_contexts)
    {
        context->invalidate();
    }
    for (auto &batch : _evaluation_batches)
    {
        batch->prepare(states, controls, _evaluation_contexts, derivatives);
    }
    for (auto &registration : _batched_costs)
    {
        if (registration.full_horizon)
        {
            registration.batch->prepare(states, derivatives);
        }
        else
        {
            auto &selected = registration.selected_states;
            selected.resize(states.rows(), registration.indices.size());
            for (std::size_t i = 0; i < registration.indices.size(); ++i)
            {
                selected.col(i) = states.col(registration.indices[i]);
            }
            registration.batch->prepare(selected, derivatives);
        }
    }
}

namespace
{
// Only the signature is used to share legacy solver initialization. Numerical
// calls are replaced by cloned native evaluators before construction returns.
casadi::Function dynamicsSignature(const ilqr::FunctionSpec &spec)
{
    if (!spec.function)
    {
        throw std::invalid_argument("null dynamics function");
    }
    const int ns = spec.function->outputSize();
    int nu = spec.function->inputSize() - ns;
    for (const auto &p : spec.parameters)
    {
        nu -= p.size;
    }
    if (nu <= 0)
    {
        throw std::invalid_argument(
            "dynamics must have state and control inputs");
    }
    auto x = cs::SX::sym("x", ns);
    auto u = cs::SX::sym("u", nu);
    return cs::Function("managed_dynamics", {x, u}, {x}, {"x", "u"}, {"f"});
}

casadi::Function managedLocal(const ilqr::FunctionSpec &spec,
                              const ilqr::StateSpace &space,
                              ilqr::LocalDerivativeKind kind)
{
    const auto *source =
        dynamic_cast<const ilqr::SymbolicStageSource *>(spec.function.get());
    if (!source || dynamic_cast<const ilqr::EuclideanSpace *>(&space))
    {
        return {};
    }
    return ilqr::compileLocalDerivatives(source->symbolicStage(), space, kind);
}
} // namespace

IterativeLQR::FunctionWorkspace::FunctionWorkspace(const ilqr::FunctionSpec &s,
                                                   int ns, int nu)
    : spec(s), function(s.function->clone()), z(s.function->inputSize()),
      value(s.function->outputSize()), weights(s.function->outputSize())
{
    int size = ns + nu;
    for (const auto &p : spec.parameters)
    {
        size += p.size;
    }
    if (size != z.size())
    {
        throw std::invalid_argument(
            "stage input must be [state; control; parameters]");
    }
    z.setZero();
    value.setZero();
    weights.setOnes();
}
IterativeLQR::FunctionWorkspace::FunctionWorkspace(
    const FunctionWorkspace &other)
    : spec(other.spec), function(other.function->clone()), local(other.local),
      z(other.z), value(other.value), weights(other.weights), jac(other.jac),
      hess(other.hess)
{
}

void IterativeLQR::FunctionWorkspace::setInput(VecConstRef x, VecConstRef u,
                                               int k,
                                               const ParameterMapPtr &params)
{
    z.head(x.size()) = x;
    z.segment(x.size(), u.size()) = u;
    int offset = x.size() + u.size();
    for (const auto &p : spec.parameters)
    {
        z.segment(offset, p.size) = params->at(p.name).col(k);
        offset += p.size;
    }
}

void IterativeLQR::FunctionWorkspace::setDirections(MatConstRef retraction,
                                                    int nu)
{
    directions.setZero(z.size(), retraction.cols() + nu);
    directions.topLeftCorner(retraction.rows(), retraction.cols()) = retraction;
    directions.block(retraction.rows(), retraction.cols(), nu, nu)
        .setIdentity();
    localJac.resize(value.size(), directions.cols());
}

void IterativeLQR::FunctionWorkspace::evaluateLocal()
{
    int offset = 0;
    for (int i = 0; i < local.function().n_in(); ++i)
    {
        const int size = local.function().size1_in(i);
        local.setInput(i, z.segment(offset, size));
        offset += size;
    }
    local.call();
    value = local.getOutput(0);
    localJac = local.getOutput(1);
}

ilqr::TangentFunction *
IterativeLQR::FunctionWorkspace::matchingTangent(const ilqr::StateSpace *space)
{
    auto *tangent = dynamic_cast<ilqr::TangentFunction *>(function.get());
    const auto *product = tangent ? dynamic_cast<const ilqr::ProductSpace *>(
                                        &tangent->tangentInputSpace())
                                  : nullptr;
    if (space && product && !function->hasJacobianProduct() &&
        product->components().size() == 2 &&
        product->components()[0].get() == space &&
        dynamic_cast<const ilqr::EuclideanSpace *>(
            product->components()[1].get()))
    {
        return tangent;
    }
    return nullptr;
}

void IterativeLQR::FunctionWorkspace::linearizeLocal(
    const ilqr::StateSpace *space, MatConstRef retraction, int nu)
{
    if (local.is_valid())
    {
        evaluateLocal();
        return;
    }
    auto *graph =
        dynamic_cast<ilqr::StageTangentLinearization *>(function.get());
    if (space && graph)
    {
        localJac.resize(value.size(), retraction.cols() + nu);
        graph->valueAndStageLinearization(z, *space, nu, retraction, value,
                                          localJac);
        return;
    }
    if (auto *tangent = matchingTangent(space))
    {
        const auto &product = tangent->tangentInputSpace();
        tangentJac.resize(value.size(), product.tangentSize());
        tangent->valueAndTangentLinearization(z, value, tangentJac);
        localJac = tangentJac.leftCols(retraction.cols() + nu);
        return;
    }
    setDirections(retraction, nu);
    function->valueAndJacobianProduct(z, directions, value, localJac);
}

IterativeLQR::IterativeLQR(const ilqr::FunctionManager &manager,
                           const std::string &name, int N, OptionDict opt,
                           std::shared_ptr<const ilqr::StateSpace> space)
    : IterativeLQR(dynamicsSignature(manager.at(name)), N, std::move(opt),
                   std::move(space))
{
    const auto &spec = manager.at(name);
    add_param_to_map(spec);
    auto compiled =
        managedLocal(spec, *_space, ilqr::LocalDerivativeKind::Dynamics);
    if (_codegen_enabled && !compiled.is_null())
    {
        compiled = utils::codegen(compiled, _codegen_workdir);
    }
    for (int i = 0; i < _N; ++i)
    {
        _dyn[i].native = std::make_shared<FunctionWorkspace>(spec, _ns, _nu);
        if (!compiled.is_null())
        {
            _dyn[i].native->local = compiled;
        }
        _dyn[i].native->function->setEvaluationContext(_evaluation_contexts[i]);
    }
}

void IterativeLQR::validateIndices(const std::vector<int> &indices) const
{
    std::set<int> unique;
    for (int k : indices)
    {
        if (k < 0 || k > _N || !unique.insert(k).second)
        {
            throw std::invalid_argument(
                "node indices must be unique and in [0,N]");
        }
    }
}

void IterativeLQR::stateEquality(int k, Eigen::MatrixXd &C,
                                 Eigen::VectorXd &h) const
{
    if (_x_lb.col(k).allFinite() && _x_lb.col(k).isApprox(_x_ub.col(k), 0))
    {
        C.resize(_nx, _nx);
        h.resize(_nx);
        Eigen::MatrixXd unused(_nx, _nx);
        _space->difference(_x_lb.col(k), state(k), h);
        _space->differenceJacobians(_x_lb.col(k), state(k), unused, C);
        return;
    }
    int rows = 0;
    for (int i = 0; i < _ns; ++i)
    {
        rows += _x_lb(i, k) == _x_ub(i, k);
    }
    C.resize(rows, _nx);
    h.resize(rows);
    if (rows == 0)
    {
        return;
    }
    Eigen::MatrixXd Rx(_ns, _nx);
    _space->retractionJacobian(state(k), Rx);
    int row = 0;
    for (int i = 0; i < _ns; ++i)
    {
        if (_x_lb(i, k) == _x_ub(i, k))
        {
            C.row(row) = Rx.row(i);
            h(row++) = state(k)(i) - _x_lb(i, k);
        }
    }
}

void IterativeLQR::add_param_to_map(const ilqr::FunctionSpec &spec)
{
    for (const auto &p : spec.parameters)
    {
        auto it = _param_map->find(p.name);
        if (it != _param_map->end() && it->second.rows() != p.size)
        {
            throw std::invalid_argument("parameter dimension conflict: " +
                                        p.name);
        }
    }
    for (const auto &p : spec.parameters)
    {
        if (!_param_map->count(p.name))
        {
            (*_param_map)[p.name].setConstant(
                p.size, _N + 1, std::numeric_limits<double>::quiet_NaN());
        }
    }
}

void IterativeLQR::setCost(std::vector<int> indices,
                           const ilqr::FunctionManager &manager,
                           const std::string &name)
{
    setManagedCost(std::move(indices), manager.at(name), name, false);
}
void IterativeLQR::setResidual(std::vector<int> indices,
                               const ilqr::FunctionManager &manager,
                               const std::string &name)
{
    setManagedCost(std::move(indices), manager.at(name), name, true);
}
void IterativeLQR::setManagedCost(std::vector<int> indices,
                                  const ilqr::FunctionSpec &spec,
                                  const std::string &name, bool residual)
{
    validateIndices(indices);
    if (!residual &&
        (spec.function->outputSize() != 1 || !spec.function->hasHessian()))
    {
        throw std::invalid_argument(
            "scalar costs require one output and an exact Hessian; use "
            "residuals for Gauss-Newton");
    }
    auto c =
        std::make_shared<ManagedCostEntity>(spec, name, residual, _ns, _nu);
    auto compiled =
        managedLocal(spec, *_space,
                     residual ? ilqr::LocalDerivativeKind::Jacobian
                              : ilqr::LocalDerivativeKind::ExactScalar);
    if (!compiled.is_null())
    {
        if (_codegen_enabled)
        {
            compiled = utils::codegen(compiled, _codegen_workdir);
        }
        c->workspace->local = compiled;
    }
    add_param_to_map(spec);
    c->param = _param_map;
    c->indices = std::move(indices);
    _cost_map[name] = c;
    _cost_values[name].setConstant(_N + 1,
                                   std::numeric_limits<double>::quiet_NaN());
    updateIndices();
}

void IterativeLQR::setConstraint(std::vector<int> indices,
                                 const ilqr::FunctionManager &manager,
                                 const std::string &name,
                                 std::vector<Eigen::VectorXd> targets)
{
    validateIndices(indices);
    const auto &spec = manager.at(name);
    if (!targets.empty() && targets.size() != indices.size())
    {
        throw std::invalid_argument(
            "one constraint target is required per node");
    }
    auto c = std::make_shared<ConstraintEntity>();
    c->native = std::make_shared<FunctionWorkspace>(spec, _ns, _nu);
    auto compiled =
        managedLocal(spec, *_space, ilqr::LocalDerivativeKind::Jacobian);
    if (!compiled.is_null())
    {
        if (_codegen_enabled)
        {
            compiled = utils::codegen(compiled, _codegen_workdir);
        }
        c->native->local = compiled;
    }
    c->name = name;
    c->nativeC.setZero(spec.function->outputSize(), _ns);
    c->nativeD.setZero(spec.function->outputSize(), _nu);
    c->indices = indices;
    // Initialize default target through the same checked public operation.
    c->setTargetValue(Eigen::VectorXd::Zero(spec.function->outputSize()));
    for (size_t i = 0; i < targets.size(); ++i)
    {
        if (targets[i].size() != spec.function->outputSize() ||
            !targets[i].allFinite())
        {
            throw std::invalid_argument(
                "constraint target size or value invalid");
        }
        c->targets[indices[i]] = targets[i];
    }
    add_param_to_map(spec);
    c->param = _param_map;
    _constr_map[name] = c;
    _constr_values[name].setConstant(spec.function->outputSize(), _N + 1,
                                     std::numeric_limits<double>::quiet_NaN());
    updateIndices();
}

IterativeLQR::ManagedCostEntity::ManagedCostEntity(
    const ilqr::FunctionSpec &spec, std::string name, bool residual, int ns,
    int nu)
    : workspace(std::make_shared<FunctionWorkspace>(spec, ns, nu)),
      name(std::move(name)), residual(residual)
{
    _q.setZero(ns);
    _r.setZero(nu);
}
IterativeLQR::CostEntityBase::Ptr IterativeLQR::ManagedCostEntity::clone() const
{
    auto result = std::make_shared<ManagedCostEntity>(*this);
    result->workspace = std::make_shared<FunctionWorkspace>(*workspace);
    return result;
}
double IterativeLQR::ManagedCostEntity::evaluate(VecConstRef x, VecConstRef u,
                                                 int k)
{
    auto &w = *workspace;
    w.setInput(x, u, k, param);
    w.function->evaluate(w.z, w.value);
    _cost_eval = residual ? 0.5 * w.value.squaredNorm() : w.value(0);
    return _cost_eval;
}
void IterativeLQR::ManagedCostEntity::quadratize(VecConstRef x, VecConstRef u,
                                                 int k, Eigen::MatrixXd &Q,
                                                 Eigen::MatrixXd &R,
                                                 Eigen::MatrixXd &P)
{
    auto &w = *workspace;
    w.setInput(x, u, k, param);
    w.jac.resize(w.value.size(), w.z.size());
    if (residual)
    {
        w.function->valueAndJacobian(w.z, w.value, w.jac);
    }
    else
    {
        w.function->jacobian(w.z, w.jac);
    }
    const int ns = x.size(), nu = u.size();
    const auto Jx = w.jac.leftCols(ns);
    const auto Ju = w.jac.middleCols(ns, nu);
    if (residual)
    {
        _q.noalias() = Jx.transpose() * w.value;
        _r.noalias() = Ju.transpose() * w.value;
        Q.noalias() += Jx.transpose() * Jx;
        R.noalias() += Ju.transpose() * Ju;
        P.noalias() += Ju.transpose() * Jx;
    }
    else
    {
        _q = Jx.row(0).transpose();
        _r = Ju.row(0).transpose();
        w.hess.resize(w.z.size(), w.z.size());
        w.function->weightedHessian(w.z, w.weights, w.hess);
        Q += w.hess.topLeftCorner(ns, ns);
        R += w.hess.block(ns, ns, nu, nu);
        P += w.hess.block(ns, 0, nu, ns);
    }
}

bool IterativeLQR::ManagedCostEntity::quadratizeLocal(
    VecConstRef x, VecConstRef u, int k, const ilqr::StateSpace *space,
    MatConstRef retraction, Eigen::MatrixXd &Q, Eigen::MatrixXd &R,
    Eigen::MatrixXd &P, Eigen::VectorXd &q, Eigen::VectorXd &r)
{
    auto *tangent = !residual && !workspace->local.is_valid()
        ? workspace->matchingTangent(space)
        : nullptr;
    auto *graph = space && !residual && !workspace->local.is_valid()
        ? dynamic_cast<ilqr::StageTangentQuadratic *>(workspace->function.get())
        : nullptr;
    if (!residual && !workspace->local.is_valid() && !tangent && !graph)
    {
        return false;
    }
#ifdef HORIZON_PROFILING
    utils::Timer cost_timer("quadratize_cost_" + name, on_timer_toc);
#endif
    auto &w = *workspace;
    w.setInput(x, u, k, param);
    if (graph)
    {
        const int nx = retraction.cols(), nu = u.size();
        w.localGradient.resize(nx + nu);
        w.localHess.resize(nx + nu, nx + nu);
        graph->valueAndStageQuadratic(w.z, *space, nu, retraction, w.value,
                                      w.localGradient, w.localHess);
        Q += w.localHess.topLeftCorner(nx, nx);
        R += w.localHess.bottomRightCorner(nu, nu);
        P += w.localHess.bottomLeftCorner(nu, nx);
        q += w.localGradient.head(nx);
        r += w.localGradient.tail(nu);
        _cost_eval = w.value(0);
        return true;
    }
    if (tangent)
    {
        const int nx = retraction.cols(), nu = u.size();
        const int nt = tangent->tangentInputSpace().tangentSize();
        w.tangentJac.resize(w.value.size(), nt);
        w.tangentHess.resize(nt, nt);
        tangent->valueAndTangentLinearization(w.z, w.value, w.tangentJac);
        tangent->weightedTangentHessian(w.z, w.weights, w.tangentHess);
        Q += w.tangentHess.topLeftCorner(nx, nx);
        R += w.tangentHess.block(nx, nx, nu, nu);
        P += w.tangentHess.block(nx, 0, nu, nx);
        q += w.tangentJac.row(0).head(nx).transpose();
        r += w.tangentJac.row(0).segment(nx, nu).transpose();
        _cost_eval = w.value(0);
        return true;
    }
    if (!residual)
    {
        w.evaluateLocal();
        const int nx = retraction.cols(), nu = u.size();
        const auto hessian = w.local.getOutput(2);
        Q += hessian.topLeftCorner(nx, nx);
        R += hessian.bottomRightCorner(nu, nu);
        P += hessian.bottomLeftCorner(nu, nx);
        q += w.localJac.leftCols(nx).row(0).transpose();
        r += w.localJac.rightCols(nu).row(0).transpose();
        _cost_eval = w.value(0);
        return true;
    }
    {
#ifdef HORIZON_PROFILING
        utils::Timer derivative_timer("quadratize_derivatives_" + name,
                                      on_timer_toc);
#endif
        w.linearizeLocal(space, retraction, u.size());
    }
    const auto Jx = w.localJac.leftCols(retraction.cols());
    const auto Ju = w.localJac.rightCols(u.size());
    Q.noalias() += Jx.transpose() * Jx;
    R.noalias() += Ju.transpose() * Ju;
    P.noalias() += Ju.transpose() * Jx;
    q.noalias() += Jx.transpose() * w.value;
    r.noalias() += Ju.transpose() * w.value;
    _cost_eval = .5 * w.value.squaredNorm();
    return true;
}
void IterativeLQR::ConstraintEntity::linearizeLocal(
    VecConstRef x, VecConstRef u, int k, const ilqr::StateSpace *space,
    MatConstRef retraction)
{
#ifdef HORIZON_PROFILING
    utils::Timer timer("linearize_" + name + "_inner", on_timer_toc);
#endif
    const auto target = targets.find(k);
    const auto &desired = target == targets.end() ? _hdes : target->second;
    if (native)
    {
        auto &w = *native;
        w.setInput(x, u, k, param);
        w.linearizeLocal(space, retraction, u.size());
        _hvalue = w.value - desired;
        nativeC = w.localJac.leftCols(retraction.cols());
        nativeD = w.localJac.rightCols(u.size());
    }
    else
    {
        local.setInput(0, x);
        local.setInput(1, u);
        set_param_inputs(param, k, local);
        local.call();
        _hvalue = local.getOutput(0).col(0) - desired;
        nativeC = local.getOutput(1).leftCols(retraction.cols());
        nativeD = local.getOutput(1).rightCols(u.size());
    }
}
