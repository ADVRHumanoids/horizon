#include "ilqr/ilqr_impl.h"
#include <Eigen/SparseQR>
#include <algorithm>
#include <cmath>

namespace
{
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using QR = Eigen::SparseQR<Sparse, Eigen::COLAMDOrdering<int>>;
void block(std::vector<Triplet> &entries, int row, int col, const Matrix &value,
           double scale = 1.)
{
    for (int j = 0; j < value.cols(); ++j)
    {
        for (int i = 0; i < value.rows(); ++i)
        {
            if (value(i, j) != 0)
            {
                entries.emplace_back(row + i, col + j, scale * value(i, j));
            }
        }
    }
}
double norm(const Vector &step, const Vector &scale)
{
    return std::sqrt(step.dot(scale.cwiseProduct(step)));
}
} // namespace

void IterativeLQR::prepareGlobalization()
{
    auto model = std::make_unique<GlobalizationModel>();
    auto &m = *model;
    const int controls = (_N + 1) * _nx, variables = controls + _N * _nu;
    m.g.setZero(variables);
    std::vector<Triplet> hessian, jacobian;
    std::vector<double> residual;
    auto append = [&](const Vector &r, double weight)
    {
        const int row = residual.size();
        for (double v : r)
        {
            residual.push_back(weight * v);
        }
        return row;
    };
    for (int k = 0; k <= _N; ++k)
    {
        const auto &c = _cost[k];
        m.g.segment(k * _nx, _nx) = c.q();
        block(hessian, k * _nx, k * _nx, c.Q());
        if (k < _N)
        {
            m.g.segment(controls + k * _nu, _nu) = c.r();
            block(hessian, controls + k * _nu, controls + k * _nu, c.R());
            block(hessian, controls + k * _nu, k * _nx, c.P());
            block(hessian, k * _nx, controls + k * _nu, c.P().transpose());
        }
    }
    for (int k = 0; k < _N; ++k)
    {
        const auto &dyn = _dyn[k];
        const bool euclidean =
            dynamic_cast<const ilqr::EuclideanSpace *>(_space.get());
        // The step solvers premultiply manifold defects by -Dfrom^{-1}.
        // Undo that row transform for predictions in the actual residual norm.
        Matrix left =
            euclidean ? Matrix::Identity(_nx, _nx) : Matrix(-dyn.Dfrom);
        const int row = append(left * dyn.d, 1. / _N);
        block(jacobian, row, k * _nx, left * dyn.A(), 1. / _N);
        block(jacobian, row, controls + k * _nu, left * dyn.B(), 1. / _N);
        block(jacobian, row, (k + 1) * _nx, -left, 1. / _N);
    }
    m.dynamics_rows = residual.size();
    for (int k = 0; k <= _N; ++k)
    {
        const double weight = k < _N ? 1. / _N : 1.;
        const auto &c = _constraint[k];
        const int row = append(c.h(), weight);
        block(jacobian, row, k * _nx, c.C(), weight);
        if (k < _N)
        {
            block(jacobian, row, controls + k * _nu, c.D(), weight);
        }
    }
    m.hard_begin = residual.size();
    for (int k = 0; k <= _N; ++k)
    {
        Matrix C;
        Vector h;
        stateEquality(k, C, h);
        const int row = append(h, 1. / _N);
        block(jacobian, row, k * _nx, C, 1. / _N);
        if (k < _N)
        {
            for (int j = 0; j < _nu; ++j)
            {
                if (_u_lb(j, k) == _u_ub(j, k))
                {
                    jacobian.emplace_back(residual.size(),
                                          controls + k * _nu + j, 1. / _N);
                    residual.push_back((_utrj(j, k) - _u_lb(j, k)) / _N);
                }
            }
        }
    }
    m.residual = Vector::Map(residual.data(), residual.size());
    m.H.resize(variables, variables);
    m.H.setFromTriplets(hessian.begin(), hessian.end());
    m.J.resize(residual.size(), variables);
    m.J.setFromTriplets(jacobian.begin(), jacobian.end());
    m.scale = m.H.diagonal().cwiseAbs();
    const double floor = std::max(1e-8, 1e-6 * m.scale.maxCoeff());
    m.scale = m.scale.cwiseMax(floor);
    _global_xdiag = Eigen::Map<const Matrix>(m.scale.data(), _nx, _N + 1);
    _global_udiag =
        Eigen::Map<const Matrix>(m.scale.data() + controls, _nu, _N);
    m.cost = _fp_res->cost;
    m.violation = m.residual.lpNorm<1>();
    // Project the undamped gradient onto the equality-constraint nullspace.
    // This check cannot be fooled by damping that makes a Newton step tiny.
    Sparse scaledJ = m.J * m.scale.cwiseSqrt().cwiseInverse().asDiagonal();
    Sparse transpose = scaledJ.transpose();
    QR qr;
    if (m.violation <=
        10 * std::max(_constraint_violation_threshold, _defect_norm_threshold))
    {
        qr.compute(transpose);
    }
    Vector gradient = m.g.cwiseQuotient(m.scale.cwiseSqrt());
    m.stationarity = std::numeric_limits<double>::infinity();
    if (m.violation <= 10 *
                std::max(_constraint_violation_threshold,
                         _defect_norm_threshold) &&
        qr.info() == Eigen::Success)
    {
        Vector dual = qr.solve(gradient);
        if (dual.allFinite())
        {
            m.stationarity =
                (gradient - transpose * dual).lpNorm<Eigen::Infinity>();
        }
    }
    _fp_res->stationarity = m.stationarity;
    _fp_res->damping = _damping;
    _global_model = std::move(model);
}

Eigen::VectorXd IterativeLQR::globalizationResidual(const Matrix &x,
                                                    const Matrix &u)
{
    std::vector<double> residual;
    auto append = [&](const Vector &r, double scale)
    {
        for (double v : r)
        {
            residual.push_back(scale * v);
        }
    };
    for (int k = 0; k < _N; ++k)
    {
        Vector d;
        _dyn[k].computeDefect(x.col(k), u.col(k), x.col(k + 1), k, d);
        append(d, 1. / _N);
    }
    for (int k = 0; k <= _N; ++k)
    {
        auto &c = _constraint[k];
        c.evaluate(x.col(k), u.col(std::min(k, _N - 1)), k);
        append(c.h(), k < _N ? 1. / _N : 1.);
    }
    for (int k = 0; k <= _N; ++k)
    {
        if (_x_lb.col(k).allFinite() && _x_lb.col(k).isApprox(_x_ub.col(k), 0))
        {
            Vector gap(_nx);
            _space->difference(_x_lb.col(k), x.col(k), gap);
            append(gap, 1. / _N);
        }
        else
        {
            for (int j = 0; j < _ns; ++j)
            {
                if (_x_lb(j, k) == _x_ub(j, k))
                {
                    residual.push_back((x(j, k) - _x_lb(j, k)) / _N);
                }
            }
        }
        if (k < _N)
        {
            for (int j = 0; j < _nu; ++j)
            {
                if (_u_lb(j, k) == _u_ub(j, k))
                {
                    residual.push_back((u(j, k) - _u_lb(j, k)) / _N);
                }
            }
        }
    }
    return Vector::Map(residual.data(), residual.size());
}

void IterativeLQR::evaluateGlobalizationTrial(const Vector &step)
{
    for (int k = 0; k <= _N; ++k)
    {
        _space->integrate(_xtrj.col(k), step.segment(k * _nx, _nx),
                          _fp_res->xtrj.col(k));
    }
    _fp_res->utrj =
        _utrj + Eigen::Map<const Matrix>(step.data() + (_N + 1) * _nx, _nu, _N);
    _fp_res->cost = compute_cost(_fp_res->xtrj, _fp_res->utrj);
    _fp_res->constraint_violation =
        compute_constr(_fp_res->xtrj, _fp_res->utrj);
    _fp_res->defect_norm = compute_defect(_fp_res->xtrj, _fp_res->utrj);
    _fp_res->bound_violation =
        compute_bound_penalty(_fp_res->xtrj, _fp_res->utrj);
    if (!std::isfinite(_fp_res->cost) ||
        !std::isfinite(_fp_res->constraint_violation) ||
        !std::isfinite(_fp_res->defect_norm) ||
        !std::isfinite(_fp_res->bound_violation))
    {
        throw ilqr::EvaluationError("nonfinite globalization trial");
    }
}

void IterativeLQR::restoreGlobalizationBaseline()
{
    _fp_res->xtrj = _xtrj;
    _fp_res->utrj = _utrj;
    _fp_res->cost = compute_cost(_xtrj, _utrj);
    _fp_res->constraint_violation = compute_constr(_xtrj, _utrj);
    _fp_res->defect_norm = compute_defect(_xtrj, _utrj);
    _fp_res->bound_violation = compute_bound_penalty(_xtrj, _utrj);
    // Constraint::evaluate restores h as well as public per-node values.
    globalizationResidual(_xtrj, _utrj);
    _fp_res->accepted = false;
    _fp_res->alpha = 0;
}

void IterativeLQR::increaseGlobalizationDamping()
{
    _damping = std::min(1e12, 10 * _damping);
}

// Min-scaled-norm correction, or constrained feasibility Gauss-Newton step.
Eigen::VectorXd IterativeLQR::feasibilityStep(const GlobalizationModel &m,
                                              const Vector &residual,
                                              bool correction)
{
    const int n = m.g.size(),
              hard = correction ? m.J.rows() : m.J.rows() - m.hard_begin;
    Sparse J = correction ? m.J : Sparse(m.J.bottomRows(hard));
    Sparse H(n, n);
    Vector rhs = Vector::Zero(n + hard);
    if (!correction)
    {
        Sparse soft = m.J.topRows(m.hard_begin);
        H = soft.transpose() * soft;
        rhs.head(n) = -soft.transpose() * residual.head(m.hard_begin);
    }
    std::vector<Triplet> entries;
    for (int col = 0; col < H.outerSize(); ++col)
    {
        for (Sparse::InnerIterator e(H, col); e; ++e)
        {
            entries.emplace_back(e.row(), e.col(), e.value());
        }
    }
    for (int j = 0; j < n; ++j)
    {
        entries.emplace_back(j, j, (correction ? 1. : _damping) * m.scale(j));
    }
    for (int col = 0; col < J.outerSize(); ++col)
    {
        for (Sparse::InnerIterator e(J, col); e; ++e)
        {
            entries.emplace_back(n + e.row(), e.col(), e.value());
            entries.emplace_back(e.col(), n + e.row(), e.value());
        }
    }
    rhs.tail(hard) = correction ? -residual : -residual.tail(hard).eval();
    Sparse K(n + hard, n + hard);
    K.setFromTriplets(entries.begin(), entries.end());
    QR qr;
    qr.compute(K);
    if (qr.info() != Eigen::Success)
    {
        return Vector();
    }
    Vector solution = qr.solve(rhs);
    if (!solution.allFinite() ||
        (K * solution - rhs).norm() > 1e-6 * (1 + rhs.norm()))
    {
        return Vector();
    }
    return solution.head(n);
}

bool IterativeLQR::globalizationStep()
{
    auto &m = *_global_model;
    auto &result = *_fp_res;
    result.trial_count = 0;
    result.direction_solves = 0;
    result.used_correction = false;
    result.used_restoration = false;
    result.acceptance_reason.clear();
    const bool filter = _globalization == "filter_sqp";
    // Reestimate weights between iterations; never change them within trials.
    _global_mu *= .1;
    const double filter_roundoff =
        16 * std::numeric_limits<double>::epsilon() * (1 + std::abs(m.cost));
    auto filterAccept =
        [&](double cost, double violation, double slope, double alpha)
    {
        if (violation > _filter_violation_limit)
        {
            return false;
        }
        auto improves = [&](double f, double h)
        {
            return violation <= .99 * h ||
                cost <= f - 1e-4 * h + filter_roundoff;
        };
        if (!improves(m.cost, m.violation))
        {
            return false;
        }
        for (auto [f, h] : _global_filter)
        {
            if (!improves(f, h))
            {
                return false;
            }
        }
        if (m.violation <= 1e-4 && slope < 0)
        {
            return cost <= m.cost + _line_search_accept_ratio * alpha * slope +
                filter_roundoff;
        }
        // At feasibility, zero violation alone never accepts an ascent step.
        return m.violation > _constraint_violation_threshold ||
            cost <= m.cost + filter_roundoff;
    };
    for (int retry = 0; retry < 20; ++retry)
    {
        restoreGlobalizationBaseline();
        ++result.direction_solves;
        try
        {
            if (_use_kkt_solver)
            {
                kkt_solve();
            }
            else
            {
                backward_pass();
            }
        }
        catch (const std::runtime_error &)
        {
            increaseGlobalizationDamping();
            continue;
        }
        Vector direction(m.g.size());
        direction.head(_dx.size()) = Vector::Map(_dx.data(), _dx.size());
        direction.tail(_du.size()) = Vector::Map(_du.data(), _du.size());
        if (!direction.allFinite())
        {
            increaseGlobalizationDamping();
            continue;
        }
        const double slope = m.g.dot(direction);
        const double linear_violation =
            (m.residual + m.J * direction).lpNorm<1>();
        const double reduction = m.violation - linear_violation;
        auto [mu_f, mu_c] = compute_merit_weights(slope, result.defect_norm,
                                                  result.constraint_violation);
        _global_mu = std::max({_global_mu, mu_f, mu_c});
        if (reduction > 1e-14)
        {
            _global_mu = std::max(_global_mu,
                                  (slope + std::abs(slope) * .1) / reduction);
        }
        _global_mu = std::max(0., _global_mu);
        result.mu_f = result.mu_c = _global_mu;
        result.rho = _rho;
        result.f_der = slope;
        result.merit_der = slope - _global_mu * reduction;
        result.damping = _damping;
        result.hxx_reg = _hxx_reg;
        const double baseline = m.cost + _global_mu * m.violation;
        const double roundoff = 16 * std::numeric_limits<double>::epsilon() *
            (1 + std::abs(baseline));
        for (double alpha = 1.; alpha >= _alpha_min; alpha *= .5)
        {
            Vector step = alpha * direction;
            ++result.trial_count;
            result.alpha = alpha;
            result.accepted = false;
            try
            {
                evaluateGlobalizationTrial(step);
            }
            catch (const ilqr::EvaluationError &)
            {
                continue;
            }
            Vector residual = globalizationResidual(result.xtrj, result.utrj);
            double violation = residual.lpNorm<1>();
            bool accepted =
                filter && filterAccept(result.cost, violation, slope, alpha);
            if (filter && !accepted && alpha == 1. &&
                linear_violation <= std::max(1e-7, .1 * m.violation))
            {
                Vector correction = feasibilityStep(m, residual, true);
                if (correction.size() == step.size() &&
                    norm(correction, m.scale) <= .5 * norm(step, m.scale))
                {
                    Vector corrected = step + correction;
                    ++result.trial_count;
                    try
                    {
                        evaluateGlobalizationTrial(corrected);
                        residual =
                            globalizationResidual(result.xtrj, result.utrj);
                        violation = residual.lpNorm<1>();
                        accepted =
                            filterAccept(result.cost, violation, slope, alpha);
                        if (accepted)
                        {
                            step = corrected;
                            result.used_correction = true;
                        }
                    }
                    catch (const ilqr::EvaluationError &)
                    {
                        accepted = false;
                    }
                }
            }
            const double predicted_cost =
                -m.g.dot(step) - .5 * step.dot(m.H * step);
            const double predicted_violation =
                m.violation - (m.residual + m.J * step).lpNorm<1>();
            result.predicted_reduction =
                predicted_cost + _global_mu * predicted_violation;
            result.actual_reduction =
                baseline - (result.cost + _global_mu * violation);
            result.model_agreement = result.predicted_reduction > roundoff
                ? result.actual_reduction / result.predicted_reduction
                : 0.;
            if (!filter)
            {
                accepted = result.predicted_reduction > 0 &&
                    result.merit_der < 0 &&
                    result.actual_reduction >=
                        -_line_search_accept_ratio * alpha * result.merit_der -
                            roundoff &&
                    (result.model_agreement >= 1e-4 ||
                     (std::abs(result.actual_reduction) <= roundoff &&
                      result.predicted_reduction <= roundoff));
            }
            result.merit = result.cost + _global_mu * violation;
            result.step_length = step.norm();
            result.accepted = accepted;
            result.acceptance_reason =
                accepted ? (filter ? "filter" : "merit") : "rejected";
            if (!accepted)
            {
                report_result(result);
                continue;
            }
            if (filter)
            {
                // A filter step is judged by the objective or feasibility
                // model, never by a merit weight that did not govern its
                // acceptance.
                const bool objective_step = m.violation <= 1e-4 && slope < 0;
                result.predicted_reduction =
                    objective_step ? predicted_cost : predicted_violation;
                result.actual_reduction = objective_step
                    ? m.cost - result.cost
                    : m.violation - violation;
                result.model_agreement = result.predicted_reduction > roundoff
                    ? result.actual_reduction / result.predicted_reduction
                    : 0.;
            }
            double agreement = result.model_agreement;
            // Agreement is undefined at roundoff. Retain the full Newton step
            // and release damping rather than amplifying numerical noise.
            if (result.predicted_reduction >= 0 &&
                result.predicted_reduction <= roundoff &&
                std::abs(result.actual_reduction) <= roundoff)
            {
                agreement = 1.;
            }
            result.model_agreement = agreement;
            report_result(result);
            if (agreement < .25 || alpha < .25)
            {
                increaseGlobalizationDamping();
            }
            else if (agreement > .75 && alpha == 1.)
            {
                _damping = std::max(1e-12, _damping / 3.);
            }
            if (filter)
            {
                _global_filter.erase(
                    std::remove_if(_global_filter.begin(), _global_filter.end(),
                                   [&](auto p)
                                   {
                                       return result.cost <= p.first &&
                                           violation <= p.second;
                                   }),
                    _global_filter.end());
                _global_filter.emplace_back(m.cost, m.violation);
            }
            _xtrj = result.xtrj;
            _utrj = result.utrj;
            _fp_res_history.push_back(result);
            return true;
        }
        if (filter &&
            m.violation > std::max(_constraint_violation_threshold,
                                   _defect_norm_threshold))
        {
            break;
        }
        increaseGlobalizationDamping();
    }
    restoreGlobalizationBaseline();
    if (filter && restoreFeasibility())
    {
        return true;
    }
    result.acceptance_reason =
        filter ? "restoration_failed" : "direction_limit";
    report_result(result);
    return false;
}

bool IterativeLQR::restoreFeasibility()
{
    const Matrix initial_x = _xtrj, initial_u = _utrj;
    const double initial_violation = _global_model->violation;
    _fp_res->used_restoration = true;
    if (initial_violation <=
        std::max(_constraint_violation_threshold, _defect_norm_threshold))
    {
        return false;
    }
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        // Relinearize at each restoration iterate; AL duals and penalties stay
        // frozen.
        linearize_quadratize();
        prepareGlobalization();
        auto &m = *_global_model;
        Vector step = feasibilityStep(m, m.residual, false);
        ++_fp_res->direction_solves;
        bool accepted = false;
        if (step.size() == m.g.size())
        {
            for (double alpha = 1.; alpha >= _alpha_min; alpha *= .5)
            {
                ++_fp_res->trial_count;
                try
                {
                    evaluateGlobalizationTrial(alpha * step);
                    Vector residual =
                        globalizationResidual(_fp_res->xtrj, _fp_res->utrj);
                    if (residual.squaredNorm() <=
                            (1 - 1e-4 * alpha) * m.residual.squaredNorm() &&
                        (residual.size() == m.hard_begin
                             ? 0.
                             : residual.tail(residual.size() - m.hard_begin)
                                   .lpNorm<Eigen::Infinity>()) <=
                            std::max(1e-10, _constraint_violation_threshold))
                    {
                        _fp_res->alpha = alpha;
                        _fp_res->step_length = alpha * step.norm();
                        _xtrj = _fp_res->xtrj;
                        _utrj = _fp_res->utrj;
                        accepted = true;
                        break;
                    }
                }
                catch (const ilqr::EvaluationError &)
                {
                }
            }
        }
        if (!accepted)
        {
            restoreGlobalizationBaseline();
            increaseGlobalizationDamping();
            continue;
        }
        const double violation =
            globalizationResidual(_xtrj, _utrj).lpNorm<1>();
        bool filter_ok = violation <= _filter_violation_limit;
        for (auto [f, h] : _global_filter)
        {
            filter_ok = filter_ok &&
                (violation <= .99 * h || _fp_res->cost <= f - 1e-4 * h);
        }
        if (violation <= .9 * initial_violation && filter_ok)
        {
            _fp_res->accepted = true;
            _fp_res->acceptance_reason = "restoration";
            report_result(*_fp_res);
            _fp_res_history.push_back(*_fp_res);
            return true;
        }
    }
    _xtrj = initial_x;
    _utrj = initial_u;
    restoreGlobalizationBaseline();
    return false;
}
