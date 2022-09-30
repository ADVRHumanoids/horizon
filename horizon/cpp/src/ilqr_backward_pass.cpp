#include "ilqr_impl.h"

struct HessianIndefinite : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

void IterativeLQR::backward_pass()
{
    TIC(backward_pass);

    // initialize backward recursion from final cost..
    _value.back().S = _cost.back().Q();
    _value.back().s = _cost.back().q();

    // regularize final cost
    _value.back().S.diagonal().array() += _hxx_reg;

    // ..and initialize constraints and bounds
    _constraint_to_go->set(_constraint.back());

    if(_log) std::cout << "n_constr[" << _N << "] = " <<
                  _constraint_to_go->dim() << " (before bounds)\n";

    add_bound_constraint(_N);

    if(_log) std::cout << "n_constr[" << _N << "] = " <<
                  _constraint_to_go->dim() << "\n";

    // backward pass
    int i = _N - 1;
    while(i >= 0)
    {
        try
        {
            backward_pass_iter(i);
            --i;
        }
        catch(HessianIndefinite&)
        {
            increase_regularization();
            if(_verbose) std::cout << "increasing reg at k = " << i << ", hxx_reg = " << _hxx_reg << "\n";
            // retry with increased reg
            return backward_pass();
        }
    }

    // compute dx[0]
    optimize_initial_state();

    // here we should've treated all constraints
    if(_constraint_to_go->dim() > 0)
    {
        // some of them could be infeasible unless the initial
        // satisfies them already, let's check the residual
        // from the computed dx[0]
        Eigen::VectorXd residual;
        residual = _constraint_to_go->C()*_bp_res[0].dx +
                    _constraint_to_go->h();

        // infeasible warning
        if(residual.lpNorm<1>() > 1e-8)
        {

            std::cout << "warn at k = 0: " << _constraint_to_go->dim() <<
                         " constraints not satified, residual inf-norm is " <<
                         residual.lpNorm<Eigen::Infinity>() << "\n";

            if(_log)
            {
                std::cout << "C = \n" << _constraint_to_go->C().format(2) << "\n" <<
                             "h = " << _constraint_to_go->h().transpose().format(2) << "\n";
            }
        }

    }

}

void IterativeLQR::backward_pass_iter(int i)
{
    TIC(backward_pass_inner);

    // constraint handling
    // this will filter out any constraint that can't be
    // fullfilled with the current u_k, and needs to be
    // propagated to the previous time step
    auto constr_feas = handle_constraints(i);

    // num of feasible constraints
    const int nc = constr_feas.h.size();

    // intermediate cost
    const auto& cost = _cost[i];
    const auto r = cost.r();
    const auto q = cost.q();
    const auto& Q = cost.Q();
    const auto& R = cost.R();
    const auto& P = cost.P();

    // dynamics
    const auto& dyn = _dyn[i];
    const auto& A = dyn.A();
    const auto& B = dyn.B();
    const auto& d = dyn.d;

    // ..value function
    const auto& value_next = _value[i+1];
    const auto& Snext = value_next.S;
    const auto& snext = value_next.s;

    THROW_NAN(Snext);
    THROW_NAN(snext);

    // ..workspace
    auto& tmp = _tmp[i];
    auto& K = tmp.kkt;
    auto& kx0 = tmp.kx0;
    auto& u_lam = tmp.u_lam;

    // components of next node's value function
    TIC(form_value_fn_inner);
    tmp.s_plus_S_d.noalias() = snext + Snext*d;
    tmp.S_A.noalias() = Snext*A;

    tmp.hx.noalias() = q + A.transpose()*tmp.s_plus_S_d;
    tmp.Hxx.noalias() = Q + A.transpose()*tmp.S_A;
    tmp.Hxx.diagonal().array() += _hxx_reg;

    // remaining components of next node's value function
    tmp.hu.noalias() = r + B.transpose()*tmp.s_plus_S_d;
    tmp.Huu.noalias() = R + B.transpose()*Snext*B;
    tmp.Hux.noalias() = P + B.transpose()*tmp.S_A;
    tmp.Huu.diagonal().array() += _huu_reg;
    TOC(form_value_fn_inner);

    // print
    if(_log)
    {
        Eigen::VectorXd eigS = Snext.eigenvalues().real();
        std::cout << "eig(Hxx[" << i+1 << "]) in [" <<
                     eigS.minCoeff() << ", " << eigS.maxCoeff() << "] \n";

        Eigen::VectorXd eigHuu = tmp.Huu.eigenvalues().real();
        std::cout << "eig(Huu[" << i << "]) in [" <<
                     eigHuu.minCoeff() << ", " << eigHuu.maxCoeff() << "] \n";
    }

    // todo: second-order terms from dynamics

    // form kkt matrix
    TIC(form_kkt_inner);
    K.setZero(nc + _nu, nc + _nu);
    K.topLeftCorner(_nu, _nu) = tmp.Huu;
    K.topRightCorner(_nu, nc) = constr_feas.D.transpose();
    K.bottomLeftCorner(nc, _nu) = constr_feas.D;
    K.bottomRightCorner(nc, nc).diagonal().array() -= _kkt_reg;

    kx0.resize(_nu + nc, _nx + 1);
    kx0.leftCols(_nx) << -tmp.Hux,
                         -constr_feas.C;
    kx0.col(_nx) << -tmp.hu,
                    -constr_feas.h;
    TOC(form_kkt_inner);

    // solve kkt equation
    TIC(solve_kkt_inner);
    THROW_NAN(K);
    THROW_NAN(kx0);
    switch(_kkt_decomp_type)
    {
        case Lu:
            tmp.lu.compute(K);
            u_lam = tmp.lu.solve(kx0);
            break;

        case Qr:
            tmp.qr.compute(K);
            u_lam = tmp.qr.solve(kx0);
            break;

        case Ldlt:
            tmp.ldlt.compute(K);
            u_lam = tmp.ldlt.solve(kx0);
            break;

        default:
             throw std::invalid_argument("kkt decomposition supports only qr, lu, or ldlt");

    }

    if(_log)
    {
        std::cout << "kkt_err[" << i << "] = " <<
                     (K*u_lam - kx0).lpNorm<Eigen::Infinity>() << "\n";

        std::cout << "feas_constr[" << i << "] = " <<
                      nc << "\n";

        std::cout << "infeas_constr[" << i << "] = " <<
                     _constraint_to_go->dim() << "\n";
    }

    THROW_NAN(u_lam);
    TOC(solve_kkt_inner);

    // check
    if(!u_lam.allFinite() || u_lam.hasNaN())
    {
        throw HessianIndefinite("");
    }

    // save solution
    auto& res = _bp_res[i];
    auto& Lu = res.Lu;
    auto& lu = res.lu;
    auto& lam = res.glam;
    Lu = u_lam.topLeftCorner(_nu, _nx);
    lu = u_lam.col(_nx).head(_nu);
    lam = u_lam.col(_nx).tail(nc);

    // save optimal value function
    TIC(upd_value_fn_inner);
    auto& value = _value[i];
    auto& S = value.S;
    auto& s = value.s;

    S.noalias() = tmp.Hxx + Lu.transpose()*(tmp.Huu*Lu + tmp.Hux) + tmp.Hux.transpose()*Lu;
    S = 0.5*(S + S.transpose());  // note: symmetrize
    s.noalias() = tmp.hx + tmp.Hux.transpose()*lu + Lu.transpose()*(tmp.hu + tmp.Huu*lu);
    THROW_NAN(S);
    THROW_NAN(s);
    TOC(upd_value_fn_inner);

}

void IterativeLQR::optimize_initial_state()
{
    Eigen::VectorXd& dx = _bp_res[0].dx;
    Eigen::VectorXd& lam = _bp_res[0].dx_lam;

    // typical case: initial state is fixed
    if(fixed_initial_state())
    {
        dx = _x_lb.col(0) - state(0);
        return;
    }

    // cost
    auto& S = _value[0].S;
    auto& s = _value[0].s;

    // constraints and bounds
    auto C = _constraint_to_go->C();
    auto h = _constraint_to_go->h();

    // construct kkt matrix
    TIC(construct_state_kkt);
    Eigen::MatrixXd& K = _tmp[0].x_kkt;
    K.resize(s.size() + h.size(), s.size() + h.size());
    K.topLeftCorner(S.rows(), S.cols()) = S;
    K.topRightCorner(C.cols(), C.rows()) = C.transpose();
    K.bottomLeftCorner(C.rows(), C.cols()) = C;
    K.bottomRightCorner(C.rows(), C.rows()).setZero();
    TOC(construct_state_kkt);

    // residual vector
    Eigen::VectorXd k = _tmp[0].x_k0;
    k.resize(s.size() + h.size());
    k << -s,
         -h;

    THROW_NAN(K);
    THROW_NAN(k);

    // solve kkt equation
    TIC(solve_state_kkt);
    auto& lu = _tmp[0].x_lu;
    auto& qr = _tmp[0].x_qr;
    auto& ldlt = _tmp[0].x_ldlt;
    Eigen::VectorXd& dx_lam = _tmp[0].dx_lam;

    switch(_kkt_decomp_type)
    {
        case Lu:

            lu.compute(K);
            dx_lam = lu.solve(k);
            break;

        case Qr:

            dx_lam = qr.solve(k);
            break;

        case Ldlt:

            ldlt.compute(K);
            dx_lam = ldlt.solve(k);
            break;

        default:
             throw std::invalid_argument("kkt decomposition supports only qr, lu, or ldlt");

    }
    TOC(solve_state_kkt);
    THROW_NAN(dx_lam);

    if(_log)
    {
        std::cout << "state_kkt_err = " <<
                     (K*dx_lam - k).lpNorm<Eigen::Infinity>() << "\n";
    }

    // save solution
    dx = dx_lam.head(s.size());
    lam = dx_lam.tail(h.size());

    // check constraints
    Eigen::MatrixXd Cinf = C;
    Eigen::VectorXd hinf = h;

    _constraint_to_go->clear();

    for(int i = 0; i < hinf.size(); i++)
    {
        // feasible, do nothing
        if(std::fabs(Cinf.row(i)*dx + hinf[i]) <
                _constraint_violation_threshold)
        {
            continue;
        }

        // infeasible, add it back to constraint to go
        // this will generate an infeasibility warning
        _constraint_to_go->add(Cinf.row(i),
                               hinf.row(i));
    }
}

void IterativeLQR::add_bound_constraint(int k)
{
    Eigen::RowVectorXd x_ei, u_ei;

    // state bounds
    u_ei.setZero(_nu);
    for(int i = 0; i < _nx; i++)
    {
        // if initial state is fixed, don't add
        // constraints for it
        if(k == 0 && fixed_initial_state())
        {
            continue;
        }

        // equality
        if(_x_lb(i, k) == _x_ub(i, k))
        {
            x_ei = x_ei.Unit(_nx, i);

            Eigen::Matrix<double, 1, 1> hd;
            hd(0) = _xtrj(i, k) - _x_lb(i, k);

            _constraint_to_go->add(x_ei, u_ei, hd);

            if(_log)
            {
                std::cout << k << ": detected state equality constraint (index " <<
                             i << ", value = " << _x_lb(i, k) << ") \n";
            }

        }
    }

    // input bounds
    x_ei.setZero(_nx);
    for(int i = 0; i < _nu; i++)
    {
        if(k == _N)
        {
            break;
        }

        // equality
        if(_u_lb(i, k) == _u_ub(i, k))
        {
            u_ei = u_ei.Unit(_nu, i);

            Eigen::Matrix<double, 1, 1> hd;
            hd(0) = _utrj(i, k) - _u_lb(i, k);

            _constraint_to_go->add(x_ei, u_ei, hd);

            if(_log)
            {
                std::cout << k << ": detected input equality constraint (index " <<
                             i << ", value = " << _u_lb(i, k) << ") \n";
            }
        }
    }

}

bool IterativeLQR::auglag_update()
{
    // check if we need to update the aug lag estimate
    if(!_enable_auglag)
    {
        return false;
    }

    // current solution too coarse based on merit derivative
    if(std::fabs(_fp_res->merit_der) >
            _merit_der_threshold*(1 + _fp_res->merit))
    {
        return false;
    }

    // current solution does satisfy bounds,
    // we dont need to increase rho further
    if(_fp_res->bound_violation < _constraint_violation_threshold)
    {
        return false;
    }

    if(_verbose)
    {
        std::cout << "[ilqr] performing auglag update \n";
    }

    // grow rho
    _rho *= _rho_growth_factor;

    // update lag mult estimate
    for(int i = 0; i < _N + 1; i++)
    {
        _auglag_cost[i]->update_lam(
                    _xtrj.col(i),
                    _utrj.col(i),
                    i);

        _auglag_cost[i]->setRho(_rho);

        _lam_bound_x.col(i) = _auglag_cost[i]->getStateMultiplier();

        _lam_bound_u.col(i) = _auglag_cost[i]->getInputMultiplier();
    }

    _fp_res->mu_b = _lam_bound_u.lpNorm<1>() + _lam_bound_x.lpNorm<1>();

    _fp_res->cost = compute_cost(_fp_res->xtrj, _fp_res->utrj);

    return true;
}

void IterativeLQR::increase_regularization()
{
    if(_hxx_reg < 1e-6)
    {
        _hxx_reg = 1.0;
    }

    _hxx_reg *= _hxx_reg_growth_factor;

    if(_hxx_reg < _hxx_reg_base)
    {
        _hxx_reg = _hxx_reg_base;
    }
}

void IterativeLQR::reduce_regularization()
{
    _hxx_reg /= std::pow(_hxx_reg_growth_factor, 1./3.);

    if(_hxx_reg < _hxx_reg_base)
    {
        _hxx_reg = _hxx_reg_base;
    }
}

IterativeLQR::FeasibleConstraint IterativeLQR::handle_constraints(int i)
{
    TIC(handle_constraints_inner);

    // some shorthands for..

    // ..dynamics
    auto& dyn = _dyn[i];
    const auto& A = dyn.A();
    const auto& B = dyn.B();
    const auto& d = dyn.d;  // note: has been computed during linearization phase

    // ..workspace
    auto& tmp = _tmp[i];
    auto& Cf = tmp.Cf;
    auto& Df = tmp.Df;
    auto& hf = tmp.hf;
    auto& cod = tmp.ccod;
    auto& qr = tmp.cqr;
    auto& svd = tmp.csvd;

    // ..backward pass result
    auto& res = _bp_res[i];

    TIC(constraint_prepare_inner);
    // back-propagate constraint to go from next step to current step
    _constraint_to_go->propagate_backwards(A, B, d);

    // add current step intermediate constraint
    _constraint_to_go->add(_constraint[i]);

    // add bounds
    add_bound_constraint(i);

    // number of constraints
    int nc = _constraint_to_go->dim();
    res.nc = nc;

    if(_log)
    {
        std::cout << "n_constr[" << i << "] = " <<
                      nc << "\n";
    }

    // no constraint to handle, do nothing
    if(nc == 0)
    {
        Cf.setZero(0, _nx);
        Df.setZero(0, _nu);
        hf.setZero(0);
        return FeasibleConstraint{Cf, Df, hf};
    }

    // decompose constraint into a feasible and infeasible components
    auto C = _constraint_to_go->C();
    auto D = _constraint_to_go->D();
    auto h = _constraint_to_go->h();
    TOC(constraint_prepare_inner);
    THROW_NAN(C);
    THROW_NAN(D);
    THROW_NAN(h);

    // cod of D
    TIC(constraint_decomp_inner);
    int rank = -1;
    switch(_constr_decomp_type)
    {
        case Cod:
            cod.setThreshold(_svd_threshold);
            cod.compute(D);
            rank = cod.rank();
            if(cod.maxPivot() < _svd_threshold)
            {
                rank = 0;
            }
            tmp.codQ = cod.matrixQ();
            break;

        case Qr:
            qr.setThreshold(_svd_threshold);
            qr.compute(D);
            rank = qr.rank();
            if(qr.maxPivot() < _svd_threshold)
            {
                rank = 0;
            }
            tmp.codQ = qr.matrixQ();
            if(_log)
            {
                std::cout << "matrixR diagonal entries = " <<
                qr.matrixR().diagonal().head(rank).transpose().format(2) << "\n";
            }
            break;

        case Svd:
            svd.setThreshold(_svd_threshold);
            svd.compute(D, Eigen::ComputeFullU);
            rank = svd.rank();
            if(svd.singularValues()[0] < _svd_threshold)
            {
                rank = 0;
            }
            tmp.codQ = svd.matrixU();
            break;

       default:
            throw std::invalid_argument("constraint decomposition supports only qr, svd, or cod");

    }

    THROW_NAN(tmp.codQ);
    MatConstRef codQ1 = tmp.codQ.leftCols(rank);
    MatConstRef codQ2 = tmp.codQ.rightCols(nc - rank);
    TOC(constraint_decomp_inner);

    // feasible part
    TIC(constraint_upd_to_go_inner);
    Cf.noalias() = codQ1.transpose()*C;
    Df.noalias() = codQ1.transpose()*D;
    hf.noalias() = codQ1.transpose()*h;

    // infeasible part
    Eigen::MatrixXd Cinf = codQ2.transpose()*C;
    Eigen::VectorXd hinf = codQ2.transpose()*h;

    _constraint_to_go->clear();

    for(int j = 0; j < hinf.size(); j++)
    {
        // i-th infeasible constraint is in the form 0x = 0
        if(std::fabs(hinf[j]) < 1e-9 &&
                Cinf.row(j).lpNorm<Eigen::Infinity>() < 1e-9)
        {
            std::cout << "warn at k = " << i << ": removing linearly dependent constraint \n";
            continue;
        }

        _constraint_to_go->add(Cinf.row(j),
                               hinf.row(j));
    }

    return FeasibleConstraint{Cf, Df, hf};

}


