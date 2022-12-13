#include "ilqr_impl.h"


bool IterativeLQR::forward_pass(double alpha)
{
    TIC(forward_pass);

    // reset values
    _fp_res->accepted = false;
    _fp_res->alpha = alpha;
    _fp_res->step_length = 0.0;
    _fp_res->hxx_reg = _hxx_reg;

    // initialize forward pass with initial state
    _fp_res->xtrj.col(0) = alpha*_bp_res[0].dx + state(0);

    // do forward pass
    for(int i = 0; i < _N; i++)
    {
        forward_pass_iter(i, alpha);
    }

    // set cost value and constraint violation after the forward pass
    _fp_res->cost = compute_cost(_fp_res->xtrj, _fp_res->utrj);
    _fp_res->constraint_violation = compute_constr(_fp_res->xtrj, _fp_res->utrj);
    _fp_res->defect_norm = compute_defect(_fp_res->xtrj, _fp_res->utrj);
    _fp_res->bound_violation = compute_bound_penalty(_fp_res->xtrj, _fp_res->utrj);

    return true;
}

void IterativeLQR::forward_pass_iter(int i, double alpha)
{
    TIC(forward_pass_inner)

    // note!
    // this function will update the control at t = i, and
    // the state at t = i+1

    // some shorthands
    const auto xnext = state(i+1);
    const auto xi = state(i);
    const auto ui = input(i);
    const auto xi_upd = _fp_res->xtrj.col(i);
    auto& tmp = _tmp[i];
    tmp.dx = xi_upd - xi;

    // dynamics
    const auto& dyn = _dyn[i];
    const auto& A = dyn.A();
    const auto& B = dyn.B();
    const auto& d = dyn.d;

    // backward pass solution
    const auto& res = _bp_res[i];
    const auto& L = res.Lu;

    // update control
    tmp.du = alpha * res.lu;

    if(_closed_loop_forward_pass)
    {
        tmp.du += L * tmp.dx;
    }

    _fp_res->utrj.col(i) = ui + tmp.du;

    // update next state
    auto xnext_upd = xnext + A*tmp.dx + B*tmp.du + alpha*d;
    _fp_res->xtrj.col(i+1) = xnext_upd;

#if false
    // compute largest multiplier..
    // ..for dynamics (lam_x = S*dx + s)
    tmp.lam_x = _value[i].S*tmp.dx + _value[i].s;
    _fp_res->lam_x_max = std::max(_fp_res->lam_x_max, tmp.lam_x.cwiseAbs().maxCoeff());

    // ..for constraints (lam_g = TBD)
    if(res.nc > 0)
    {
        tmp.lam_g = res.glam + res.Gu*(l + L*tmp.dx) + res.Gx*tmp.dx;
        _fp_res->lam_g_max = std::max(_fp_res->lam_g_max, tmp.lam_g.cwiseAbs().maxCoeff());
    }
#endif

    // compute step length
    _fp_res->step_length += tmp.du.cwiseAbs().sum();

}

double IterativeLQR::compute_merit_slope(double cost_slope,
                                         double mu_f,
                                         double mu_c,
                                         double defect_norm,
                                         double constr_viol)
{
    // see Nocedal and Wright, Theorem 18.2, pg. 541
    // available online http://www.apmath.spbu.ru/cnsa/pdf/monograf/Numerical_Optimization2006.pdf

    return cost_slope - mu_f*defect_norm - mu_c*constr_viol;

}

double IterativeLQR::compute_cost_slope()
{
    TIC(compute_cost_slope);

    double der = 0.;

    Eigen::VectorXd dx, du;
    dx = _bp_res[0].dx;

    for(int i = 0; i < _N; i++)
    {
        du = _bp_res[i].lu + _bp_res[i].Lu*dx;
        der += _cost[i].r().dot(du) + _cost[i].q().dot(dx);
        dx = _dyn[i].A()*dx + _dyn[i].B()*du + _dyn[i].d;
    }

    return der;
}

double IterativeLQR::compute_merit_value(double mu_f,
                                         double mu_c,
                                         double cost,
                                         double defect_norm,
                                         double constr_viol)
{
    // we define a merit function as follows
    // m(alpha) = J + mu_f * |D| + mu_c * |G|
    // where:
    //   i) J is the cost
    //  ii) D is the vector of defects (or gaps)
    // iii) G is the vector of equality constraints
    //  iv) mu_f (feasibility) is an estimate of the largest lag. mult.
    //      for the dynamics constraint (a.k.a. co-state)
    //   v) mu_c (constraint) is an estimate of the largest lag. mult.
    //      for the equality constraints

    return cost + mu_f*defect_norm + mu_c*constr_viol;
}


std::pair<double, double> IterativeLQR::compute_merit_weights(
        double cost_der,
        double defect_norm,
        double constr_viol)
{
    TIC(compute_merit_weights);

    // note: we here assume dx = 0, since this function runs before
    // the forward pass

    double lam_x_max = 0.0;
    double lam_g_max = 0.0;

    for(int i = 0; i < _N; i++)
    {
        auto& res = _bp_res[i];

        // compute largest multiplier..
        // ..for dynamics (lam_x = S*dx + s)
        _lam_x.col(i) = _value[i].s;
        lam_x_max = std::max(lam_x_max, _lam_x.col(i).cwiseAbs().maxCoeff());

        // ..for constraints (lam_g = TBD)
        if(res.glam.size() > 0)
        {
            _lam_g[i] = res.glam;
            lam_g_max = std::max(lam_g_max, _lam_g[i].cwiseAbs().maxCoeff());
        }
    }

    const double merit_safety_factor = 2.0;
    double mu_f = lam_x_max * merit_safety_factor;
    double mu_c = std::max(lam_g_max * merit_safety_factor, 0.0);

//    double g = defect_norm + mu_c/mu_f*constr_viol;
//    double rho = 0.5;
//    double mu = 2.0 * cost_der / ((1 - rho)*g);
//    mu = std::max(mu, 0.0);

    return {mu_f, mu_c};

}

double IterativeLQR::compute_cost(const Eigen::MatrixXd& xtrj, const Eigen::MatrixXd& utrj)
{
    TIC(compute_cost);

    double cost = 0.0;

    // intermediate cost
    for(int i = 0; i < _N; i++)
    {
        cost += _cost[i].evaluate(xtrj.col(i), utrj.col(i), i);
    }

    // add final cost
    // note: u not used
    // todo: enforce this!
    cost += _cost[_N].evaluate(xtrj.col(_N), utrj.col(_N-1), _N);

    return cost / _N;
}

double IterativeLQR::compute_bound_penalty(const Eigen::MatrixXd &xtrj,
                                           const Eigen::MatrixXd &utrj)
{
    TIC(compute_bound_penalty);

    double res = 0.0;

    auto xineq = _x_lb.array() < _x_ub.array();
    auto uineq = _u_lb.array() < _u_ub.array();

    res += xineq.select(_x_lb - xtrj, 0).cwiseMax(0).lpNorm<1>();
    res += xineq.select(_x_ub - xtrj, 0).cwiseMin(0).lpNorm<1>();
    res += uineq.select(_u_lb - utrj, 0).cwiseMax(0).lpNorm<1>();
    res += uineq.select(_u_ub - utrj, 0).cwiseMin(0).lpNorm<1>();

    return res / _N;
}

double IterativeLQR::compute_constr(const Eigen::MatrixXd& xtrj, const Eigen::MatrixXd& utrj)
{
    TIC(compute_constr);

    double constr = 0.0;

    // intermediate constraint violation
    for(int i = 0; i < _N; i++)
    {
        if(!_constraint[i].is_valid())
        {
            continue;
        }

        _constraint[i].evaluate(xtrj.col(i), utrj.col(i), i);
        _fp_res->constraint_values[i] = _constraint[i].h().lpNorm<1>();
        constr += _fp_res->constraint_values[i];

    }

    // state and input equality constraint violation
    auto xeq = _x_lb.array() == _x_ub.array();
    constr += xeq.select(_x_lb - xtrj, 0).lpNorm<1>();

    auto ueq = _u_lb.array() == _u_ub.array();
    constr += ueq.select(_u_lb - utrj, 0).lpNorm<1>();

    // add final constraint violation
    if(_constraint[_N].is_valid())
    {
        // note: u not used
        // todo: enforce this!
        _constraint[_N].evaluate(xtrj.col(_N), utrj.col(_N-1), _N);
        _fp_res->constraint_values[_N] = _constraint[_N].h().lpNorm<1>();
        constr += _fp_res->constraint_values[_N];
    }

    return constr / _N;
}

double IterativeLQR::compute_defect(const Eigen::MatrixXd& xtrj, const Eigen::MatrixXd& utrj)
{
    TIC(compute_defect);

    double defect = 0.0;

    // compute defects on given trajectory
    for(int i = 0; i < _N; i++)
    {
        _dyn[i].computeDefect(xtrj.col(i),
                              utrj.col(i),
                              xtrj.col(i+1),
                              i,
                              _tmp[i].defect);

        defect += _tmp[i].defect.lpNorm<1>();

        _fp_res->defect_values.col(i) = _tmp[i].defect;
    }

    return defect / _N;
}

bool IterativeLQR::line_search(int iter)
{
    TIC(line_search);

    const double step_reduction_factor = 0.5;
    const double alpha_min = _alpha_min;
    double alpha = _step_length;
    const double eta = _line_search_accept_ratio;


    // compute merit function weights
    double cost_der = compute_cost_slope();
    auto [mu_f, mu_c] = compute_merit_weights(
            cost_der,
            _fp_res->defect_norm,
            _fp_res->constraint_violation);

    _fp_res->mu_f = mu_f;
    _fp_res->mu_c = mu_c;
    _fp_res->rho = _rho;

    // compute merit function initial value
    double merit = compute_merit_value(mu_f, mu_c,
            _fp_res->cost,
            _fp_res->defect_norm,
            _fp_res->constraint_violation);

    // compute merit function directional derivative

    double merit_der = compute_merit_slope(cost_der,
            mu_f, mu_c,
            _fp_res->defect_norm,
            _fp_res->constraint_violation);

    _fp_res->f_der = cost_der;
    _fp_res->merit_der = merit_der;

    if(iter == 0 && !_rti)
    {
        reset_iterate_filter();
        _fp_res->accepted = true;

        _fp_res->alpha = 0;
        _fp_res->accepted = iter == 0;
        _fp_res->merit = merit;
        report_result(*_fp_res);
    }



    // run line search
    while(alpha >= alpha_min)
    {
        // run forward pass
        forward_pass(alpha);

        // compute merit
        _fp_res->merit = compute_merit_value(mu_f, mu_c,
                                             _fp_res->cost,
                                             _fp_res->defect_norm,
                                             _fp_res->constraint_violation);

        if(_use_it_filter)
        {
            // evaluate filter
            IterateFilter::Pair test_pair;
            test_pair.f = _fp_res->cost;
            test_pair.h = _fp_res->defect_norm + _fp_res->constraint_violation;
            _fp_res->accepted = _it_filt.add(test_pair);
        }
        else
        {
            // evaluate Armijo's condition
            _fp_res->accepted = _fp_res->merit <= merit + eta*alpha*merit_der;
        }

        // if full step ok, we can reduce regularization
        if(_fp_res->accepted)
        {
            ++_fp_accepted;
        }

        if(alpha < _step_length/4.)
        {
            _fp_accepted = 0;
        }

        // if line search disabled, we accept
        if(!_enable_line_search)
        {
            _fp_res->accepted = true;
        }


        // invoke user defined callback
        report_result(*_fp_res);

        if(_fp_res->accepted)
        {
            break;
        }

        // reduce step size and try again
        alpha *= step_reduction_factor;
    }

    if(!_fp_res->accepted)
    {
        report_result(*_fp_res);
        std::cout << "[ilqr] line search failed, increasing regularization..\n";
        increase_regularization();
        _fp_accepted = 0;
        return false;
    }


    if(_enable_line_search &&
            _fp_res->alpha > 0.1)
    {
        reduce_regularization();
        _fp_accepted = 0;
    }

    _xtrj = _fp_res->xtrj;
    _utrj = _fp_res->utrj;

    // save result in history
    _fp_res_history.push_back(*_fp_res);

    // note: we should update the lag mult at the solution
    // by including the dx part

    return true;
}

void IterativeLQR::reset_iterate_filter()
{
    _it_filt.clear();
    IterateFilter::Pair test_pair;
    test_pair.f = std::numeric_limits<double>::lowest();
    test_pair.h = _fp_res->defect_norm + _fp_res->constraint_violation;
    test_pair.h = std::max(1e2*test_pair.h, 1e3);
}

bool IterativeLQR::should_stop()
{

    const double constraint_violation_threshold = _constraint_violation_threshold;
    const double defect_norm_threshold = _defect_norm_threshold;
    const double merit_der_threshold = _merit_der_threshold;
    const double step_length_threshold = _step_length_threshold;

    TIC(should_stop);

    // first, evaluate feasibility
    if(_fp_res->constraint_violation > constraint_violation_threshold)
    {
        return false;
    }

    if(_fp_res->defect_norm > defect_norm_threshold)
    {
        return false;
    }

    if(_enable_auglag &&
            _fp_res->bound_violation > constraint_violation_threshold)
    {
        return false;
    }


    // here we're feasible

    // exit if merit function directional derivative (normalized)
    // is too close to zero
    if(std::fabs(_fp_res->f_der) < merit_der_threshold*(1 + _fp_res->cost))
    {
        std::cout << "exiting due to small merit derivative \n";
        return true;
    }

    // exit if step size (normalized) is too short
    if(_fp_res->step_length < step_length_threshold*(1 + _utrj.norm()))
    {
        std::cout << "exiting due to small control increment \n";
        return true;
    }

    return false;
}
