#include "sqp.h"

using namespace horizon;


template<class CASADI_TYPE>
const casadi::DMDict& SQPGaussNewton<CASADI_TYPE>::solve(
        const casadi::DM& initial_guess_x,
        const casadi::DM& p,
        const casadi::DM& lbx,
        const casadi::DM& ubx,
        const casadi::DM& lbg,
        const casadi::DM& ubg)
{
    // clear tic toc entries
    _hessian_computation_time.clear();
    _qp_computation_time.clear();
    _line_search_time.clear();

    // initialize x0 and trajectory
    x0_ = initial_guess_x;
    casadi_utils::toEigen(x0_, _sol);
    _variable_trj[0] = x0_;
    _iteration_to_solve = 0;

    // set parameters as second input of f and df
    _f.setInput(1, Eigen::VectorXd::Map(p->data(), p.size1()));
    _df.setInput(1, Eigen::VectorXd::Map(p->data(), p.size1()));

    // do the same on g and A (i.e., dg)
    _g_dict.input[_g.name_in(1)] = p;
    _A_dict.input[_g.name_in(1)] = p;



    for(unsigned int k = 0; k < _max_iter; ++k) ///BREAK CRITERIA #1
    {
        // 1. Cost function is linearized around actual x0
        eval(_f, 0, _sol, false); // cost function
        eval(_df, 0, _sol, true); // cost function Jacobian
        _J = _df.getSparseOutput(0);

        // 2. Constraints are linearized around actual x0
        _g_dict.input[_g.name_in(0)] = x0_;
        _A_dict.input[_dg.name_in(0)] = x0_;
        eval(_g, _g_dict);
        eval(_dg, _A_dict);
        g_ = _g_dict.output[_g.name_out(0)];
        A_ = _A_dict.output[_dg.name_out(0)];

        // 2. We compute Gauss-Newton Hessian approximation and gradient function
        auto tic = std::chrono::high_resolution_clock::now();
        _H.resize(_J.cols(), _J.cols());
        _I.resize(_H.rows(), _H.cols());
        _I.setIdentity();
//        _H.selfadjointView<Eigen::Lower>().rankUpdate(_J.transpose());
        _H = _J.transpose() * _J + _eps_regularization * _I;


        auto toc = std::chrono::high_resolution_clock::now();
        _hessian_computation_time.push_back((toc-tic).count()*1E-9);

        _grad = _J.transpose()*_f.getOutput(0);

        //3. Setup QP
        casadi_utils::toCasadiMatrix(_grad, grad_);

        if(!H_.is_init())
            H_ = casadi_utils::WrappedSparseMatrix<double>(_H);
        else
            H_.update_values(_H);

        if(!_conic || _reinitialize_qp_solver)
        {
            _conic_init_input["h"] = H_.get().sparsity();
            _conic_init_input["a"] = A_.sparsity();
            _conic = std::make_unique<casadi::Function>(casadi::conic("qp_solver",
                                                                      _qp_solver,
                                                                      _conic_init_input,
                                                                      _qp_opts));
        }

        _conic_dict.input["h"] = H_.get();
        _conic_dict.input["g"] = grad_;
        _conic_dict.input["a"] = A_;
        _conic_dict.input["lba"] = lbg - g_;
        _conic_dict.input["uba"] = ubg - g_;
        _conic_dict.input["lbx"] = lbx - x0_;
        _conic_dict.input["ubx"] = ubx - x0_;
        _conic_dict.input["x0"] = x0_;
        if(_lam_a.size() > 0)
        {
            casadi::DM lama;
            casadi_utils::toCasadiMatrix(_lam_a, lama);
            _conic_dict.input["lam_a0"] = lama;
        }
        if(_lam_x.size() > 0)
        {
            casadi::DM lamx;
            casadi_utils::toCasadiMatrix(_lam_x, lamx);
            _conic_dict.input["lam_x0"] = lamx;
        }

        tic = std::chrono::high_resolution_clock::now();
        _conic->call(_conic_dict.input, _conic_dict.output);
        toc = std::chrono::high_resolution_clock::now();
        _qp_computation_time.push_back((toc-tic).count()*1E-9);

        casadi_utils::toEigen(_conic_dict.output["x"], _dx);
        casadi_utils::toEigen(_conic_dict.output["lam_a"], _lam_a);
        casadi_utils::toEigen(_conic_dict.output["lam_x"], _lam_x);


        casadi_utils::toEigen(x0_, _sol);
        tic = std::chrono::high_resolution_clock::now();
        bool success = lineSearch(_sol, _dx, _lam_x, _lam_a, lbg, ubg, lbx, ubx, k);
        toc = std::chrono::high_resolution_clock::now();
        _line_search_time.push_back((toc-tic).count()*1E-9);
        if(success)
        {
            casadi_utils::toCasadiMatrix(_sol, x0_);
            // store trajectory
            _variable_trj[k+1] = x0_;

            // decrease regularization on success
            _eps_regularization = std::max(_eps_regularization*1e-3,
                                           _eps_regularization_base);
        }
        else
        {
            _eps_regularization = std::max(_eps_regularization*1e3,
                                           1.0);

            std::cout << "Linesearch failed, increasing regularization" << std::endl;
        }

        ///BREAK CRITERIA #2
        if(fabs(_fpr.constraint_violation) <= _constraint_violation_tolerance &&
                fabs(_fpr.merit_der/_fpr.merit) <= _merit_derivative_tolerance)
            break;

        /// BREAK CRITERIA #3
        if(_dx.norm() <= _solution_convergence)
            break;

    }


    _solution["x"] = x0_;
    double norm_f = _f.getOutput(0).norm();
    _solution["f"] = 0.5*norm_f*norm_f;
    _solution["g"] = casadi::norm_2(_g_dict.output[_g.name_out(0)].get_elements());

    return _solution;
}

template<class CASADI_TYPE>
bool SQPGaussNewton<CASADI_TYPE>::lineSearch(
        Eigen::VectorXd &x,
        const Eigen::VectorXd &dx,
        const Eigen::VectorXd &lam_x,
        const Eigen::VectorXd &lam_a,
        const casadi::DM &lbg,
        const casadi::DM &ubg,
        const casadi::DM &lbx,
        const casadi::DM &ubx,
        int iter)
{
    casadi_utils::toEigen(lbg, _lbg_);
    casadi_utils::toEigen(ubg, _ubg_);
    casadi_utils::toEigen(lbx, _lbx_);
    casadi_utils::toEigen(ubx, _ubx_);

    _x0_ = x;

    const double merit_safety_factor = 2.0;
    double norminf_lam_x = lam_x.lpNorm<Eigen::Infinity>();
    double norminf_lam_a = lam_a.lpNorm<Eigen::Infinity>();
    double norminf_lam = merit_safety_factor*std::max(norminf_lam_x, norminf_lam_a);

    double initial_cost = computeCost(_f);

    double cost_derr = computeCostDerivative(dx, _grad);

    casadi_utils::toEigen(_g_dict.output[_g.name_out(0)], _g_);
    double constraint_violation = computeConstraintViolation(_g_, _x0_, _lbg_, _ubg_, _lbx_, _ubx_);

    double merit_der = cost_derr - norminf_lam * constraint_violation;
    double initial_merit = initial_cost + norminf_lam*constraint_violation;

    // report initial value (only if iter == 0)
    if(iter == 0)
    {
        _fpr.iter = iter;
        _fpr.alpha = 0;
        _fpr.cost = initial_cost;
        _fpr.constraint_violation = constraint_violation;
        _fpr.merit = initial_merit;
        _fpr.step_length = _alpha * dx.norm();
        _fpr.accepted = true;
        _fpr.hxx_reg = _eps_regularization;
        _fpr.merit_der = merit_der;
        _fpr.mu_c = norminf_lam;

        if(_iter_cb)
        {
            _iter_cb(_fpr);
        }
    }


    _alpha = 1.0;
    bool accepted = false;
    while( _alpha > _alpha_min)
    {
        x = _x0_ + _alpha*dx;
        eval(_f, 0, x, false);
        double candidate_cost = computeCost(_f);

        casadi_utils::toCasadiMatrix(x, _x_);
        _g_dict.input[_g.name_in(0)] = _x_;
        eval(_g, _g_dict);
        casadi_utils::toEigen(_g_dict.output[_g.name_out(0)], _g_);
        double candidate_constraint_violation = computeConstraintViolation(_g_, x, _lbg_, _ubg_, _lbx_, _ubx_);

        double candidate_merit = candidate_cost + norminf_lam*candidate_constraint_violation;

        // evaluate Armijo's condition
        accepted = candidate_merit < (initial_merit + _beta*_alpha*merit_der);

        _fpr.iter = iter;
        _fpr.alpha = _alpha;
        _fpr.cost = candidate_cost;
        _fpr.constraint_violation = candidate_constraint_violation;
        _fpr.merit = candidate_merit;
        _fpr.step_length = _alpha * dx.norm();
        _fpr.accepted = accepted;
        _fpr.hxx_reg = _eps_regularization;
        _fpr.merit_der = merit_der;
        _fpr.f_der = cost_derr;
        _fpr.mu_c = norminf_lam;

        if(_iter_cb)
        {
            _iter_cb(_fpr);
        }

        if(accepted)
            break;


        if(_use_gr)
            _alpha *= 1./GR;
        else
            _alpha *= 0.5;
    }

    if(!accepted)
    {
        x = _x0_;
        return false;
    }
    return true;
}

namespace horizon {

template class SQPGaussNewton<casadi::SX>;
template class SQPGaussNewton<casadi::MX>;

}

