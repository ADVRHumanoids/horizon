#include "ilqr_impl.h"
#include "codegen_function.h"
#include <cxxabi.h>
#include <cstdlib>


utils::Timer::TocCallback on_timer_toc = [](const char*, double){};

template<class V>
std::type_info const& var_type(V const& v){
  return std::visit( [](auto&&x)->decltype(auto){ return typeid(x); }, v );
}

template <typename T>
T value_or(const IterativeLQR::OptionDict& opt, std::string key, T dfl)
{
    auto it = opt.find(key);

    if(it == opt.end())
    {
        return dfl;
    }

    try
    {
        return std::get<T>(it->second);
    }
    catch(std::bad_variant_access& e)
    {
        throw std::runtime_error(
                    std::string("bad type '") +
                    abi::__cxa_demangle(var_type(it->second).name(), 0, 0, 0) +
                    "' for parameter '" + key + "': expected " +
                    abi::__cxa_demangle(typeid(T).name(), 0, 0, 0)
                    );
    }
}

void set_param_inputs(std::shared_ptr<std::map<std::string, Eigen::MatrixXd>> params,
                      int k,
                      casadi_utils::WrappedFunction& f)

{
    for(int i = 2; i < f.function().n_in(); i++)
    {
        VecConstRef p_i_k = params->at(f.function().name_in(i)).col(k);
        THROW_NAN(p_i_k);
        f.setInput(i, p_i_k);
#ifdef HORIZON_VERBOSE
        std::cout << "setting param " << f.function().name_in(i) <<
                     " to function " << f.function().name() <<
                     " at time " << k <<
                     " with value = " << p_i_k.transpose() << "\n";
#endif
    }
}

IterativeLQR::IterativeLQR(cs::Function fdyn,
                           int N,
                           OptionDict opt):
    _nx(fdyn.size1_in(0)),
    _nu(fdyn.size1_in(1)),
    _N(N),
    _step_length(1.0),
    _hxx_reg(0.0),
    _hxx_reg_growth_factor(1e3),
    _line_search_accept_ratio(1e-4),
    _alpha_min(1e-3),
    _svd_threshold(1e-6),
    _constraint_violation_threshold(1e-6),
    _defect_norm_threshold(1e-6),
    _merit_der_threshold(1e-3),
    _step_length_threshold(1e-9),
    _cost(N+1, IntermediateCost(_nx, _nu)),
    _constraint(N+1, Constraint(_nx, _nu)),
    _value(N+1, ValueFunction(_nx)),
    _dyn(N, Dynamics(_nx, _nu)),
    _bp_res(N, BackwardPassResult(_nx, _nu)),
    _constraint_to_go(std::make_unique<ConstraintToGo>(_nx, _nu)),
    _fp_res(std::make_unique<ForwardPassResult>(_nx, _nu, _N)),
    _lam_g(_N+1),
    _tmp(_N),
    _fp_accepted(0)
{
    // set options _hxx_reg_base
    _verbose = value_or(opt, "ilqr.verbose", 0);
    _log = value_or(opt, "ilqr.log", 0);
    _rti = value_or(opt, "ilqr.rti", 0);
    _step_length = value_or(opt, "ilqr.step_length", 1.0);

    _rho_base = value_or(opt, "ilqr.rho_base", 0.0);
    _rho = _rho_base;
    _rho_growth_factor = value_or(opt, "ilqr.rho_growth_factor", 10.0);
    _enable_auglag = value_or(opt, "ilqr.enable_auglag", 0);

    _hxx_reg = value_or(opt, "ilqr.hxx_reg", 0.0);
    _hxx_reg_base = value_or(opt, "ilqr.hxx_reg_base", 0.0);
    _hxx_reg_growth_factor = value_or(opt, "ilqr.hxx_reg_growth_factor", 1e3);
    _huu_reg = value_or(opt, "ilqr.huu_reg", 0.0);
    _kkt_reg = value_or(opt, "ilqr.kkt_reg", 0.0);
    _line_search_accept_ratio = value_or(opt, "ilqr.line_search_accept_ratio", 1e-4);
    _alpha_min = value_or(opt, "ilqr.alpha_min", 1e-3);
    _svd_threshold = value_or(opt, "ilqr.svd_threshold", 1e-6);
    _constraint_violation_threshold = value_or(opt, "ilqr.constraint_violation_threshold", 1e-6);
    _defect_norm_threshold = value_or(opt, "ilqr.defect_norm_threshold", 1e-6);
    _merit_der_threshold = value_or(opt, "ilqr.merit_der_threshold", 1e-3);
    _step_length_threshold = value_or(opt, "ilqr.step_length_threshold", 1e-9);
    _closed_loop_forward_pass = value_or(opt, "ilqr.closed_loop_forward_pass", 1);
    _codegen_workdir = value_or<std::string>(opt, "ilqr.codegen_workdir", "/tmp");
    _codegen_enabled = value_or(opt, "ilqr.codegen_enabled", 0);
    _enable_line_search = value_or(opt, "ilqr.enable_line_search", 1);

    _it_filt.beta = value_or(opt, "ilqr.filter_beta", 0.99);
    _it_filt.gamma = value_or(opt, "ilqr.filter_gamma", 0.20);
    _use_it_filter = value_or(opt, "ilqr.use_filter", 0);

    auto decomp_type_str = value_or<std::string>(opt, "ilqr.constr_decomp_type", "qr");
    _constr_decomp_type = str_to_decomp_type(decomp_type_str);

    decomp_type_str = value_or<std::string>(opt, "ilqr.kkt_decomp_type", "lu");
    _kkt_decomp_type = str_to_decomp_type(decomp_type_str);

    int n_threads = value_or(opt, "ilqr.n_threads", 0);

    // initialize hxx from base value
    _hxx_reg = std::max(_hxx_reg_base, _hxx_reg);

    // set timer callback
    on_timer_toc = [this](const char * name, double usec)
    {
        _prof_info.timings[name].push_back(usec);
    };

    // construct param map
    _param_map = std::make_shared<ParameterMapPtr::element_type>();

    // set dynamics
    auto fdyn_jac = Dynamics::Jacobian(fdyn);

    // codegen if needed
    if(_codegen_enabled)
    {
        fdyn = utils::codegen(fdyn, _codegen_workdir);
        fdyn_jac = utils::codegen(fdyn_jac, _codegen_workdir);
    }

    for(auto& d : _dyn)
    {
        d.f = fdyn;
        d.df = fdyn_jac;
        d.param = _param_map;
    }

    // create dynamics parameters
    add_param_to_map(fdyn);

    // initialize trajectories
    _xtrj.setZero(_nx, _N+1);
    _utrj.setZero(_nu, _N);
    _lam_x.setZero(_nx, _N+1);
    _lam_bound_x.setZero(_nx, _N+1);
    _lam_bound_u.setZero(_nu, _N);

    // initialize bounds
    _x_lb.setConstant(_nx, _N+1, -inf);
    _x_ub.setConstant(_nx, _N+1, inf);
    _u_lb.setConstant(_nu, _N, -inf);
    _u_ub.setConstant(_nu, _N, inf);

    // a default cost so that it works out of the box
    //  *) default intermediate cost -> l(x, u) = eps*|u|^2
    //  *) default final cost        -> lf(x)   = eps*|xf|^2
    set_default_cost();

    // add auglag cost
    for(int i = 0; i < _N+1; i++)
    {
        int ui = std::min(i, _N-1);
        auto al = std::make_shared<BoundAuglagCostEntity>(
                    _N,
                    _x_lb.col(i), _x_ub.col(i),
                    _u_lb.col(ui), _u_ub.col(ui));

        al->setRho(_rho);

        _auglag_cost.push_back(al);

        _cost[i].addCost(al);

    }

    // worker threads
    if(n_threads > 0)
    {
        init_thread_pool(n_threads);
    }
}

void IterativeLQR::setStateBounds(const Eigen::MatrixXd& lb, const Eigen::MatrixXd& ub)
{
    if(_x_lb.rows() != lb.rows() || _x_lb.cols() != lb.cols() ||
        _x_ub.rows() != ub.rows() || _x_ub.cols() != ub.cols()
        )
    {
        throw std::invalid_argument("state bound size mismatch");
    }

    TIC(setStateBounds);

    _x_lb = lb;
    _x_ub = ub;

}

void IterativeLQR::setInputBounds(const Eigen::MatrixXd& lb, const Eigen::MatrixXd& ub)
{
    if(_u_lb.rows() != lb.rows() || _u_lb.cols() != lb.cols() ||
        _u_ub.rows() != ub.rows() || _u_ub.cols() != ub.cols()
        )
    {
        throw std::invalid_argument("input bound size mismatch");
    }

    _u_lb = lb;
    _u_ub = ub;
}

void IterativeLQR::setCost(std::vector<int> indices, const casadi::Function& inter_cost)
{
    // add parameters to param_map
    add_param_to_map(inter_cost);

    // create cost entity
    auto c = std::make_shared<IntermediateCostEntity>();

    // add to map
    _cost_map[inter_cost.name()] = c;

    // set param map
    c->param = _param_map;

    // set indices
    c->indices = indices;

    // set cost and derivatives
    auto cost = inter_cost;
    auto grad = IntermediateCostEntity::Gradient(inter_cost);
    auto hess = IntermediateCostEntity::Hessian(grad);

    // codegen if required (we skip it for quadratic costs)
    if(_codegen_enabled)
    {
        cost = utils::codegen(cost, _codegen_workdir);
        grad = utils::codegen(grad, _codegen_workdir);
        hess = utils::codegen(hess, _codegen_workdir);
    }

    c->setCost(cost,
               grad,
               hess);

    if(_verbose) std::cout << "adding cost '" << inter_cost << "' at k = ";

    for(int k : indices)
    {
        if(k > _N || k < 0)
        {
            throw std::invalid_argument("wrong intermediate cost node index");
        }

        if(_verbose) std::cout << k << " ";

        _cost[k].addCost(c);
    }

    if(_verbose) std::cout << "\n";
}

void IterativeLQR::setResidual(std::vector<int> indices,
                               const casadi::Function& residual)
{

    // add parameters to param_map
    add_param_to_map(residual);

    // create cost entity
    auto c = std::make_shared<IntermediateCostEntity>();

    // add to map
    _cost_map[residual.name()] = c;

    // set param map
    c->param = _param_map;

    // set indices
    c->indices = indices;

    // set cost and derivatives
    auto res = residual;
    auto jac = IntermediateResidualEntity::Jacobian(res);

    // codegen if required (we skip it for quadratic costs)
    if(_codegen_enabled)
    {
        res = utils::codegen(res, _codegen_workdir);
        jac = utils::codegen(jac, _codegen_workdir);
    }

    // local syms to evaluate residual and jacobian
    auto x_mx = casadi::MX::sym("x", _nx);
    auto u_mx = casadi::MX::sym("u", _nu);
    std::vector<casadi::MX> vars = {x_mx, u_mx};
    std::vector<std::string> var_names = {"x", "u"};

    for(int i = 2; i < residual.n_in(); i++)
    {
        vars.push_back(casadi::MX::sym(residual.name_in(i),
                                       residual.size1_in(i)));

        var_names.push_back(residual.name_in(i));
    }

    // compute res, Jx, Ju
    auto r = res(vars)[0];
    auto J = jac(vars);
    auto& Jx = J[0];
    auto& Ju = J[1];

    // functions
    auto mul = [](auto a, auto b)
    {
        return casadi::MX::mtimes(a, b);
    };

    auto cost_fn = casadi::Function(residual.name() + "_cost",
                                    vars,
                                    {0.5 * casadi::MX::sumsqr(r)},
                                    var_names, {"f"});

    auto grad_fn = casadi::Function(residual.name() + "_grad",
                                    vars,
                                    {mul(Jx.T(), r), mul(Ju.T(), r)},
                                    var_names,
                                    {"grad_x", "grad_u"});

    auto hess_fn = casadi::Function(residual.name() + "_gn_hess",
                                    vars,
                                    {mul(Jx.T(), Jx), mul(Ju.T(), Ju), mul(Ju.T(), Jx)},
                                    var_names,
                                    {"hess_x_x", "hess_u_u", "hess_u_x"});

    c->setCost(cost_fn, grad_fn, hess_fn);


    if(_verbose) std::cout << "adding residual '" << residual << "' at k = ";

    for(int k : indices)
    {
        if(k > _N || k < 0)
        {
            throw std::invalid_argument("wrong intermediate cost node index");
        }

        if(_verbose) std::cout << k << " ";

        _cost[k].addCost(c);
    }

    if(_verbose) std::cout << "\n";
}

void IterativeLQR::setFinalCost(const casadi::Function &final_cost)
{
    setCost(std::vector<int>{_N}, final_cost);
}

void IterativeLQR::setConstraint(std::vector<int> indices,
                                 const casadi::Function &inter_constraint,
                                 std::vector<Eigen::VectorXd> target_values)
{
    // add parameters to param_map
    add_param_to_map(inter_constraint);

    // create constr object
    auto c = std::make_shared<ConstraintEntity>();

    // add to map
    _constr_map[inter_constraint.name()] = c;

    // set param map
    c->param = _param_map;

    // set indices
    c->indices = indices;

    auto ic_fn = inter_constraint;
    auto ic_jac = ConstraintEntity::Jacobian(inter_constraint);

    if(_codegen_enabled)
    {
        ic_fn = utils::codegen(ic_fn, _codegen_workdir);
        ic_jac = utils::codegen(ic_jac, _codegen_workdir);
    }

    c->setConstraint(ic_fn,
                     ic_jac);

    if(_verbose) std::cout << "adding constraint '" << inter_constraint << "' at k = ";

    for(size_t i = 0; i < indices.size(); i++)
    {
        const int k = indices[i];

        if(k > _N || k < 0)
        {
            throw std::invalid_argument("wrong intermediate constraint node index");
        }

        if(target_values.size() > 0)
        {
            c->setTargetValue(target_values[i]);
        }

        if(_verbose) std::cout << k << " ";

        _constraint[k].addConstraint(c);
    }

    if(_verbose) std::cout << "\n";
}

void IterativeLQR::setFinalConstraint(const casadi::Function &final_constraint)
{
    setConstraint({_N}, final_constraint);
}

void IterativeLQR::setIndices(const std::string& f_name, const std::vector<int>& indices)
{
    if(_log)
    {
        std::cout << "setting new indices to " << f_name << ": \n";
        for(auto i : indices)
        {
            std::cout << i << " ";
        }
        std::cout << "\n";
    }

    auto cost_it = _cost_map.find(f_name);

    if(cost_it != _cost_map.end())
    {
        cost_it->second->indices = indices;
        return;
    }

    auto constr_it = _constr_map.find(f_name);

    if(constr_it != _constr_map.end())
    {
        constr_it->second->indices = indices;
        return;
    }

    throw std::runtime_error("...");

}

void IterativeLQR::updateIndices()
{
    // clear costs
    for(auto& c : _cost)
    {
        c.clear();
    }

    // set costs with updated indices
    for(auto& item : _cost_map)
    {
        for(int i : item.second->indices)
        {
            _cost[i].addCost(item.second);
        }
    }

    // add auglag
    for(int i = 0; i < _N + 1; i++)
    {
        _cost[i].addCost(_auglag_cost[i]);
    }

    // clear constraints;
    for(auto& c : _constraint)
    {
        c.clear();
    }

    // set constraints with updated indices
    for(auto& item : _constr_map)
    {
        for(int i : item.second->indices)
        {
            _constraint[i].addConstraint(item.second);
        }
    }
}

void IterativeLQR::setParameterValue(const std::string& pname, const Eigen::MatrixXd& value)
{
    auto it = _param_map->find(pname);

    if(it == _param_map->end())
    {
        std::cout << "undefined parameter name '" << pname << "' \n";
        return;
    }

    if(it->second.rows() != value.rows() ||
            it->second.cols() != value.cols())
    {
        std::ostringstream oss;

        oss << "parameter '" << pname << "' has wrong size: expected "
            << it->second.rows() << "x" << it->second.cols() << " != " 
            << value.rows() << "x" << value.cols();

        throw std::invalid_argument(oss.str());
    }

    it->second = value;
}

void IterativeLQR::setInitialState(const Eigen::VectorXd &x0)
{
    if(x0.size() != _nx)
    {
        throw std::invalid_argument("wrong initial state length");
    }

    _xtrj.col(0) = x0;
}

void IterativeLQR::setStateInitialGuess(const Eigen::MatrixXd& x0)
{
    if(x0.rows() != _xtrj.rows())
    {
        throw std::invalid_argument("wrong initial guess rows");
    }

    if(x0.cols() != _xtrj.cols())
    {
        throw std::invalid_argument("wrong initial guess cols");
    }

    _xtrj = x0;
}

void IterativeLQR::setInputInitialGuess(const Eigen::MatrixXd &u0)
{
    if(u0.rows() != _utrj.rows())
    {
        throw std::invalid_argument("wrong initial guess rows");
    }

    if(u0.cols() != _utrj.cols())
    {
        throw std::invalid_argument("wrong initial guess cols");
    }

    _utrj = u0;
}

void IterativeLQR::setIterationCallback(const CallbackType &cb)
{
    _iter_cb = cb;
}

const Eigen::MatrixXd &IterativeLQR::getStateTrajectory() const
{
    return _xtrj;
}

const Eigen::MatrixXd &IterativeLQR::getInputTrajectory() const
{
    return _utrj;
}

const utils::ProfilingInfo& IterativeLQR::getProfilingInfo() const
{
    return _prof_info;
}

const std::vector<IterativeLQR::ForwardPassResult>& IterativeLQR::getIterationHistory() const
{
    return _fp_res_history;
}

bool IterativeLQR::solve(int max_iter)
{
    // set cost value and constraint violation *before* the forward pass
    _fp_res->cost = compute_cost(_xtrj, _utrj);
    _fp_res->constraint_violation = compute_constr(_xtrj, _utrj);
    _fp_res->defect_norm = compute_defect(_xtrj, _utrj);
    _fp_res->bound_violation = compute_bound_penalty(_xtrj, _utrj);

    // clear filter
    _it_filt.clear();

    // reset counters
    _fp_accepted = 0;

    // solve
    for(int i = 0; i < max_iter; i++)
    {
        TIC(solve);

        _fp_res->iter = i;

        // compute linear-quadratic mdoel
        linearize_quadratize();

        // solve kkt
        backward_pass();

        // if line search failed, go directly
        // to next iteration
        if(!line_search(i))
        {
            continue;
        }

        // if the auglag approximation needs to be
        // updated, skip termination conditions
        if(auglag_update())
        {
            continue;
        }

        // if termination criteria fulfilled,
        // exit and report success
        if(should_stop())
        {
            return true;
        }
    }

    std::cout << "max iteration reached \n";

    return false;
}

void IterativeLQR::linearize_quadratize()
{
    TIC(linearize_quadratize);

    if(_th_pool.size() > 0)
    {
        linearize_quadratize_mt();
        return;
    }

    for(int i = 0; i < _N; i++)
    {
        TIC(linearize_quadratize_inner)
        linearize_quadratize_inner(i);
    }

    // handle final cost and constraint
    // note: these are only function of the state!
    _cost.back().quadratize(state(_N), input(_N-1), _N); // note: input not used here!
    _constraint.back().linearize(state(_N), input(_N-1), _N); // note: input not used here!
}

void IterativeLQR::linearize_quadratize_inner(int i)
{

    auto xi = state(i);
    auto ui = input(i);
    auto xnext = state(i+1);

    _dyn[i].linearize(xi, ui, i);
    _dyn[i].computeDefect(xi, ui, xnext, i, _dyn[i].d);
    _constraint[i].linearize(xi, ui, i);
    _cost[i].quadratize(xi, ui, i);

    THROW_NAN(_dyn[i].A());
    THROW_NAN(_dyn[i].B());
    THROW_NAN(_dyn[i].d);

    THROW_NAN(_constraint[i].C());
    THROW_NAN(_constraint[i].D());
    THROW_NAN(_constraint[i].h());

    THROW_NAN(_cost[i].Q());
    THROW_NAN(_cost[i].R());
    THROW_NAN(_cost[i].P());
    THROW_NAN(_cost[i].r());
    THROW_NAN(_cost[i].q());
}

void IterativeLQR::linearize_quadratize_mt()
{
    {
        // wake up workers
        std::unique_lock lg(_th_work_avail_mtx);
        _th_work_available_flag = _th_pool.size();
        _th_work_avail_cond.notify_all();
    }

    // do our part
    int istart = _th_pool.size() * (_N / _th_pool.size());

    for(int i = istart; i < _N; i++)
    {
        linearize_quadratize_inner(i);
    }

    // handle final cost and constraint
    // note: these are only function of the state!
    _cost.back().quadratize(state(_N), input(_N-1), _N); // note: input not used here!
    _constraint.back().linearize(state(_N), input(_N-1), _N); // note: input not used here!

    {
        // wait workers
        std::unique_lock lg(_th_work_done_mtx);
        _th_work_done_cond.wait(lg,
                                 [this](){ return _th_done_flag == _th_pool.size(); });
        _th_done_flag = 0;
    }

}

void IterativeLQR::linearize_quadratize_thread_main(int istart, int iend)
{
    while(!_th_exit)
    {
        {
            std::unique_lock lg1(_th_work_avail_mtx);
            _th_work_avail_cond.wait(lg1,
                                     [this](){ return _th_work_available_flag > 0; });
            _th_work_available_flag--;
        }

        for(int i = istart; i < iend; i++)
        {
            linearize_quadratize_inner(i);
        }

        std::unique_lock lg2(_th_work_done_mtx);
        _th_done_flag++;
        _th_work_done_cond.notify_all();

    }
}

void IterativeLQR::report_result(const IterativeLQR::ForwardPassResult& fpres)
{
    if(!_iter_cb)
    {
        return;
    }

    // call iteration callback
    _iter_cb(fpres);
}

void IterativeLQR::set_default_cost()
{
    const double dfl_cost_weight = 1e-160;

    auto x = cs::SX::sym("x", _nx);
    auto u = cs::SX::sym("u", _nu);

    auto l = cs::Function("dfl_cost", {x, u},
                          {dfl_cost_weight*cs::SX::sumsqr(u)},
                          {"x", "u"}, {"l"});

    auto lf = cs::Function("dfl_cost_final", {x, u}, {dfl_cost_weight*cs::SX::sumsqr(x)},
                           {"x", "u"}, {"l"});

    std::vector<int> all_indices;
    for(int i = 0; i < _N; i++)
    {
        all_indices.push_back(i);
    }

    setCost(all_indices, l);
    setFinalCost(lf);
}

bool IterativeLQR::fixed_initial_state()
{
    return _x_lb.col(0).allFinite() &&
            _x_ub.col(0).allFinite() &&
            _x_lb.col(0).isApprox(_x_ub.col(0));
}

IterativeLQR::DecompositionType IterativeLQR::str_to_decomp_type(const std::string &dt_str)
{
    if(dt_str == "ldlt")
    {
        return Ldlt;
    }
    else if(dt_str == "qr")
    {
        return Qr;
    }
    else if(dt_str == "lu")
    {
        return Lu;
    }
    else if(dt_str == "cod")
    {
        return Cod;
    }
    else if(dt_str == "svd")
    {
        return Svd;
    }
    else
    {
        throw std::invalid_argument("invalid value for option ilqr.decomp_type: select ldlt, qr, lu'");
    }
}

VecConstRef IterativeLQR::state(int i) const
{
    return _xtrj.col(i);
}

VecConstRef IterativeLQR::input(int i) const
{
    return _utrj.col(i);
}

MatConstRef IterativeLQR::gain(int i) const
{
    return _bp_res[i].Lu;
}

void IterativeLQR::init_thread_pool(int pool_size)
{
    int n_th = pool_size;

    int nodes_per_th = _N / n_th;

    _th_done_flag = 0;
    _th_work_available_flag = 0;
    _th_exit = false;

    for(int k = 0; k < pool_size; k++)
    {
        std::cout << "worker " << k << " started, will handle nodes in range [" <<
               k*nodes_per_th << ", "  <<   (k + 1)*nodes_per_th << ") \n";

        auto th_func = [this, k, nodes_per_th]()
        {
            linearize_quadratize_thread_main(k*nodes_per_th,
                                             (k + 1)*nodes_per_th);

            std::cout << "worker " << k << " exiting.. \n";
        };

        _th_pool.emplace_back(th_func);
    }
}

void IterativeLQR::add_param_to_map(const casadi::Function& f)
{
    // add parameters from this function
    for(int i = 2; i < f.n_in(); i++)
    {
        const int param_size = f.size1_in(i);

        // check if already exists
        if(_param_map->count(f.name()))
        {
            continue;
        }

        // add to map
        (*_param_map)[f.name_in(i)].setConstant(param_size,
                                         _N+1,
                                         std::numeric_limits<double>::quiet_NaN()
                                         );

        if(_verbose) std::cout << "adding parameter '" << f.name_in(i) << "', " <<
                     "size = " << param_size << "\n";
    }
}

const Eigen::MatrixXd &IterativeLQR::Dynamics::A() const
{
    return df.getOutput(0);
}

const Eigen::MatrixXd &IterativeLQR::Dynamics::B() const
{
    return df.getOutput(1);
}

IterativeLQR::Dynamics::Dynamics(int nx, int)
{
    d.setZero(nx);
}

Eigen::Ref<const Eigen::VectorXd> IterativeLQR::Dynamics::integrate(VecConstRef x,
                                                                    VecConstRef u,
                                                                    int k)
{
    TIC(integrate_dynamics_inner);

    f.setInput(0, x);
    f.setInput(1, u);
    set_param_inputs(param, k, f);
    f.call();
    return f.getOutput(0);
}

void IterativeLQR::Dynamics::linearize(VecConstRef x,
                                       VecConstRef u,
                                       int k)
{
    TIC(linearize_dynamics_inner);

    df.setInput(0, x);
    df.setInput(1, u);
    set_param_inputs(param, k, df);
    df.call();
}

void IterativeLQR::Dynamics::computeDefect(VecConstRef x,
                                           VecConstRef u,
                                           VecConstRef xnext,
                                           int k,
                                           Eigen::VectorXd& _d)
{
    TIC(compute_defect_inner)

    auto xint = integrate(x, u, k);
    _d = xint - xnext;
}

void IterativeLQR::Dynamics::setDynamics(casadi::Function _f)
{
    f = _f;
    df = _f.factory("df", _f.name_in(), {"jac:f:x", "jac:f:u"});
}

casadi::Function IterativeLQR::Dynamics::Jacobian(const casadi::Function &f)
{
    auto df = f.factory("df", f.name_in(), {"jac:f:x", "jac:f:u"});
    return df;
}

IterativeLQR::BoundAuglagCostEntity::BoundAuglagCostEntity(int N,
                                                           VecConstRef xlb,
                                                           VecConstRef xub,
                                                           VecConstRef ulb,
                                                           VecConstRef uub):
    _xlb(xlb), _xub(xub), _ulb(ulb), _uub(uub), _N(N)
{
    _r.setZero(ulb.size());
    _q.setZero(xlb.size());
    _xlam.setZero(xlb.size());
    _ulam.setZero(ulb.size());
}

void IterativeLQR::BoundAuglagCostEntity::setRho(double rho)
{
    _rho = rho;
}

double IterativeLQR::BoundAuglagCostEntity::evaluate(VecConstRef x,
                                                     VecConstRef u,
                                                     int k)
{
    double value = 0.0;

    // positive if ub violated, negtive if lb violated
    _x_violation = (x - _xub).cwiseMax(0) + (x - _xlb).cwiseMin(0);
    value += 0.5 * _rho * _x_violation.squaredNorm();
    value += _xlam.dot(_x_violation);

    if(k < _N)
    {
        _u_violation = (u - _uub).cwiseMax(0) + (u - _ulb).cwiseMin(0);
        value += 0.5 * _rho * _u_violation.squaredNorm();
        value += _ulam.dot(_u_violation);
    }

    return value;
}

void IterativeLQR::BoundAuglagCostEntity::quadratize(VecConstRef x,
                                                     VecConstRef u,
                                                     int k,
                                                     Eigen::MatrixXd& Q,
                                                     Eigen::MatrixXd& R,
                                                     Eigen::MatrixXd& P)
{
    evaluate(x, u, k);

    _q = _xlam + _rho * _x_violation;

    for(int i = 0; i < x.size(); i++)
    {
        Q(i, i) += _x_violation(i) != 0.0 ? _rho : 0.0;
    }

    if(k < _N)
    {
        _r = _ulam + _rho * _u_violation;

        for(int i = 0; i < u.size(); i++)
        {
            R(i, i) += _u_violation(i) != 0.0 ? _rho : 0.0;
        }
    }
}

void IterativeLQR::BoundAuglagCostEntity::update_lam(VecConstRef x, VecConstRef u, int k)
{
    evaluate(x, u, k);

    _xlam += _rho * _x_violation;

    if(k < _N)
    {
        _ulam += _rho * _u_violation;

//        std::cout << "u_err[" << k << "] = " << _u_violation.transpose().format(3) << ",  ";
//        std::cout << "u_lam[" << k << "] = " << _ulam.transpose().format(3) << "\n";
    }
}

VecConstRef IterativeLQR::BoundAuglagCostEntity::getStateMultiplier() const
{
    return _xlam;
}

VecConstRef IterativeLQR::BoundAuglagCostEntity::getInputMultiplier() const
{
    return _ulam;
}

void IterativeLQR::IntermediateCostEntity::setCost(casadi::Function _l,
                                                   casadi::Function _dl,
                                                   casadi::Function _ddl)
{
    l = _l;
    dl = _dl;
    ddl = _ddl;
}

Eigen::Ref<const Eigen::VectorXd> IterativeLQR::IntermediateCostEntity::q() const
{
    return dl.getOutput(0).col(0);
}

Eigen::Ref<const Eigen::VectorXd> IterativeLQR::IntermediateCostEntity::r() const
{
    return dl.getOutput(1).col(0);
}

double IterativeLQR::IntermediateCostEntity::evaluate(VecConstRef x,
                                                      VecConstRef u,
                                                      int k)
{
    // compute cost value
    l.setInput(0, x);
    l.setInput(1, u);
    set_param_inputs(param, k, l);
    l.call();

    return l.getOutput(0).value();
}

void IterativeLQR::IntermediateCostEntity::quadratize(VecConstRef x,
                                                      VecConstRef u,
                                                      int k,
                                                      Eigen::MatrixXd& Q,
                                                      Eigen::MatrixXd& R,
                                                      Eigen::MatrixXd& P)
{
#ifdef HORIZON_PROFILING
    horizon::utils::Timer tm("quadratize_" + l.function().name() + "_inner",
                             on_timer_toc);
#endif

    // compute cost gradient
    dl.setInput(0, x);
    dl.setInput(1, u);
    set_param_inputs(param, k, dl);
    dl.call();

    // compute cost hessian
    ddl.setInput(0, x);
    ddl.setInput(1, u);
    set_param_inputs(param, k, ddl);

    std::vector<Eigen::Ref<Eigen::MatrixXd>> out = {Q, R, P};
    ddl.call_accumulate(out);

}

casadi::Function IterativeLQR::IntermediateCostEntity::Gradient(const casadi::Function& f)
{
    return f.factory(f.name() + "_grad",
                     f.name_in(),
                     {"grad:l:x", "grad:l:u"});
}

casadi::Function IterativeLQR::IntermediateCostEntity::Hessian(const casadi::Function& df)
{
    return df.factory(df.name() + "_hess",
                      df.name_in(),
                      {"jac:grad_l_x:x", "jac:grad_l_u:u", "jac:grad_l_u:x"});
}

void IterativeLQR::IntermediateResidualEntity::setResidual(casadi::Function _res,
                                                           casadi::Function _dres)
{
    res = _res;
    dres = _dres;

    int nx = _res.size1_in(0);
    int nu = _res.size1_in(1);

    _q.setZero(nx);
    _r.setZero(nu);
}

double IterativeLQR::IntermediateResidualEntity::evaluate(VecConstRef x,
                                                          VecConstRef u,
                                                          int k)
{
    res.setInput(0, x);
    res.setInput(1, u);
    set_param_inputs(param, k, res);
    res.call();

    return res.getOutput(0).squaredNorm();
}

void IterativeLQR::IntermediateResidualEntity::quadratize(VecConstRef x,
                                                          VecConstRef u,
                                                          int k,
                                                          Eigen::MatrixXd& Q,
                                                          Eigen::MatrixXd& R,
                                                          Eigen::MatrixXd& P)
{
#ifdef HORIZON_PROFILING
    horizon::utils::Timer tm("quadratize_" + res.function().name() + "_inner",
                             on_timer_toc);
#endif

    evaluate(x, u, k);

    // compute residual jacobian
    dres.setInput(0, x);
    dres.setInput(1, u);
    set_param_inputs(param, k, dres);
    dres.call();

    // dres -> Jx, Ju
    // q = 2*Jx'*res
    // r = 2*Ju'*res
    // Q = 2*Jx'*Jx
    // R = 2*Ju'*Ju
    // P = 2*Ju'*Jx

    // |r(x, u)|^2
    // Jx'*r

    const auto& Jx = dres.getOutput(0);
    const auto& Ju = dres.getOutput(1);
    const auto& r = res.getOutput(0);

    _q.noalias() = 2.0 * Jx.transpose()*r;
    _r.noalias() = 2.0 * Ju.transpose()*r;
    Q.noalias() += 2.0 * Jx.transpose()*Jx;
    R.noalias() += 2.0 * Ju.transpose()*Ju;
    P.noalias() += 2.0 * Ju.transpose()*Jx;
}

casadi::Function IterativeLQR::IntermediateResidualEntity::Jacobian(const casadi::Function &f)
{
    return f.factory(f.name() + "_jac", f.name_in(), {"jac:res:x", "jac:res:u"});
}

const Eigen::MatrixXd& IterativeLQR::IntermediateCost::Q() const
{
    return _Q;
}

const Eigen::MatrixXd& IterativeLQR::IntermediateCost::R() const
{
    return _R;
}

const Eigen::MatrixXd& IterativeLQR::IntermediateCost::P() const
{
    return _P;
}

VecConstRef IterativeLQR::IntermediateCost::q() const
{
    return _q;
}

VecConstRef IterativeLQR::IntermediateCost::r() const
{
    return _r;
}

IterativeLQR::IntermediateCost::IntermediateCost(int nx, int nu)
{
    _Q.setZero(nx, nx);
    _R.setZero(nu, nu);
    _P.setZero(nu, nx);
    _q.setZero(nx);
    _r.setZero(nu);
}

double IterativeLQR::IntermediateCost::evaluate(VecConstRef x,
                                                VecConstRef u,
                                                int k)
{
    TIC(evaluate_cost_inner);

    double cost = 0.0;

    for(auto& it : items)
    {
        cost += it->evaluate(x, u, k);
    }

    return cost;
}

void IterativeLQR::IntermediateCost::quadratize(VecConstRef x,
                                                VecConstRef u,
                                                int k)
{
    TIC(quadratize_inner)

    const int nx = _Q.rows();
    const int nu = _R.rows();

    _Q.setZero(nx, nx);
    _R.setZero(nu, nu);
    _P.setZero(nu, nx);
    _q.setZero(nx);
    _r.setZero(nu);

    // todo: can optimize the first sum

    for(auto& it : items)
    {
        it->quadratize(x, u, k, _Q, _R, _P);
        _q += it->q();
        _r += it->r();
    }
}

void IterativeLQR::IntermediateCost::clear()
{
    items.clear();
}

void IterativeLQR::IntermediateCost::addCost(CostEntityBase::Ptr cost)
{
    items.emplace_back(cost);
}


IterativeLQR::ValueFunction::ValueFunction(int nx)
{
    S.setZero(nx, nx);
    s.setZero(nx);
}

IterativeLQR::BackwardPassResult::BackwardPassResult(int nx, int nu)
{
    Lu.setZero(nu, nx);
    lu.setZero(nu);
}

IterativeLQR::ForwardPassResult::ForwardPassResult(int nx, int nu, int N):
    hxx_reg(0),
    alpha(0),
    accepted(false)
{
    xtrj.setZero(nx, N+1);
    utrj.setZero(nu, N);
    merit = 0.0;
    step_length = 0.0;
    constraint_values.setZero(N+1);
    defect_values.setZero(nx, N);

    mu_b = 0;
}

void IterativeLQR::ForwardPassResult::print(int N) const
{
    printf("%2.2d al=%.1e  reg=%.1e  rho=%.1e  m=%.3e  f=%.3e  df=%.3e  mu_f=%.1e  mu_c=%.1e  mu_b=%.1e  du=%.3e  con=%.3e  bound=%.3e  gap=%.3e \n",
           iter, alpha, hxx_reg, rho, merit/N, cost/N, f_der/N, mu_f, mu_c, mu_b, step_length, constraint_violation/N, bound_violation, defect_norm/N);
}

IterativeLQR::ConstraintToGo::ConstraintToGo(int nx, int nu):
    _dim(0)
{
    const int c_max = nx*10;  // todo: better estimate
    _C.setZero(c_max, nx);
    _h.setZero(c_max);
    _D.setZero(c_max, nu);
}

void IterativeLQR::ConstraintToGo::set(MatConstRef C, VecConstRef h)
{
    _dim = h.size();
    _C.topRows(_dim) = C;
    _h.head(_dim) = h;
    _D.array().topRows(_dim) = 0;  // note! we assume no input dependence here!
}

void IterativeLQR::ConstraintToGo::add(MatConstRef C, MatConstRef D, VecConstRef h)
{
    const int constr_size = h.size();

    if(_dim + constr_size >= _h.size())
    {
        throw std::runtime_error("maximum constraint-to-go dimension "
            "exceeded: try reducing the svd_threshold parameter");
    }

    _C.middleRows(_dim, constr_size) = C;
    _D.middleRows(_dim, constr_size) = D;
    _h.segment(_dim, constr_size) = h;
    _dim += constr_size;
}

void IterativeLQR::ConstraintToGo::add(MatConstRef C, VecConstRef h)
{
    const int constr_size = h.size();

    if(_dim + constr_size >= _h.size())
    {
        throw std::runtime_error("maximum constraint-to-go dimension "
            "exceeded: try reducing the svd_threshold parameter");
    }

    _C.middleRows(_dim, constr_size) = C;
    _D.middleRows(_dim, constr_size).setZero();
    _h.segment(_dim, constr_size) = h;
    _dim += constr_size;
}

void IterativeLQR::ConstraintToGo::set(const IterativeLQR::Constraint &constr)
{
    if(!constr.is_valid())
    {
        clear();
        return;
    }

    _dim = constr.h().size();
    _C.topRows(_dim) = constr.C();
    _h.head(_dim) = constr.h();
    _D.array().topRows(_dim) = 0;  // note! we assume no input dependence here!
}

void IterativeLQR::ConstraintToGo::propagate_backwards(MatConstRef A,
                                                       MatConstRef B,
                                                       VecConstRef d)
{
    // tbd: avoid allocations

    _D.topRows(_dim) = C()*B;
    _h.head(_dim) = h() + C()*d;  // note: check sign!
    _C.topRows(_dim) = C()*A;  // note: C must be modified last!
}

void IterativeLQR::ConstraintToGo::add(const Constraint &constr)
{
    if(!constr.is_valid())
    {
        return;
    }

    add(constr.C(), constr.D(), constr.h());
}

void IterativeLQR::ConstraintToGo::clear()
{
    _dim = 0;
}

int IterativeLQR::ConstraintToGo::dim() const
{
    return _dim;
}

Eigen::Ref<const Eigen::MatrixXd> IterativeLQR::ConstraintToGo::C() const
{
    return _C.topRows(_dim);
}

MatConstRef IterativeLQR::ConstraintToGo::D() const
{
    return _D.topRows(_dim);
}

Eigen::Ref<const Eigen::VectorXd> IterativeLQR::ConstraintToGo::h() const
{
    return _h.head(_dim);
}

const Eigen::MatrixXd &IterativeLQR::ConstraintEntity::C() const
{
    return df.getOutput(0);
}

const Eigen::MatrixXd &IterativeLQR::ConstraintEntity::D() const
{
    return df.getOutput(1);
}

Eigen::Ref<const Eigen::VectorXd> IterativeLQR::ConstraintEntity::h() const
{
    return _hvalue;
}

bool IterativeLQR::ConstraintEntity::is_valid() const
{
    return f.is_valid();
}

IterativeLQR::ConstraintEntity::ConstraintEntity()
{
}

void IterativeLQR::ConstraintEntity::linearize(VecConstRef x, VecConstRef u, int k)
{
    if(!is_valid())
    {
        return;
    }

#ifdef HORIZON_PROFILING
    horizon::utils::Timer tm("linearize_" + f.function().name() + "_inner",
                             on_timer_toc);
#endif

    // compute constraint value
    evaluate(x, u, k);

    // compute constraint jacobian
    df.setInput(0, x);
    df.setInput(1, u);
    set_param_inputs(param, k, df);
    df.call();
}

void IterativeLQR::ConstraintEntity::evaluate(VecConstRef x, VecConstRef u, int k)
{
    if(!is_valid())
    {
        return;
    }

    // compute constraint value
    f.setInput(0, x);
    f.setInput(1, u);
    set_param_inputs(param, k, f);
    f.call();

    // remove target value and save it
    _hvalue = f.getOutput(0).col(0) - _hdes;
}

void IterativeLQR::ConstraintEntity::setConstraint(casadi::Function h)
{
    f = h;
    df = Jacobian(h);
    _hdes.setZero(f.function().size1_out(0));
}

void IterativeLQR::ConstraintEntity::setConstraint(casadi::Function h, casadi::Function dh)
{
    f = h;
    df = dh;
    _hdes.setZero(f.function().size1_out(0));
}

void IterativeLQR::ConstraintEntity::setTargetValue(const Eigen::VectorXd &hdes)
{
    if(hdes.size() != _hdes.size())
    {
        throw std::invalid_argument("target value size mismatch");
    }

    _hdes = hdes;
}

casadi::Function IterativeLQR::ConstraintEntity::Jacobian(const casadi::Function& h)
{
    return h.factory(h.name() + "_jac", h.name_in(), {"jac:h:x", "jac:h:u"});
}

const Eigen::MatrixXd &IterativeLQR::Constraint::C() const
{
    return _C;
}

const Eigen::MatrixXd &IterativeLQR::Constraint::D() const
{
    return _D;
}

VecConstRef IterativeLQR::Constraint::h() const
{
    return _h;
}

int IterativeLQR::Constraint::size() const
{
    return _C.rows();
}

bool IterativeLQR::Constraint::is_valid() const
{
    return !items.empty();
}

IterativeLQR::Constraint::Constraint(int nx, int nu)
{
    _C.setZero(0, nx);
    _D.setZero(0, nu);
}

void IterativeLQR::Constraint::linearize(VecConstRef x, VecConstRef u, int k)
{
    TIC(linearize_constraint_inner);

    int i = 0;
    for(auto& it : items)
    {
        int nc = it->C().rows();
        it->linearize(x, u, k);
        _C.middleRows(i, nc) = it->C();
        _D.middleRows(i, nc) = it->D();
        i += nc;
    }
}

void IterativeLQR::Constraint::evaluate(VecConstRef x, VecConstRef u, int k)
{
    TIC(evaluate_constraint_inner)

    int i = 0;
    for(auto& it : items)
    {
        int nc = it->C().rows();
        it->evaluate(x, u, k);
        _h.segment(i, nc) = it->h();
        i += nc;

//        std::cout << "constr[" << k << "]: " << it->f.function().name() <<
//                     "[" << nc << "] \n";
    }
}

void IterativeLQR::Constraint::addConstraint(ConstraintEntity::Ptr h)
{
    items.emplace_back(h);

    int total_size = size();
    total_size += h->f.function().size1_out(0);

    _C.setZero(total_size, _C.cols());
    _D.setZero(total_size, _D.cols());
    _h.setZero(total_size);
}

void IterativeLQR::Constraint::clear()
{
    items.clear();
    _C.setZero(0, _C.cols());
    _D.setZero(0, _D.cols());
    _h.setZero(0);
}

IterativeLQR::~IterativeLQR() = default;
