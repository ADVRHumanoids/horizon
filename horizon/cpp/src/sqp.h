#ifndef __HORIZON__SQP__H__
#define __HORIZON__SQP__H__

#include <casadi/casadi.hpp>
#include "wrapped_function.h"
#include <Eigen/Dense>
#include <memory>


#include "profiling.h"

namespace horizon{

typedef Eigen::Ref<const Eigen::VectorXd> VecConstRef;
typedef Eigen::Ref<const Eigen::MatrixXd> MatConstRef;

template <class CASADI_TYPE>
class SQPGaussNewton
{
public:
    struct IODMDict{
        casadi::DMDict input;
        casadi::DMDict output;
    };


    SQPGaussNewton(const std::string& name, const std::string& qp_solver,
                   const CASADI_TYPE& f, const CASADI_TYPE& g, const CASADI_TYPE& x, const casadi::Dict& opts = casadi::Dict()):
        _name(name), _qp_solver(qp_solver),
        _x(x),
        _max_iter(1000),
        _reinitialize_qp_solver(false),
        _opts(opts), _qp_opts(opts)
    {
        _f = casadi::Function("f", {_x}, {f}, {"x"}, {"f"});
        _df = _f.function().factory("df", {"x"}, {"jac:f:x"});


        _g = casadi::Function("g",{_x}, {g}, {"x"}, {"g"});
        _dg = _g.factory("dg", {"x"}, {"jac:g:x"});

        if(opts.contains("max_iter"))
        {
            _max_iter = opts.at("max_iter");
            _qp_opts.erase("max_iter");
        }

        if(opts.contains("reinitialize_qpsolver"))
        {
            _reinitialize_qp_solver = opts.at("reinitialize_qpsolver");
            _qp_opts.erase("reinitialize_qpsolver");
        }


        _variable_trj.resize(_max_iter+1, casadi::DM(x.rows(), x.columns()));

    }

    void printConicOptions(std::ostream &stream=casadi::uout()) const
    {
        if(_conic)
            _conic->print_options(stream);
    }

    const casadi::DMDict& solve(const casadi::DM& initial_guess_x, const casadi::DM& lbx, const casadi::DM& ubx,
                                const casadi::DM& lbg, const casadi::DM& ubg, const double alpha=1.)
    {
        bool sparse = true;
        x0_ = initial_guess_x;
        casadi_utils::toEigen(x0_, _sol);
        _variable_trj[0] = x0_;
        for(unsigned int k = 0; k < _max_iter; ++k)
        {
            //1. Cost function is linearized around actual x0
            _f.setInput(0, _sol); // cost function
            _f.call();

            _df.setInput(0, _sol); // cost function Jacobian
            _df.call(sparse);
            _J = _df.getSparseOutput(0);

            //2. Constraints are linearized around actual x0
            _g_dict.input["x"] = x0_;
            _g.call(_g_dict.input, _g_dict.output);

            _A_dict.input["x"] = x0_;
            _dg.call(_A_dict.input, _A_dict.output);

            g_ = _g_dict.output["g"];
            A_ = _A_dict.output["jac_g_x"];

            //2. We compute Gauss-Newton Hessian approximation and gradient function
            _H.resize(_J.cols(), _J.cols());
            _H = _J.transpose()*_J; ///TODO: to optimize

            _grad = _J.transpose()*_f.getOutput(0);

            //3. Setup QP
            casadi_utils::toCasadiMatrix(_grad, grad_);

            ///TODO: Optimize using directly sparsity
            casadi_utils::toCasadiMatrix(_H.toDense(), H_);

            if(!_conic || _reinitialize_qp_solver)
            {
                _conic_init_input["h"] = H_.sparsity();
                _conic_init_input["a"] = A_.sparsity();
                _conic = std::make_unique<casadi::Function>(casadi::conic("qp_solver", _qp_solver, _conic_init_input, _qp_opts));
            }

            _conic_dict.input["h"] = H_;
            _conic_dict.input["g"] = grad_;
            _conic_dict.input["a"] = A_;
            _conic_dict.input["lba"] = lbg - g_;
            _conic_dict.input["uba"] = ubg - g_;
            _conic_dict.input["lbx"] = lbx - x0_;
            _conic_dict.input["ubx"] = ubx - x0_;
            _conic_dict.input["x0"] = x0_;

            _conic->call(_conic_dict.input, _conic_dict.output);

            //4. Take full step
            x0_ = x0_ + alpha*_conic_dict.output["x"];
            casadi_utils::toEigen(x0_, _sol);


            // store trajectory
            _variable_trj[k+1] = x0_;
        }

        _solution["x"] = x0_;
        double norm_f = _f.getOutput(0).norm();
        _solution["f"] = 0.5*norm_f*norm_f;
        _solution["g"] = casadi::norm_2(_g_dict.output["g"].get_elements());
        return _solution;
    }

    /**
     * @brief getVariableTrajectory
     * @return vector of variable solutions (one per iteration)
     */
    const casadi::DMVector& getVariableTrajectory() const
    {
        return _variable_trj;
    }

    /**
     * @brief getObjectiveIterations
     * @return 0.5*norm2 of objective (one per iteration)
     */
    const std::vector<double>& getObjectiveIterations()
    {
        Eigen::VectorXd tmp;
        _objective.clear();
        _objective.reserve(_variable_trj.size());
        for(unsigned int k = 0; k < _variable_trj.size(); ++k)
        {
            casadi_utils::toEigen(_variable_trj[k], tmp);
            _f.setInput(0, tmp); // cost function
            _f.call();
            double norm = _f.getOutput(0).norm();
            _objective.push_back(0.5*norm*norm);
        }
        return _objective;
    }

    /**
     * @brief getConstraintNormIterations
     * @return norm2 of the constraint vector (one per iteration)
     */
    const std::vector<double>& getConstraintNormIterations()
    {
        _constraints_norm.clear();
        _constraints_norm.reserve(_variable_trj.size());
        for(auto sol : _variable_trj)
        {
            _g_dict.input["x"] = sol;
            _g.call(_g_dict.input, _g_dict.output);
            _constraints_norm.push_back(casadi::norm_2(_g_dict.output["g"].get_elements()));
        }
        return _constraints_norm;
    }



private:


    CASADI_TYPE _x;
    std::string _name;
    std::string _qp_solver;

    // Cost function and Jacobian
    casadi_utils::WrappedFunction _f, _df;

    // Constraint and Jacobian
    casadi::Function _g, _dg;


    int _max_iter;
    bool _reinitialize_qp_solver;

    std::unique_ptr<casadi::Function> _conic;
    casadi::SpDict _conic_init_input;
    IODMDict _conic_dict;

    casadi::DMDict _solution;

    casadi::Dict _opts;
    casadi::Dict _qp_opts;

    casadi::DMVector _variable_trj;
    std::vector<double> _objective, _constraints_norm;

    Eigen::SparseMatrix<double> _J;
    Eigen::SparseMatrix<double> _H;
    Eigen::VectorXd _grad;
    casadi::DM grad_;
    casadi::DM g_;
    casadi::DM A_;
    casadi::DM H_;
    casadi::DM x0_;
    Eigen::VectorXd _sol;

    IODMDict _g_dict;
    IODMDict _A_dict;


};

}

#endif
