#ifndef __HORIZON__SQP__H__
#define __HORIZON__SQP__H__

#include <casadi/casadi.hpp>
#include "wrapped_function.h"
#include <Eigen/Dense>
#include <memory>
#include <chrono>


#include "profiling.h"

#include "ilqr.h"

#define GR 1.61803398875

namespace horizon{

typedef Eigen::Ref<const Eigen::VectorXd> VecConstRef;
typedef Eigen::Ref<const Eigen::MatrixXd> MatConstRef;


template <class CASADI_TYPE> ///casadi::SX or casadi::MX
class SQPGaussNewton
{
public:

    static void setQPOasesOptionsMPC(casadi::Dict& opts)
    {
        opts["enableEqualities"] = true;
        opts["initialStatusBounds"] = "inactive";
        opts["numRefinementSteps"] = 0;
        opts["enableDriftCorrection"] = 0;
        opts["terminationTolerance"] = 10e9 * std::numeric_limits<double>::epsilon();
        opts["enableFlippingBounds"] = false;
        opts["enableNZCTests"] = false;
        opts["enableRamping"] = false;
        opts["enableRegularisation"] = true;
        opts["numRegularisationSteps"] = 2;
        opts["epsRegularisation"] = 5. * 10e3 * std::numeric_limits<double>::epsilon();
    }

    static void setQPOasesOptionsReliable(casadi::Dict& opts)
    {
        opts["enableEqualities"] = false;
        opts["numRefinementSteps"] = 2;
        opts["enableFullLITest"] = true;
        opts["epsLITests"] = 10e5 * std::numeric_limits<double>::epsilon();
        opts["maxDualJump"] = 10e8;
        opts["enableCholeskyRefactorisation"] = 1;
    }


    struct IODMDict{
        casadi::DMDict input;
        casadi::DMDict output;
    };

    /**
     * @brief SQPGaussNewton SQP method with Gauss-Newton approximaiton (Hessian = J'J)
     * @param name solver name
     * @param qp_solver name internally used with casadi::conic (check casadi documetation for conic)
     * @param f cost function as casadi::Function with single input and single output (otherwise throw)
     * @param g constraints as casadi::Function with single input and single output (otherwise throw)
     * @param opts options for SQPGaussNewton and internal conic (check casadi documetation for conic).
     * NOTE: options for SQPGaussNewton are:
     *          "max_iter": iterations used to find solution
     *          "reinitialize_qpsolver": if true the internal qp solver is initialized EVERY iteration
     *          "solution_convergence": iteration are stoped if solution does not change under this threshold
     */
    SQPGaussNewton(const std::string& name, const std::string& qp_solver,
                   const casadi::Function& f,
                   const casadi::Function& g,
                   const casadi::Dict& opts = casadi::Dict()):
        _name(name), _qp_solver(qp_solver),
        _max_iter(1000),
        _reinitialize_qp_solver(false),
        _qp_opts(opts),
        _alpha(1.), _beta(1e-4), _solution_convergence(1e-6),_alpha_min(1e-3),
        _constraint_violation_tolerance(1e-6), _merit_derivative_tolerance(1e-6), _use_gr(false),
        _fpr(0, 0, 0), ///TODO: this needs to be improved!
        _merit_eps(1e-6)
    {

        _f = f;
        _df = f.factory("df", {f.name_in()}, {"jac:" + f.name_out(0) +":" + f.name_in(0)});

        _g = g;
        _dg = _g.factory("dg", {g.name_in()}, {"jac:" + g.name_out(0) + ":" + g.name_in(0)});

        for(const auto &input : _g.name_in())
        {
            _g_dict.input[input] = casadi::DM();
            _A_dict.input[input] = casadi::DM();
        }

        parseOptions();

        _variable_trj.resize(_max_iter+1, casadi::DM(f.size1_in(0), f.size2_in(0)));

        _hessian_computation_time.reserve(_max_iter);
        _qp_computation_time.reserve(_max_iter);
        _line_search_time.reserve(_max_iter);

        _fpr.bound_violation = NAN;
        _fpr.defect_norm = NAN;
        _fpr.mu_b = NAN;
        _fpr.mu_f = NAN;
        _fpr.rho = NAN;
    }

    /**
     * @brief SQPGaussNewton SQP method with Gauss-Newton approximaiton (Hessian = J'J)
     * @param name solver name
     * @param qp_solver name internally used with casadi::conic (check casadi documetation for conic)
     * @param f RESIDUAL of cost function
     * @param g constraints
     * @param x variables
     * @param opts options for SQPGaussNewton and internal conic (check casadi documetation for conic).
     * NOTE: options for SQPGaussNewton are:
     *                                          "max_iter": iterations used to find solution
     *                                          "reinitialize_qpsolver": if true the internal qp solver is initialized EVERY iteration
     */
    SQPGaussNewton(const std::string& name, const std::string& qp_solver,
                   const CASADI_TYPE& f, const CASADI_TYPE& g, const CASADI_TYPE& x, const casadi::Dict& opts = casadi::Dict()):
        _name(name), _qp_solver(qp_solver),
        _max_iter(1000),
        _reinitialize_qp_solver(false),
        _qp_opts(opts),
        _alpha(1.), _beta(1e-4), _solution_convergence(1e-6), _alpha_min(1e-3),
        _constraint_violation_tolerance(1e-6), _merit_derivative_tolerance(1e-6), _use_gr(false),
        _fpr(0, 0, 0), ///TODO: this needs to be improved!
        _merit_eps(1e-6)
    {
        _f = casadi::Function("f", {x}, {f}, {"x"}, {"f"});
        _df = _f.function().factory("df", {"x"}, {"jac:f:x"});


        _g = casadi::Function("g",{x}, {g}, {"x"}, {"g"});
        _dg = _g.factory("dg", {"x"}, {"jac:g:x"});

        parseOptions();

        _variable_trj.resize(_max_iter+1, casadi::DM(x.rows(), x.columns()));

        _hessian_computation_time.reserve(_max_iter);
        _qp_computation_time.reserve(_max_iter);
        _line_search_time.reserve(_max_iter);
    }

    void parseOptions()
    {
        if(_qp_opts.count("beta"))
        {
            _beta = _qp_opts.at("beta");
            _qp_opts.erase("beta");
        }

        if(_qp_opts.count("eps_regularization"))
        {
            _eps_regularization = _qp_opts.at("eps_regularization");
            _eps_regularization_base = _eps_regularization;
            _qp_opts.erase("eps_regularization");
        }

        if(_qp_opts.count("alpha_min"))
        {
            _alpha_min = _qp_opts.at("alpha_min");
            _qp_opts.erase("alpha_min");
        }

        if(_qp_opts.count("max_iter"))
        {
            _max_iter = _qp_opts.at("max_iter");
            _qp_opts.erase("max_iter");
        }

        if(_qp_opts.count("reinitialize_qpsolver"))
        {
            _reinitialize_qp_solver = _qp_opts.at("reinitialize_qpsolver");
            _qp_opts.erase("reinitialize_qpsolver");
        }

        if(_qp_opts.count("constraint_violation_tolerance"))
        {
            _constraint_violation_tolerance = _qp_opts.at("constraint_violation_tolerance");
            _qp_opts.erase("constraint_violation_tolerance");
        }

        if(_qp_opts.count("merit_derivative_tolerance"))
        {
            _merit_derivative_tolerance = _qp_opts.at("merit_derivative_tolerance");
            _qp_opts.erase("merit_derivative_tolerance");
        }

        if(_qp_opts.count("solution_convergence"))
        {
            _solution_convergence = _qp_opts.at("solution_convergence");
            _qp_opts.erase("solution_convergence");
        }

        if(_qp_opts.count("use_golden_ratio_update"))
        {
            _use_gr = _qp_opts.at("use_golden_ratio_update");
            _qp_opts.erase("use_golden_ratio_update");
        }

        if(_qp_opts.count("merit_eps"))
        {
            _merit_eps = _qp_opts.at("merit_eps");
            _qp_opts.erase("merit_eps");
        }

    }

    void printConicOptions(std::ostream &stream=casadi::uout()) const
    {
        if(_conic)
            _conic->print_options(stream);
    }

    /**
     * @brief setAlphaMin set the minumi allowed alpha during linesearch
     * @param alpha min in Newton's method step
     */
    void setAlphaMin(const double alpha_min)
    {
        _alpha_min = alpha_min;
    }

    const double& getAlpha() const
    {
        return _alpha;
    }

    /**
     * @brief solve NLP for given max iteration. Internal qp solver is reinitialized if "reinitialize_qpsolver" option was passed
     * as true in constructor options
     * @param initial_guess_x initial guess
     * @param lbx lower variables bound
     * @param ubx upper variables bound
     * @param lbg lower constraints bound
     * @param ubg upper constraints bound
     * @return solution dictionary containing: "x" solution, "f" 0.5*norm2 cost function, "g" norm2 constraints vector
     */
    const casadi::DMDict& solve(const casadi::DM& initial_guess_x,
                                const casadi::DM& p,
                                const casadi::DM& lbx, const casadi::DM& ubx,
                                const casadi::DM& lbg, const casadi::DM& ubg);

    double computeCost(const casadi_utils::WrappedFunction& f)
    {
        return f.getOutput(0).squaredNorm();
    }

    double computeCostDerivative(const Eigen::VectorXd& dx, const Eigen::VectorXd& grad)
    {
        return dx.dot(grad);
    }

    double computeConstraintViolation(const Eigen::VectorXd& g, const Eigen::VectorXd& x,
                                      const Eigen::VectorXd& lbg, const Eigen::VectorXd& ubg,
                                      const Eigen::VectorXd& lbx, const Eigen::VectorXd& ubx)
    {
        return (lbg-g).cwiseMax(0.).lpNorm<1>() + (ubg-g).cwiseMin(0.).lpNorm<1>() +
               (lbx-x).cwiseMax(0.).lpNorm<1>() + (ubx-x).cwiseMin(0.).lpNorm<1>();
    }


    bool lineSearch(Eigen::VectorXd& x, const Eigen::VectorXd& dx, const Eigen::VectorXd& lam_x, const Eigen::VectorXd& lam_a,
                    const casadi::DM& lbg, const casadi::DM& ubg,
                    const casadi::DM& lbx, const casadi::DM& ubx, int iter);

    void f(const CASADI_TYPE& f, const CASADI_TYPE& x, bool reinitialize_qp_solver = true)
    {
        _reinitialize_qp_solver = reinitialize_qp_solver;

        _f = casadi::Function("f", {x}, {f}, {"x"}, {"f"});
        _df = _f.function().factory("df", {"x"}, {"jac:f:x"});
    }

    void g(const CASADI_TYPE& g, const CASADI_TYPE& x, bool reinitialize_qp_solver = true)
    {
        _reinitialize_qp_solver = reinitialize_qp_solver;

        _g = casadi::Function("g",{x}, {g}, {"x"}, {"g"});
        _dg = _g.factory("dg", {"x"}, {"jac:g:x"});
    }

    bool f(const casadi::Function& f, bool reinitialize_qp_solver = true)
    {
        _reinitialize_qp_solver = reinitialize_qp_solver;

        if(f.n_in() != 1)
            return false;
        if(f.n_out() != 1)
            return false;

        _f = f;
        _df = f.factory("df", {f.name_in(0)}, {"jac:" + f.name_out(0) +":" + f.name_in(0)});

        return true;
    }

    bool g(const casadi::Function& g, bool reinitialize_qp_solver = true)
    {
        _reinitialize_qp_solver = reinitialize_qp_solver;

        if(g.n_in() != 1)
            return false;
        if(g.n_out() != 1)
            return false;

        _g = g;
        _dg = g.factory("dg", {g.name_in(0)}, {"jac:" + g.name_out(0) + ":" + g.name_in(0)});

        return true;
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
     * @brief getNumberOfIterations
     * @return number of iteration to solve NLP
     */
    unsigned int getNumberOfIterations()
    {
        return _iteration_to_solve;
    }

    /**
     * @brief getObjectiveIterations
     * @return 0.5*norm2 of objective (one per iteration)
     */
    const std::vector<double>& getObjectiveIterations()
    {
        Eigen::VectorXd tmp;
        _objective.clear();
        _objective.reserve(_iteration_to_solve);
        for(unsigned int k = 0; k < _iteration_to_solve; ++k)
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
        _constraints_norm.reserve(_iteration_to_solve);
        for(unsigned int k = 0; k < _iteration_to_solve; ++k)
        {
            _g_dict.input[_g.name_in(0)] = _variable_trj[k];
            _g.call(_g_dict.input, _g_dict.output);
            _constraints_norm.push_back(casadi::norm_2(_g_dict.output[_g.name_out(0)].get_elements()));
        }
        return _constraints_norm;
    }

    /**
     * @brief getHessianComputationTime
     * @return vector of times needed to compute hessian (one value per iteration)
     */
    const std::vector<double>& getHessianComputationTime() const
    {
        return _hessian_computation_time;
    }

    /**
     * @brief getQPComputationTime
     * @return vector of times needed to solve qp (one value per iteration)
     */
    const std::vector<double>& getQPComputationTime() const
    {
        return _qp_computation_time;
    }

    const std::vector<double>& getLineSearchComputationTime() const
    {
        return _line_search_time;
    }

    void setBeta(const double beta)
    {
        _beta = beta;
    }

    double getBeta()
    {
        return _beta;
    }

    typedef std::function<bool(const IterativeLQR::ForwardPassResult& res)> CallbackType;
    void setIterationCallback(const CallbackType& cb)
    {
        _iter_cb = cb;
    }



private:

    /**
     * @brief eval to evaluate WrappedFunctions on point x
     * @param wf WrappedFunction to evaluate
     * @param i which input to set
     * @param x point
     * @param sparse if result will be sparse
     */
    void eval(casadi_utils::WrappedFunction& wf, const int i, const Eigen::VectorXd& x, const bool sparse)
    {
        wf.setInput(i, x); // cost function
        wf.call(sparse);
    }

    /**
     * @brief eval to evaluate casadi Function on input map
     * @param cf casadi Function to evaluate
     * @param dict input/output dict
     * @param input_map
     */
    void eval(casadi::Function& cf, IODMDict& dict)
    {
        cf.call(dict.input, dict.output);
    }


    bool checkIsStationary(const Eigen::VectorXd& grad, const double tol)
    {
        for(unsigned int i = 0; i < grad.size(); ++i)
        {
            if (fabs(grad[i]) > tol)
            {
                    std::cout<<"grad[i]: "<<grad[i]<<std::endl;
                    return false;
            }
        }
        return true;
    }




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

    casadi::Dict _qp_opts;

    casadi::DMVector _variable_trj;
    std::vector<double> _objective, _constraints_norm;

    Eigen::SparseMatrix<double> _J;
    Eigen::SparseMatrix<double> _H;
    Eigen::SparseMatrix<double> _I;
    Eigen::VectorXd _grad;
    casadi::DM grad_;
    casadi::DM g_;
    casadi::DM A_;
    casadi_utils::WrappedSparseMatrix<double> H_;
    casadi::DM x0_;
    Eigen::VectorXd _sol, _dx, _lam_a, _lam_x;

    IODMDict _g_dict;
    IODMDict _A_dict;

    double _alpha, _alpha_min;

    std::vector<double> _hessian_computation_time;
    std::vector<double> _qp_computation_time;
    std::vector<double> _line_search_time;


    unsigned int _iteration_to_solve;

    IterativeLQR::ForwardPassResult _fpr;
    CallbackType _iter_cb;

    double _beta;

    double _solution_convergence;
    double _constraint_violation_tolerance;
    double _merit_derivative_tolerance;

    double _eps_regularization = 0.0;
    double _eps_regularization_base = 0.0;

    bool _use_gr;

    //line search
    Eigen::VectorXd _lbg_, _ubg_, _lbx_, _ubx_;
    Eigen::VectorXd _x0_;
    casadi::DM _x_;
    Eigen::VectorXd _g_;
    double _merit_eps;


};

}

#endif
