#include "ilqr/ilqr.h"
#include "ilqr/casadi_function.h"
#include <casadi/core/serializer.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <casadi/core/callback.hpp>
#include <mutex>
#include <cstdint>

namespace py = pybind11;
using namespace horizon::ilqr;

namespace
{
// CasADi's SWIG and pybind11 wrappers cannot exchange C++ objects directly.
// Transfer into an already allocated SWIG Function; numerical callbacks and
// their analytical Jacobians remain entirely in C++.
class NativeCasadiCallback final : public casadi::Callback
{
public:
    struct Workspace
    {
        explicit Workspace(const horizon::ilqr::Function& f): function(f.clone()), zero(Vector::Zero(f.inputSize())) {}
        std::unique_ptr<horizon::ilqr::Function> function;
        Vector zero;
        std::mutex mutex;
    };
    NativeCasadiCallback(std::string name, std::shared_ptr<Workspace> workspace, bool derivative=false):
        _workspace(std::move(workspace)), _derivative(derivative)
    { construct(name); }
    casadi_int get_n_in() override { return _derivative ? 2 : 1; }
    casadi_int get_n_out() override { return 1; }
    casadi::Sparsity get_sparsity_in(casadi_int i) override
    { return casadi::Sparsity::dense(i==0 ? _workspace->function->inputSize() : _workspace->function->outputSize(),1); }
    casadi::Sparsity get_sparsity_out(casadi_int) override
    { return casadi::Sparsity::dense(_workspace->function->outputSize(),_derivative ? _workspace->function->inputSize() : 1); }
    bool has_eval_buffer() const override { return true; }
    int eval_buffer(const double** args, const std::vector<casadi_int>&,
                    double** results, const std::vector<casadi_int>&) const override
    {
        if(!results[0]) return 0;
        std::lock_guard<std::mutex> lock(_workspace->mutex);
        const auto& f = _workspace->function;
        Eigen::Map<const Vector> x(args[0] ? args[0] : _workspace->zero.data(), f->inputSize());
        if(_derivative)
        { Eigen::Map<Matrix> J(results[0],f->outputSize(),f->inputSize()); f->jacobian(x,J); }
        else
        { Eigen::Map<Vector> y(results[0],f->outputSize()); f->evaluate(x,y); }
        return 0;
    }
    bool has_jacobian() const override { return !_derivative; }
    casadi::Function get_jacobian(const std::string& name, const std::vector<std::string>&,
        const std::vector<std::string>&, const casadi::Dict&) const override
    {
        auto callback=std::make_shared<NativeCasadiCallback>(name,_workspace,true);
        _derivatives.push_back(callback);
        return *callback;
    }
private:
    std::shared_ptr<Workspace> _workspace;
    bool _derivative;
    mutable std::vector<std::shared_ptr<NativeCasadiCallback>> _derivatives;
};
}

void bindIlqrFunctions(py::module_& m)
{
    py::class_<NativeCasadiCallback,std::shared_ptr<NativeCasadiCallback>>(m,"_NativeCasadiCallback");
    py::register_exception<EvaluationError>(m,"EvaluationError");
    py::class_<Function,std::shared_ptr<Function>>(m,"Function")
        .def("inputSize",&Function::inputSize)
        .def("outputSize",&Function::outputSize)
        .def("hasHessian",&Function::hasHessian)
        .def("evaluate",[](Function& f,const Vector& x){Vector y(f.outputSize());f.evaluate(x,y);return y;})
        .def("jacobian",[](Function& f,const Vector& x){Matrix J(f.outputSize(),f.inputSize());f.jacobian(x,J);return J;})
        .def("valueAndJacobian",[](Function& f,const Vector& x){
            Vector y(f.outputSize());Matrix J(f.outputSize(),f.inputSize());f.valueAndJacobian(x,y,J);return py::make_tuple(y,J);});
    py::class_<CasadiFunction,Function,std::shared_ptr<CasadiFunction>>(m,"CasadiFunction")
        .def(py::init([](py::object f,bool exact){
            return std::make_shared<CasadiFunction>(casadi::Function::deserialize(f.attr("serialize")().cast<std::string>()),exact);
        }),py::arg("function"),py::arg("exact_hessian")=true);
    py::class_<Parameter>(m,"Parameter")
        .def(py::init<std::string,int>())
        .def_readonly("name",&Parameter::name).def_readonly("size",&Parameter::size);
    py::class_<FunctionManager>(m,"FunctionManager")
        .def(py::init<>())
        .def("_exportCasadi",[](const FunctionManager& fm,const std::string& name,py::object destination){
            // Require a genuine CasADi SWIG Function allocated by the caller.
            auto cls=py::module_::import("casadi").attr("Function");
            if(!py::isinstance(destination,cls)) throw std::invalid_argument("destination must be a CasADi Function");
            auto address=py::int_(destination.attr("this")).cast<std::uintptr_t>();
            if(!address) throw std::invalid_argument("null CasADi Function");
            auto callback=std::make_shared<NativeCasadiCallback>(name+"_native",
                std::make_shared<NativeCasadiCallback::Workspace>(*fm.at(name).function));
            *reinterpret_cast<casadi::Function*>(address)=*callback;
            return callback;
        })
        .def("add",[](FunctionManager& fm,const std::string& name,std::shared_ptr<Function> f,std::vector<Parameter> p){
            fm.add(name,std::move(f),std::move(p));
        },py::arg("name"),py::arg("function"),py::arg("parameters")=std::vector<Parameter>{})
        // A single MX-vector stream preserves symbol identity across the SWIG /
        // pybind11 boundary. No serialization occurs during numerical solves.
        .def("_addExpression",[](FunctionManager& fm,const std::string& name,const std::string& encoded,
                                  int nin,const std::vector<std::shared_ptr<Function>>& functions,bool exact){
            casadi::StringDeserializer stream(encoded);
            auto graph=stream.unpack_mx_vector();
            if(nin<2 || graph.size()!=std::size_t(nin+1+2*functions.size()))
                throw std::invalid_argument("invalid serialized native graph");
            SymbolicFunctionManager lowered;
            std::vector<casadi::MX> from,to;
            auto replace=[&](const casadi::MX& e){
                return from.empty()?e:casadi::MX::substitute(e,casadi::MX::vertcat(from),casadi::MX::vertcat(to));
            };
            for(std::size_t i=0;i<functions.size();++i)
            {
                auto native=lowered.addNative(name+"_native_"+std::to_string(i),functions[i]);
                auto result=native(replace(graph[nin+1+2*i]));
                from.push_back(graph[nin+2+2*i]);to.push_back(std::move(result));
            }
            lowered.addExpression(name,{graph.begin(),graph.begin()+nin},replace(graph[nin]),exact);
            const auto& spec=lowered.at(name);fm.add(name,spec.function,spec.parameters);
        });
    py::class_<StateSpace,std::shared_ptr<StateSpace>>(m,"StateSpace")
        .def("stateSize",&StateSpace::stateSize).def("tangentSize",&StateSpace::tangentSize)
        .def("neutral",[](const StateSpace& s){Vector q(s.stateSize());s.neutral(q);return q;})
        .def("integrate",[](const StateSpace& s,const Vector& q,const Vector& v){
            if(q.size()!=s.stateSize() || v.size()!=s.tangentSize())throw std::invalid_argument("state/tangent size mismatch");
            Vector out(s.stateSize());s.integrate(q,v,out);return out;
        })
        .def("difference",[](const StateSpace& s,const Vector& a,const Vector& b){
            if(a.size()!=s.stateSize() || b.size()!=s.stateSize())throw std::invalid_argument("state size mismatch");
            Vector out(s.tangentSize());s.difference(a,b,out);return out;
        });
    py::class_<EuclideanSpace,StateSpace,std::shared_ptr<EuclideanSpace>>(m,"EuclideanSpace").def(py::init<int>());
    py::class_<QuaternionSpace,StateSpace,std::shared_ptr<QuaternionSpace>>(m,"QuaternionSpace").def(py::init<>());
}
