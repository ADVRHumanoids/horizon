#include "spot_problem.h"
#include "spot_model.h"
#include "spot_contact_projection.h"
#include "ilqr/casadi_function.h"
#include "ilqr_test_access.h"
#include <pinocchio/spatial/explog.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdlib>

namespace spot_test
{
using Clock=std::chrono::steady_clock;
namespace
{
double milliseconds(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
casadi::MX slice(const casadi::MX& x,int begin,int end) {return x(casadi::Slice(begin,end));}
}

struct Problem::Impl
{
    Formulation form;
    int N,threads,nu;
    double dt,setup_ms;
    bool project_contacts;
    std::shared_ptr<const Robot> robot;
    SymbolicFunctionManager manager;
    std::unique_ptr<horizon::IterativeLQR> solver;
    std::shared_ptr<const Function> dynamics,stage;
    Matrix reference,xguess,uguess,acceleration,force,torque;

    Impl(Formulation form,int N,int threads,ProblemOptions diagnostics):form(form),N(N),threads(threads),nu(controlSize(form)),dt(2.0/N),project_contacts(diagnostics.project_contacts)
    {
        if(N<12 || threads<0) throw std::invalid_argument("Spot needs at least twelve intervals and nonnegative threads");
        auto start=Clock::now();robot=std::make_shared<Robot>(HORIZON_SPOT_URDF);
        stage=makeStage(robot,form);
        auto native=manager.addNative("robot_terms",stage);
        using casadi::MX;
        auto x=MX::sym("x",nx),u=MX::sym("u",nu),ref=MX::sym("base_reference",7);
        auto terms=native(MX::vertcat({x,u}));
        // Reuse the same native robot call in dynamics, costs and constraints.
        // The step evaluator only integrates the recovered acceleration; all
        // robot physics and Jacobians are supplied by the shared native stage.
        auto step=manager.addNative("state_step",makeDynamics(robot,Formulation::AccelerationForce,dt,project_contacts));
        manager.addExpression("dynamics",{x,u},step(MX::vertcat({x,slice(terms,18,36),MX::zeros(nf,1)})),false);
        dynamics=manager.at("dynamics").function;
        auto qv=slice(x,3,6),rv=slice(ref,3,6);
        auto relative=ref(6)*qv-x(6)*rv-MX::cross(rv,qv);
        auto pose=MX::vertcat({400*(slice(x,0,3)-slice(ref,0,3)),200*relative});
        auto anchors=MX::zeros(nf,1);auto support=MX::zeros(nf,1);
        for(int i=0;i<nf;++i) {anchors(i)=robot->anchors(i);support(i)=robot->support(i);}
        // The physical running objective is identical for both controls.
        auto residual=MX::vertcat({pose,.02*slice(x,nq,nx),.01*slice(terms,18,36),
            .0005*(slice(terms,36,48)-support),.001*slice(terms,48,60),
            200*(slice(terms,60,72)-anchors),.1*slice(terms,72,84)});
        manager.addResidual("running",{x,u,ref},residual);
        manager.addResidual("terminal",{x,u,ref},MX::vertcat({pose,200*(slice(terms,60,72)-anchors)}));
        // The projected dynamics enforce J(q)*v=0 exactly, and the twelve
        // joint columns of J are invertible on this leg branch. Thus six zero
        // base velocities imply all eighteen zero velocities at a feasible
        // terminal state. Avoid an algebraically redundant endpoint equation.
        manager.addExpression("terminal_velocity",{x,u},slice(x,nq,
            project_contacts || diagnostics.terminal_velocity==ProblemOptions::Base ? nq+6 : nx),false);
        if(form==Formulation::AccelerationForce)
            manager.addExpression("newton_euler_contact",{x,u},
                MX::vertcat({slice(terms,0,6)/robot->weight,slice(terms,6,18)/10}),false);

        horizon::IterativeLQR::OptionDict options{{"ilqr.n_threads",threads},
            {"ilqr.constraint_violation_threshold",1e-7},{"ilqr.defect_norm_threshold",1e-7},
            {"ilqr.merit_der_threshold",1e-8},{"ilqr.step_length_threshold",1e-8},
            {"ilqr.svd_threshold",diagnostics.rank_threshold},{"ilqr.constr_decomp_type",std::string("svd")},
            {"ilqr.kkt_decomp_type",diagnostics.kkt_decomposition},
            {"ilqr.huu_reg",1e-8/(diagnostics.scale_regularization?N:1)},
            {"ilqr.hxx_reg_base",1e-8/(diagnostics.scale_regularization?N:1)}};
        solver=std::make_unique<horizon::IterativeLQR>(manager,"dynamics",N,options,robot->state);
        std::vector<int> nodes;for(int k=0;k<N;++k) nodes.push_back(k);
        solver->setResidual(nodes,manager,"running");solver->setResidual({N},manager,"terminal");
        if(diagnostics.terminal_velocity!=ProblemOptions::None)
            solver->setConstraint({N},manager,"terminal_velocity");
        if(form==Formulation::AccelerationForce) solver->setConstraint(nodes,manager,"newton_euler_contact");
        reference.resize(7,N+1);
        for(int k=0;k<=N;++k)
        {
            double s=std::pow(std::sin(std::acos(-1.0)*k/N),2);
            reference.col(k)=robot->standing.head(7);
            reference(0,k)+=.005*s;reference(1,k)-=.004*s;reference(2,k)-=.06*s;
            Eigen::Vector3d rotation(.025*s,-.02*s,.015*s);
            Eigen::Quaterniond quat=Eigen::Quaterniond::Identity();
            if(rotation.norm()>0) quat=Eigen::Quaterniond(Eigen::AngleAxisd(rotation.norm(),rotation.normalized()));
            reference.col(k).tail(4)=quat.coeffs();
        }
        solver->setParameterValue("base_reference",reference);
        Matrix lb=Matrix::Constant(nx,N+1,-std::numeric_limits<double>::infinity()),ub=-lb;
        Vector initial(nx);initial<<robot->standing,Vector::Zero(nv);lb.col(0)=ub.col(0)=initial;
        solver->setStateBounds(lb,ub);
        xguess=initial.replicate(1,N+1);uguess=Matrix::Zero(nu,N);
        Engine engine(robot);Vector physical(nv+3*nf);
        engine.evaluate(robot->standing,Vector::Zero(nv),Vector::Zero(nv),robot->support,physical);
        for(int k=0;k<N;++k)
            if(form==Formulation::Torque) uguess.col(k)=physical.segment(6,12);
            else uguess.col(k).tail(nf)=robot->support;
        setup_ms=milliseconds(start);
    }
};

Problem::Problem(Formulation f,int N,int threads,ProblemOptions options):impl(std::make_unique<Impl>(f,N,threads,options)) {}
Problem::~Problem()=default;
void Problem::diagnoseLqStep(std::ostream& out,int iterations)
{
    impl->solver->setStateInitialGuess(impl->xguess);impl->solver->setInputInitialGuess(impl->uguess);
    if(iterations) impl->solver->solve(iterations);
    horizon::IterativeLQRTestAccess::diagnose(*impl->solver,out);
}
const Matrix& Problem::states() const {return impl->solver->getStateTrajectory();}
const Matrix& Problem::accelerations() const {return impl->acceleration;}
const Matrix& Problem::forces() const {return impl->force;}
const Matrix& Problem::torques() const {return impl->torque;}
std::map<std::string,std::vector<double>> Problem::phaseTimings() const {return impl->solver->getProfilingInfo().timings;}

Metrics Problem::solve(int max_iterations)
{
    auto& p=*impl;Metrics m;m.setup_ms=p.setup_ms;
    p.solver->setStateInitialGuess(p.xguess);p.solver->setInputInitialGuess(p.uguess);
    auto running=p.manager.at("running").function->clone(),terminal=p.manager.at("terminal").function->clone();
    Vector z(nx+p.nu+7),value(running->outputSize());
    for(int k=0;k<=p.N;++k)
    {
        z<<p.xguess.col(k),p.uguess.col(std::min(k,p.N-1)),p.reference.col(k);
        if(k<p.N) {running->evaluate(z,value);m.initial_objective+=.5*value.squaredNorm()/p.N;}
        else {Vector end(terminal->outputSize());terminal->evaluate(z,end);m.initial_objective+=.5*end.squaredNorm();}
    }
    int iterations=0;p.solver->setIterationCallback([&](const auto& r) {
        iterations=std::max(iterations,r.iter+1);
        if(std::getenv("HORIZON_SPOT_DEBUG")) r.print();
        return true;
    });
    auto start=Clock::now();m.converged=p.solver->solve(max_iterations);m.solve_ms=milliseconds(start);m.iterations=iterations;
    p.solver->setIterationCallback({});
    Engine engine(p.robot);Vector stage_z(nx+p.nu),terms(84),af(nv+nf),physical(nv+3*nf),prediction(nx),gap(2*nv);
    auto eval=p.stage->clone();p.acceleration.resize(nv,p.N);p.force.resize(nf,p.N);p.torque.resize(12,p.N);
    ContactProjection contact_projection(p.robot);
    for(int k=0;k<=p.N;++k)
    {
        auto x=p.solver->state(k);auto u=p.solver->input(std::min(k,p.N-1));
        Eigen::Quaterniond actual(x(6),x(3),x(4),x(5));
        auto ref=p.reference.col(k);Eigen::Quaterniond desired(ref(6),ref(3),ref(4),ref(5));
        m.pose_position=std::max(m.pose_position,(x.head(3)-ref.head(3)).norm());
        m.pose_rotation=std::max(m.pose_rotation,pin::log3(desired.toRotationMatrix().transpose()*actual.toRotationMatrix()).norm());
        m.quaternion_error=std::max(m.quaternion_error,std::abs(actual.norm()-1));
        m.depth=std::max(m.depth,p.robot->standing(2)-x(2));
        for(int j=7;j<nq;++j) m.joint_margin=std::min(m.joint_margin,std::min(x(j)-p.robot->model->lowerPositionLimit(j),p.robot->model->upperPositionLimit(j)-x(j)));
        stage_z<<x,u;eval->evaluate(stage_z,terms);
        for(int foot=0;foot<4;++foot)
        {
            m.foot_drift=std::max(m.foot_drift,(terms.segment<3>(60+3*foot)-p.robot->anchors.segment<3>(3*foot)).norm());
            m.foot_velocity=std::max(m.foot_velocity,terms.segment<3>(72+3*foot).norm());
        }
        z<<x,u,ref;
        if(k==p.N)
        {
            Vector end(terminal->outputSize());terminal->evaluate(z,end);m.objective+=.5*end.squaredNorm();
            m.endpoint_velocity=x.tail(nv).norm();continue;
        }
        running->evaluate(z,value);m.objective+=.5*value.squaredNorm()/p.N;
        p.acceleration.col(k)=terms.segment(18,nv);p.force.col(k)=terms.segment(36,nf);p.torque.col(k)=terms.segment(48,12);
        engine.evaluate(x.head(nq),x.tail(nv),p.acceleration.col(k),p.force.col(k),physical);
        m.newton_euler=std::max(m.newton_euler,physical.head(6).norm());
        m.contact_acceleration=std::max(m.contact_acceleration,physical.segment(nv,nf).cwiseAbs().maxCoeff());
        m.peak_torque=std::max(m.peak_torque,p.torque.col(k).cwiseAbs().maxCoeff());
        for(int foot=0;foot<4;++foot)
        {
            auto f=p.force.col(k).segment<3>(3*foot);m.min_normal=std::min(m.min_normal,f(2));
            m.min_friction_margin=std::min(m.min_friction_margin,.7*f(2)-f.head(2).norm());
        }
        p.robot->configuration->integrate(x.head(nq),p.dt*x.tail(nv)+.5*p.dt*p.dt*p.acceleration.col(k),prediction.head(nq));
        prediction.tail(nv)=x.tail(nv)+p.dt*p.acceleration.col(k);
        if(p.project_contacts)
        {
            Vector raw=prediction;contact_projection.evaluate(raw,prediction);
        }
        p.robot->state->difference(prediction,p.solver->state(k+1),gap);m.dynamics_gap=std::max(m.dynamics_gap,gap.norm());
    }
    return m;
}

double Problem::derivativeError(bool dynamics)
{
    auto& p=*impl;auto function=(dynamics?p.dynamics:p.stage)->clone();
    auto space=std::make_shared<ProductSpace>(std::vector<std::shared_ptr<const StateSpace>>{
        p.robot->configuration,std::make_shared<EuclideanSpace>(nv+p.nu)});
    Vector z(nx+p.nu);z<<p.xguess.col(0),p.uguess.col(0);
    Vector d=Vector::LinSpaced(nv,-.03,.04);p.robot->configuration->integrate(p.robot->standing,d,z.head(nq));
    z.segment(nq,nv)=Vector::LinSpaced(nv,-.02,.03);
    if(p.form==Formulation::Torque) z.tail(12)+=Vector::LinSpaced(12,-.3,.2);
    else {z.segment(nx,nv)=Vector::LinSpaced(nv,-.1,.2);z.tail(nf)+=Vector::LinSpaced(nf,-.3,.2);}
    Matrix jac(function->outputSize(),z.size()),R(z.size(),space->tangentSize());
    Vector joint_value(function->outputSize()),separate_value(function->outputSize());
    function->valueAndJacobian(z,joint_value,jac);function->evaluate(z,separate_value);
    double value_error=(joint_value-separate_value).cwiseAbs().maxCoeff()/(1+separate_value.cwiseAbs().maxCoeff());
    space->retractionJacobian(z,R);Matrix analytical=jac*R,numerical(analytical.rows(),analytical.cols());
    for(int j=0;j<space->tangentSize();++j)
    {
        Vector step=Vector::Zero(space->tangentSize()),plus(z.size()),minus(z.size()),yp(function->outputSize()),ym(function->outputSize());step(j)=1e-6;
        space->integrate(z,step,plus);space->integrate(z,-step,minus);function->evaluate(plus,yp);function->evaluate(minus,ym);
        numerical.col(j)=(yp-ym)/2e-6;
    }
    Eigen::Index row,col;
    Matrix relative=((analytical-numerical).array().abs()/(1+numerical.array().abs())).matrix();
    double error=relative.maxCoeff(&row,&col);
    if(error>1e-6)
        std::cerr<<name(p.form)<<" "<<(dynamics?"dynamics":"stage")<<" derivative ("<<row<<","<<col
            <<") analytical="<<analytical(row,col)<<" numerical="<<numerical(row,col)<<"\n";
    return std::max(error,value_error);
}
double Problem::spaceDerivativeError()
{
    auto& p=*impl;Vector q(nq),w=Vector::LinSpaced(nq,-.4,.8),d=Vector::LinSpaced(nv,-.02,.03);
    p.robot->configuration->integrate(p.robot->standing,d,q);Matrix actual(nv,nv),numeric(nv,nv);
    p.robot->configuration->retractionHessian(q,w,actual);
    p.robot->configuration->StateSpace::retractionHessian(q,w,numeric);
    return (actual-numeric).cwiseAbs().maxCoeff();
}
Samples Problem::profileFunctions(int samples)
{
    if(samples<1) throw std::invalid_argument("positive profiling samples required");
    auto& p=*impl;Samples result;Vector z(nx+p.nu);z<<p.xguess.col(0),p.uguess.col(0);
    for(auto entry:std::vector<std::pair<std::shared_ptr<const Function>,std::vector<double>*>>{{p.dynamics,&result.dynamics_us},{p.stage,&result.stage_us}})
    {
        auto fn=entry.first->clone();Vector value(fn->outputSize());Matrix jac(fn->outputSize(),fn->inputSize());
        for(int i=0;i<10;++i) fn->valueAndJacobian(z,value,jac);
        for(int i=0;i<samples;++i)
        {
            auto start=Clock::now();fn->valueAndJacobian(z,value,jac);
            entry.second->push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count());
        }
    }
    return result;
}
}
