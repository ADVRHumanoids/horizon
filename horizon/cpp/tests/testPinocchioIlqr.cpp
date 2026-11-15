#include <gtest/gtest.h>
#include "ilqr/ilqr.h"
#include "ilqr/casadi_function.h"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

using namespace horizon::ilqr;
namespace pin=pinocchio;

namespace
{
class RobotSpace final : public StateSpace
{
public:
    explicit RobotSpace(std::shared_ptr<const pin::Model> model):model(std::move(model)) {}
    int stateSize() const override {return model->nq;}
    int tangentSize() const override {return model->nv;}
    void neutral(VectorOutput x) const override {x=pin::neutral(*model);}
    void integrate(Input q,Input v,VectorOutput out) const override {out=pin::integrate(*model,q,v);}
    void difference(Input from,Input to,VectorOutput out) const override {out=pin::difference(*model,from,to);}
    // The base numerical derivatives work for arbitrary user operations.
    // Use analytical overrides for performance-sensitive applications.
    std::shared_ptr<const pin::Model> model;
};

class FramePosition final : public TangentFunction
{
public:
    FramePosition(std::shared_ptr<const RobotSpace> space,pin::FrameIndex frame):
        TangentFunction(space,3),space(std::move(space)),data(*this->space->model),frame(frame),
        J(6,this->space->tangentSize()) {}
    std::unique_ptr<Function> clone() const override {return std::make_unique<FramePosition>(space,frame);}
private:
    void evaluateImpl(Input q,VectorOutput out) override
    {
        pin::forwardKinematics(*space->model,data,q);
        pin::updateFramePlacements(*space->model,data);
        out=data.oMf[frame].translation();
    }
    void tangentJacobian(Input q,MatrixOutput jac) override
    {
        pin::computeJointJacobians(*space->model,data,q);
        pin::updateFramePlacements(*space->model,data);
        J.setZero();
        pin::getFrameJacobian(*space->model,data,frame,pin::LOCAL_WORLD_ALIGNED,J);
        jac=J.topRows(3); // native nv columns; no ambient padding
    }
    std::shared_ptr<const RobotSpace> space;
    pin::Data data;
    pin::FrameIndex frame;
    Matrix J;
};

class RobotDynamics final : public TangentFunction
{
public:
    explicit RobotDynamics(std::shared_ptr<const RobotSpace> space):
        TangentFunction(std::make_shared<ProductSpace>(std::vector<std::shared_ptr<const StateSpace>>{
            space,std::make_shared<EuclideanSpace>(space->tangentSize())}),space->stateSize()),
        space(std::move(space)),prediction(this->space->stateSize()),
        R(this->space->stateSize(),this->space->tangentSize()),
        dq(this->space->tangentSize(),this->space->tangentSize()),du(dq) {}
    std::unique_ptr<Function> clone() const override {return std::make_unique<RobotDynamics>(space);}
private:
    void evaluateImpl(Input z,VectorOutput out) override
    {space->integrate(z.head(space->stateSize()),z.tail(space->tangentSize()),out);}
    void tangentJacobian(Input z,MatrixOutput jac) override
    {
        const int nq=space->stateSize(),nv=space->tangentSize();
        dq.setZero(); du.setZero();
        pin::dIntegrate(*space->model,z.head(nq),z.tail(nv),dq,pin::ARG0);
        pin::dIntegrate(*space->model,z.head(nq),z.tail(nv),du,pin::ARG1);
        evaluateImpl(z,prediction); space->retractionJacobian(prediction,R);
        jac.leftCols(nv).noalias()=R*dq;
        jac.rightCols(nv).noalias()=R*du;
    }
    std::shared_ptr<const RobotSpace> space;
    Vector prediction;
    Matrix R,dq,du;
};
}

TEST(PinocchioIlqr, NativeFkSymbolicCartesianErrorAndRobotShooting)
{
    auto model=std::make_shared<pin::Model>();
    auto base=model->addJoint(0,pin::JointModelFreeFlyer(),pin::SE3::Identity(),"base");
    auto joint=model->addJoint(base,pin::JointModelRY(),pin::SE3::Identity(),"joint");
    auto frame=model->addFrame(pin::Frame("tool",joint,0,
        pin::SE3(Eigen::Matrix3d::Identity(),Eigen::Vector3d(1,0,.2)),pin::OP_FRAME));
    auto space=std::make_shared<RobotSpace>(model);
    ASSERT_NE(model->nq,model->nv);
    auto fk_model=std::make_shared<FramePosition>(space,frame);
    SymbolicFunctionManager functions;
    auto fk=functions.addNative("fk",fk_model);
    auto q=casadi::MX::sym("q",model->nq),u=casadi::MX::sym("u",model->nv);
    auto ref=casadi::MX::sym("reference",3),error=casadi::MX::sym("error",3);
    casadi::Function cartesianError("cartesianError",{error},{10*error});
    functions.addResidual("tracking",{q,u,ref},cartesianError({fk(q)-ref}).at(0));
    functions.addResidual("effort",{q,u},u);
    functions.add("dynamics",std::make_shared<RobotDynamics>(space));

    Vector start(model->nq),goal(model->nq),v=Vector::LinSpaced(model->nv,-.15,.2),target(3);
    space->neutral(start); space->integrate(start,v,goal); fk_model->evaluate(goal,target);
    auto residual=functions.at("tracking").function->clone();
    Vector z(model->nq+model->nv+3);z<<goal,Vector::Zero(model->nv),target;
    Matrix J(3,z.size()),R(model->nq,model->nv),numerical(3,model->nv);
    residual->jacobian(z,J);space->retractionJacobian(goal,R);
    for(int k=0;k<model->nv;++k)
    {
        Vector d=Vector::Zero(model->nv),zp=z,zm=z,yp(3),ym(3); d(k)=1e-6;
        space->integrate(goal,d,zp.head(model->nq));space->integrate(goal,-d,zm.head(model->nq));
        residual->evaluate(zp,yp);residual->evaluate(zm,ym);numerical.col(k)=(yp-ym)/2e-6;
    }
    EXPECT_LT((J.leftCols(model->nq)*R-numerical).norm(),1e-6);

    const int N=3;
    horizon::IterativeLQR solver(functions,"dynamics",N,
        {{"ilqr.n_threads",2},{"ilqr.merit_der_threshold",1e-10}},space);
    solver.setResidual({0,1,2},functions,"effort");
    solver.setResidual({N},functions,"tracking");
    solver.setParameterValue("reference",target.replicate(1,N+1));
    Matrix lb=Matrix::Constant(model->nq,N+1,-std::numeric_limits<double>::infinity()),ub=-lb;
    lb.col(0)=ub.col(0)=start;lb.col(N)=ub.col(N)=goal;solver.setStateBounds(lb,ub);
    solver.solve(20);
    Vector gap(model->nv),prediction(model->nq);
    space->difference(goal,solver.state(N),gap);EXPECT_LT(gap.norm(),1e-6);
    for(int k=0;k<N;++k)
    {
        space->integrate(solver.state(k),solver.input(k),prediction);
        space->difference(prediction,solver.state(k+1),gap);EXPECT_LT(gap.norm(),1e-6);
    }
}
