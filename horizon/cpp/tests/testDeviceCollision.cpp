#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "horizon/device_collision.h"
#include "ilqr/ilqr.h"
#include "ilqr/casadi_function.h"
#ifdef HORIZON_DEVICE_TEST_KYON

#include "horizon/pinocchio_collision.h"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#endif

using namespace horizon::ilqr;
using namespace horizon::collision;
namespace
{
DeviceRobotModel cartesianRobot()
{
    DeviceRobotModel m;m.nq=m.nv=6;m.joints.resize(7);
    for(int i=1;i<=6;++i)
    {
        auto& j=m.joints[i];j.type=DeviceRobotModel::Prismatic;
        j.parent=(i==1 || i==4)?0:i-1;j.q=j.v=i-1;
        for(int a=0;a<3;++a)j.axis[a]=(a==(i-1)%3)?1.:0.;
    }
    m.spheres={{3,{0,0,0},.1},{6,{0,0,0},.15}};return m;
}
std::shared_ptr<const SdfGrid> field()
{
    std::vector<float> values(9*9*9);
    for(int z=0;z<9;++z)for(int y=0;y<9;++y)for(int x=0;x<9;++x)
        values[(z*9+y)*9+x]=float(-4.+z+.03*(x-4)*(x-4));
    return std::make_shared<SdfGrid>(Eigen::Vector3i(9,9,9),Eigen::Vector3d(-4,-4,-4),1.,std::move(values));
}
Matrix unpack(Input h,int n)
{
    Matrix out(n,n);int k=0;
    for(int a=0;a<n;++a)for(int b=a;b<n;++b){out(a,b)=out(b,a)=h[k++];}
    return out;
}
void compare(DeviceCollisionCost& batch,MatrixInput states,Function& scalar,Function& fk,
             int nv,double weight,bool tangent=false)
{
    batch.prepare(states,true);
    Vector points(fk.outputSize()),r(scalar.outputSize());
    Matrix jc(points.size(),tangent?nv:states.rows()),j(r.size(),nv);
    for(int node=0;node<states.cols();++node)
    {
        if(tangent)dynamic_cast<TangentFunction&>(fk).valueAndTangentLinearization(states.col(node),points,jc);
        else fk.valueAndJacobian(states.col(node),points,jc);
        scalar.valueAndJacobianProduct(points,jc,r,j);
        const Vector g=weight*j.transpose()*r;const Matrix H=weight*j.transpose()*j;
        EXPECT_NEAR(batch.value(node),.5*weight*r.squaredNorm(),1e-10*(1+weight*r.squaredNorm()));
        EXPECT_LT((batch.gradient(node)-g).norm(),1e-9*(1+g.norm()));
        EXPECT_LT((unpack(batch.packedHessian(node),nv)-H).norm(),1e-9*(1+H.norm()));
    }
}
std::shared_ptr<Function> cartesianCenters()
{
    return std::make_shared<CallbackFunction>(6,6,
        [](Input x,VectorOutput y){y<<x[0],x[3],x[1],x[4],x[2],x[5];},
        [](Input,MatrixOutput j){j.setZero();for(int i=0;i<6;++i)j(i,(i%2)*3+i/2)=1.;});
}
}

TEST(DeviceCollision, CostGradientHessianAndManifoldIndependentDifferences)
{
    DeviceCollisionCost batch(cartesianRobot(),field(),{{0,1}},17.);
    SphereDistanceFunction scalar({.1,.15},{field()},{{0,1}});
    Matrix states(6,5);states.col(0)<<-.05,.02,.01,.1,.03,-.02;
    for(int k=1;k<5;++k)states.col(k)=states.col(0)+Vector::Constant(6,.02*k);
    states.col(2)<<.1,.1,.1,.1,.1,.1; // Coincident centers.
    states(0,4)=4.; // Exact upper grid boundary.
    auto fk=cartesianCenters();compare(batch,states,scalar,*fk,6,17.);
    const Vector gradient=batch.gradient(0);const double cost=batch.value(0);
    for(int a=0;a<6;++a)
    {
        Matrix changed=states;changed(a,0)+=1e-6;batch.prepare(changed,false);double plus=batch.value(0);
        changed(a,0)-=2e-6;batch.prepare(changed,false);double minus=batch.value(0);
        EXPECT_NEAR(gradient[a],(plus-minus)/2e-6,1e-7);
    }
    batch.resetStatistics();batch.prepare(states,false);
    EXPECT_NEAR(batch.value(0),cost,1e-12);
    EXPECT_THROW(batch.gradient(0),std::logic_error);
    const auto stat=batch.statistics();
    EXPECT_EQ(stat.download_bytes,5u*sizeof(double)+sizeof(int));
    EXPECT_EQ(stat.upload_bytes,5u*6*sizeof(double));
}

TEST(DeviceCollision, FailureRecoveryResizeOffsetsAndCloneIsolation)
{
    DeviceCollisionCost batch(cartesianRobot(),field(),{{0,1}},1.,.02,.005,8,1);
    Matrix states=Matrix::Constant(8,3,.1);batch.prepare(states,true);
    auto clone=batch.clone();Matrix other=Matrix::Constant(8,2,.2);clone->prepare(other,true);
    const double saved=clone->value(0);
    states(1,2)=100.;EXPECT_THROW(batch.prepare(states,true),EvaluationError);
    EXPECT_THROW(batch.value(0),std::logic_error);
    EXPECT_EQ(clone->value(0),saved);
    states(1,2)=.1;batch.prepare(states,true);EXPECT_TRUE(std::isfinite(batch.value(0)));
    batch.prepare(other,false);EXPECT_EQ(clone->value(0),saved);
    EXPECT_THROW(batch.packedHessian(0),std::logic_error);
    auto invalid=cartesianRobot();invalid.joints[2].parent=2;
    EXPECT_THROW(DeviceCollisionCost(invalid,field()),std::invalid_argument);
}

TEST(DeviceCollision, FullSolveMatchesResidualsAndAvoidsSelfCollision)
{
    using casadi::MX;Matrix reference;
    for(int mode=0;mode<3;++mode)
    {
        DeviceCollisionCost batch(cartesianRobot(),nullptr,{{0,1}},100.);
        SymbolicFunctionManager functions;
        auto x=MX::sym("x",6),u=MX::sym("u",6);
        functions.addExpression("dyn",{x,u},x+.1*u,false);
        functions.addResidual("effort",{x,u},.01*u);
        if(!mode)
        {
            auto fk=functions.addNative("fk",cartesianCenters());
            auto residual=functions.addNative("self",std::make_shared<SphereDistanceFunction>(
                std::vector<double>{.1,.15},std::vector<std::shared_ptr<const DistanceField>>{},
                std::vector<SphereDistanceFunction::Pair>{{0,1}}));
            functions.addResidual("collision",{x,u},10*residual(fk(x)));
        }
        horizon::IterativeLQR solver(functions,"dyn",8,{{"ilqr.n_threads",mode==2?2:0}});
        if(mode)solver.setBatchedCost(mode==2?std::vector<int>{8,0,7,1,6,2,5,3,4}:
                                     std::vector<int>{0,1,2,3,4,5,6,7,8},"collision",batch);
        else solver.setResidual({0,1,2,3,4,5,6,7,8},functions,"collision");
        solver.setResidual({0,1,2,3,4,5,6,7},functions,"effort");
        Vector start(6);start<<-.04,0,0,.04,0,0;
        Matrix lb=Matrix::Constant(6,9,-1.),ub=-lb;lb.col(0)=ub.col(0)=start;
        solver.setStateBounds(lb,ub);solver.setStateInitialGuess(start.replicate(1,9));
        solver.setInputInitialGuess(Matrix::Zero(6,8));solver.solve(30);
        if(!mode)reference=solver.getStateTrajectory();
        else EXPECT_LT((solver.getStateTrajectory()-reference).norm(),1e-7);
        const Vector final=solver.state(8);
        EXPECT_GT((final.head(3)-final.tail(3)).norm()-.25,.015);
    }
}

TEST(DeviceCollision, TiledPairAccumulationMatchesSingleNodeAndValuePass)
{
    auto model=cartesianRobot();model.spheres.clear();
    for(int i=0;i<128;++i)model.spheres.push_back({i%2?3:6,{.001*i,0,0},.1});
    std::vector<SphereDistanceFunction::Pair> pairs;
    for(int i=0;i<128;++i)for(int j=i+1;j<128;++j)pairs.emplace_back(i,j);
    DeviceCollisionCost batch(model,nullptr,pairs);
    Matrix state=Matrix::Constant(6,1,.1);batch.prepare(state,true);
    const double cost=batch.value(0);const Vector g=batch.gradient(0),h=batch.packedHessian(0);
    Matrix trajectory=state.replicate(1,257);batch.prepare(trajectory,true);
    // The 64 MiB limit applies to residual/Jacobian scratch. Geometry,
    // transforms, configurations and packed outputs account for another 1.1 MiB.
    EXPECT_LT(batch.statistics().device_bytes,66u*1024u*1024u);
    EXPECT_NEAR(batch.value(256),cost,1e-9);
    EXPECT_LT((batch.gradient(256)-g).norm(),1e-9);
    EXPECT_LT((batch.packedHessian(256)-h).norm(),1e-8);
    batch.prepare(trajectory,false);EXPECT_NEAR(batch.value(256),cost,1e-8);
}

TEST(DeviceCollision, LineSearchRetriesOutOfGridDirectCost)
{
    using casadi::MX;
    std::vector<float> values(27);
    for(int z=0;z<3;++z)for(int y=0;y<3;++y)for(int x=0;x<3;++x)values[(z*3+y)*3+x]=.5*z;
    auto grid=std::make_shared<SdfGrid>(Eigen::Vector3i(3,3,3),Eigen::Vector3d::Zero(),.5,std::move(values));
    DeviceCollisionCost batch(cartesianRobot(),grid);
    SymbolicFunctionManager functions;auto x=MX::sym("x",6),u=MX::sym("u",6);
    functions.addExpression("dyn",{x,u},x+u,false);
    functions.addResidual("target",{x,u},100*(x-1.5));functions.addResidual("effort",{x,u},.01*u);
    horizon::IterativeLQR solver(functions,"dyn",2);solver.setBatchedCost({0,1,2},"collision",batch);
    solver.setResidual({2},functions,"target");solver.setResidual({0,1},functions,"effort");
    const Vector start=Vector::Constant(6,.1);
    Matrix lb=Matrix::Constant(6,3,-10.),ub=-lb;lb.col(0)=ub.col(0)=start;
    solver.setStateBounds(lb,ub);solver.setStateInitialGuess(start.replicate(1,3));
    solver.setInputInitialGuess(Matrix::Zero(6,2));EXPECT_NO_THROW(solver.solve(1));
    ASSERT_FALSE(solver.getIterationHistory().empty());
    EXPECT_GT(solver.getIterationHistory().back().alpha,0.);
    EXPECT_LT(solver.getIterationHistory().back().alpha,1.);
    EXPECT_LE(solver.state(2).maxCoeff(),1.);
}

TEST(DeviceCollision, ActiveNodesExcludeOutOfGridStatesAndPreserveNodeMapping)
{
    using casadi::MX;
    DeviceCollisionCost batch(cartesianRobot(),field());
    SymbolicFunctionManager functions;auto x=MX::sym("x",6),u=MX::sym("u",6);
    functions.addExpression("dyn",{x,u},x+u,false);
    functions.addResidual("effort",{x,u},u);
    horizon::IterativeLQR solver(functions,"dyn",2);
    // Only terminal cost: the fixed initial state is outside the collision SDF.
    solver.setBatchedCost({2},"collision",batch);solver.setResidual({0,1},functions,"effort");
    Vector start=Vector::Constant(6,5.);
    Matrix lb=Matrix::Constant(6,3,-10.),ub=-lb;lb.col(0)=ub.col(0)=start;
    solver.setStateBounds(lb,ub);Matrix guess=Matrix::Constant(6,3,.1);guess.col(0)=start;
    solver.setStateInitialGuess(guess);solver.setInputInitialGuess(Matrix::Zero(6,2));
    EXPECT_NO_THROW(solver.solve(1));
    const auto stat=batch.statistics();EXPECT_GT(stat.nodes,0u);
    EXPECT_EQ(stat.nodes,stat.value_batches+stat.derivative_batches);
    EXPECT_THROW(solver.setBatchedCost({2},"collision",batch),std::invalid_argument);
}

#ifdef HORIZON_DEVICE_TEST_KYON
namespace
{
class Configuration final : public StateSpace
{
public:
    explicit Configuration(std::shared_ptr<const pinocchio::Model> m):model(m){}
    int stateSize() const override{return model->nq;}
    int tangentSize() const override{return model->nv;}
    void neutral(VectorOutput q) const override{q=pinocchio::neutral(*model);}
    void integrate(Input q,Input v,VectorOutput out) const override{out=pinocchio::integrate(*model,q,v);}
    void difference(Input a,Input b,VectorOutput out) const override{out=pinocchio::difference(*model,a,b);}
    void retractionJacobian(Input q,MatrixOutput j) const override
    {j.setZero();pinocchio::integrateCoeffWiseJacobian(*model,q,j);}
    std::shared_ptr<const pinocchio::Model> model;
};
}
TEST(DeviceCollision, ArbitraryAxesContinuousJointsAndFixedFramePlacements)
{
    namespace pin=pinocchio;
    auto model=std::make_shared<pin::Model>();
    const auto base=model->addJoint(0,pin::JointModelFreeFlyer(),pin::SE3::Identity(),"base");
    const auto turn=model->addJoint(base,pin::JointModelRevoluteUnaligned(Eigen::Vector3d(1,2,3).normalized()),
        pin::SE3(Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitY()).toRotationMatrix(),Eigen::Vector3d(.2,.1,-.1)),"turn");
    const auto slide=model->addJoint(turn,pin::JointModelPrismaticUnaligned(Eigen::Vector3d(-2,1,1).normalized()),
        pin::SE3::Identity(),"slide");
    const auto continuous=model->addJoint(base,pin::JointModelRUBX(),pin::SE3::Identity(),"continuous");
    model->addFrame(pin::Frame("tip",slide,0,pin::SE3(Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitZ()).toRotationMatrix(),
        Eigen::Vector3d(.1,.2,.3)),pin::OP_FRAME));
    model->addFrame(pin::Frame("other",continuous,0,pin::SE3::Identity(),pin::OP_FRAME));
    SphereModel geometry({{"tip",Eigen::Vector3d(.2,.1,0),.15},{"other",Eigen::Vector3d(0,.3,.1),.1}});
    auto space=std::make_shared<Configuration>(model);PinocchioSphereCenters fk(model,space,geometry);
    DeviceCollisionCost batch(exportDeviceRobot(*model,geometry),field(),{{0,1}},7.,.5);
    SphereDistanceFunction scalar({.15,.1},{field()},{{0,1}},.5);
    Matrix states(model->nq,5);const Vector neutral=pin::neutral(*model);
    for(int k=0;k<5;++k)states.col(k)=pin::integrate(*model,neutral,Vector(.3*Vector::Random(model->nv)));
    compare(batch,states,scalar,fk,model->nv,7.,true);
    states.block(3,0,4,1)*=2.;EXPECT_THROW(batch.prepare(states,true),EvaluationError);
    EXPECT_THROW(batch.value(0),std::logic_error);
}
TEST(DeviceCollision, KyonFloatingBaseDerivativesAndSelfAvoidance)
{
    auto model=std::make_shared<pinocchio::Model>();pinocchio::urdf::buildModel("data/kyon_full_spheres.urdf",*model);
    auto space=std::make_shared<Configuration>(model);SphereModel geometry(loadSphereModelFromUrdf("data/kyon_full_spheres.urdf").spheres());
    PinocchioSphereCenters fk(model,space,geometry);
    std::vector<double> radii;for(auto& s:geometry.spheres())radii.push_back(s.radius);
    const auto pairs=makeSelfCollisionPairs(geometry,loadDisabledCollisionPairs("data/kyon_full_disabled_collisions.json"));
    ASSERT_GT(pairs.size(), 1000);
    Matrix states(model->nq,5);
    const Vector neutral=pinocchio::neutral(*model);
    for(int i=0;i<5;++i)states.col(i)=pinocchio::integrate(*model,neutral,Vector(.25*Vector::Random(model->nv)));
    for(int mode=0;mode<3;++mode)
    {
        auto grid=mode==1?nullptr:field();auto selected=mode==0?std::vector<SphereDistanceFunction::Pair>{}:pairs;
        DeviceCollisionCost batch(exportDeviceRobot(*model,geometry),grid,selected,23.,.5,.01);
        std::vector<std::shared_ptr<const DistanceField>> fields;if(grid)fields.push_back(grid);
        SphereDistanceFunction scalar(radii,fields,selected,.5,.01);
        compare(batch,states,scalar,fk,model->nv,23.,true);
        const Vector g=batch.gradient(0);Vector delta=Vector::Zero(model->nv);
        for(int a=0;a<model->nv;++a)
        {
            Matrix changed=states;delta[a]=1e-6;
            changed.col(0)=pinocchio::integrate(*model,Vector(states.col(0)),delta);batch.prepare(changed,false);double plus=batch.value(0);
            changed.col(0)=pinocchio::integrate(*model,Vector(states.col(0)),Vector(-delta));batch.prepare(changed,false);double minus=batch.value(0);
            EXPECT_NEAR(g[a],(plus-minus)/2e-6,1e-5);delta[a]=0.;
        }
    }
    // Choose an explicitly enabled movable pair and activate its margin.
    Vector points(fk.outputSize());Matrix jc(points.size(),model->nv);
    fk.valueAndTangentLinearization(neutral,points,jc);
    int first=-1,second=-1;double best=0.;const int ns=radii.size();
    auto distance=[&](Input p,int i,int j){Eigen::Vector3d d;for(int a=0;a<3;++a)d[a]=p[a*ns+i]-p[a*ns+j];return d;};
    for(const auto& [i,j] : pairs)
    {
        const Eigen::Vector3d d=distance(points,i,j);if(d.norm()<1e-6)continue;
        Matrix relative(3,model->nv);for(int a=0;a<3;++a)relative.row(a)=jc.row(a*ns+i)-jc.row(a*ns+j);
        const double score=(d.normalized().transpose()*relative).norm();
        if(score>best){best=score;first=i;second=j;}
    }
    ASSERT_GE(first,0);const double initial=distance(points,first,second).norm()-radii[first]-radii[second];
    DeviceCollisionCost avoidance(exportDeviceRobot(*model,geometry),nullptr,{{first,second}},1.,initial+.03,.005);
    Matrix trajectory=neutral.replicate(1,1);
    for(int iteration=0;iteration<40;++iteration)
    {
        avoidance.prepare(trajectory,true);Vector step=-.5*avoidance.gradient(0);
        trajectory.col(0)=pinocchio::integrate(*model,Vector(trajectory.col(0)),step);
    }
    fk.evaluate(trajectory.col(0),points);
    EXPECT_GT(distance(points,first,second).norm()-radii[first]-radii[second],initial+.02);
}
#endif

int main(int argc,char** argv)
{
    ::testing::InitGoogleTest(&argc,argv);
    Kokkos::ScopeGuard runtime(Kokkos::InitializationSettings().set_num_threads(2));
    return RUN_ALL_TESTS();
}
