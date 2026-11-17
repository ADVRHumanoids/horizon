#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "horizon/kokkos_collision.h"
#include "ilqr/ilqr.h"
#include "ilqr/casadi_function.h"
#include <cmath>

using namespace horizon::ilqr;
using namespace horizon::collision;

namespace
{
std::shared_ptr<const Function> centers()
{
    return std::make_shared<CallbackFunction>(6,6,
        [](Input x,VectorOutput y){y=x;},[](Input,MatrixOutput j){j.setIdentity();});
}
std::shared_ptr<const SdfGrid> grid()
{
    std::vector<float> values(7*7*7);
    for(int z=0;z<7;++z)for(int y=0;y<7;++y)for(int x=0;x<7;++x)
        values[(z*7+y)*7+x]=float(-2.+.75*z+.1*std::pow(-2.+.75*x,2));
    return std::make_shared<SdfGrid>(Eigen::Vector3i(7,7,7),Eigen::Vector3d(-2,-2,-2),.75,std::move(values));
}
std::vector<std::shared_ptr<EvaluationContext>> contexts(int n)
{
    std::vector<std::shared_ptr<EvaluationContext>> result;
    for(int i=0;i<n;++i)result.push_back(std::make_shared<EvaluationContext>());
    return result;
}
}

TEST(KokkosCollision, ValuesJacobiansAndProductsAcrossNodes)
{
    auto field=grid();
    KokkosCollisionBatch batch(centers(),{.1,.15},field,{{0,1}});
    SphereDistanceFunction scalar({.1,.15},{field},{{0,1}});
    Matrix states(6,5),controls=Matrix::Zero(2,4);
    states.col(0)<<-.5,.5,0,.1,.1,.2;
    for(int k=1;k<5;++k)states.col(k)=states.col(0)+Vector::Constant(6,.02*k);
    // Coincident centers and exact upper grid boundary exercise edge cases.
    states.col(2)<<.3,.3,.4,.4,.1,.1;
    states(0,4)=2.5;
    auto ctx=contexts(5);
    batch.prepare(states,controls,ctx,true);
    Matrix direction=Matrix::Random(6,4),expected_j(3,6),actual_j(3,6),product(3,4);
    Vector expected(3),actual(3);
    for(int k=0;k<5;++k)
    {
        auto f=batch.function()->clone();f->setEvaluationContext(ctx[k]);
        f->valueAndJacobian(states.col(k),actual,actual_j);
        scalar.valueAndJacobian(states.col(k),expected,expected_j);
        EXPECT_LT((actual-expected).norm(),1e-12);
        EXPECT_LT((actual_j-expected_j).norm(),1e-11);
        f->valueAndJacobianProduct(states.col(k),direction,actual,product);
        EXPECT_LT((product-expected_j*direction).norm(),1e-11);
    }
    auto stats=batch.statistics();
    EXPECT_EQ(stats.derivative_batches,1u);EXPECT_EQ(stats.nodes,5u);
    EXPECT_EQ(stats.sdf_queries,10u);EXPECT_EQ(stats.self_queries,5u);
    // Value passes must invalidate derivatives from the preceding state/pass.
    states.array()+=.01;states(0,4)=2.5;
    batch.prepare(states,controls,ctx,false);
    auto f=batch.function()->clone();f->setEvaluationContext(ctx.back());
    f->evaluate(states.col(4),actual);scalar.evaluate(states.col(4),expected);
    EXPECT_LT((actual-expected).norm(),1e-12);
    EXPECT_THROW(f->jacobian(states.col(4),actual_j),std::logic_error);
    Vector stale=states.col(4);stale[1]+=.1;
    EXPECT_THROW(f->evaluate(stale,actual),std::logic_error);
}

TEST(KokkosCollision, FailureRecoveryResizeAndCloneIsolation)
{
    KokkosCollisionBatch batch(centers(),{.1,.15},grid(),{{0,1}});
    auto clone=batch.clone();
    Matrix a=Matrix::Constant(6,3,.1),b=Matrix::Constant(6,2,.2);
    auto ca=contexts(3),cb=contexts(2);
    batch.prepare(a,Matrix::Zero(1,2),ca,true);
    clone->prepare(b,Matrix::Zero(1,1),cb,true);
    auto f=batch.function()->clone();f->setEvaluationContext(ca[0]);
    Vector value(3),saved(3);f->evaluate(a.col(0),saved);
    // A failure at one node invalidates all results for that batch.
    Matrix failed=a;failed(0,2)=100;
    EXPECT_THROW(batch.prepare(failed,Matrix::Zero(1,2),ca,true),EvaluationError);
    EXPECT_THROW(f->evaluate(a.col(0),value),std::logic_error);
    batch.prepare(a,Matrix::Zero(1,2),ca,true);f->evaluate(a.col(0),value);
    EXPECT_EQ((value-saved).norm(),0.);
    auto other=batch.function()->clone();other->setEvaluationContext(cb[0]);
    EXPECT_NO_THROW(other->evaluate(b.col(0),value));
    // Resize the original batch while the clone's different horizon survives.
    batch.prepare(b,Matrix::Zero(1,1),cb,false);
    EXPECT_NO_THROW(other->evaluate(b.col(0),value));
    auto empty=batch.function()->clone();empty->setEvaluationContext(contexts(1)[0]);
    EXPECT_THROW(empty->evaluate(b.col(0),value),std::logic_error);
}

TEST(KokkosCollision, SelfOnlyFiniteDifferences)
{
    KokkosCollisionBatch batch(centers(),{.1,.15},nullptr,{{0,1}},.05,.01);
    Matrix states(6,3);states.col(0)<<-.05,.08,0,.02,.1,.12;
    states.col(1)=states.col(0);states.col(2)=states.col(0);
    auto ctx=contexts(3);
    Vector v(1),plus(1),minus(1);Matrix j(1,6);
    batch.prepare(states,Matrix::Zero(1,2),ctx,true);
    auto f=batch.function()->clone();f->setEvaluationContext(ctx[0]);
    f->valueAndJacobian(states.col(0),v,j);
    Matrix derivative=j;
    for(int col=0;col<6;++col)
    {
        const double original=states(col,0);
        states(col,0)=original+1e-6;batch.prepare(states,Matrix::Zero(1,2),ctx,false);f->evaluate(states.col(0),plus);
        states(col,0)=original-1e-6;batch.prepare(states,Matrix::Zero(1,2),ctx,false);f->evaluate(states.col(0),minus);
        states(col,0)=original;
        EXPECT_NEAR(derivative(0,col),(plus[0]-minus[0])/2e-6,1e-8);
    }
}

TEST(KokkosCollision, LineSearchRetriesOutOfGridTrial)
{
    using casadi::MX;
    std::vector<float> values(27);
    for(int z=0;z<3;++z)for(int y=0;y<3;++y)for(int x=0;x<3;++x)
        values[(z*3+y)*3+x]=float(.5*z);
    auto field=std::make_shared<SdfGrid>(Eigen::Vector3i(3,3,3),Eigen::Vector3d::Zero(),.5,std::move(values));
    KokkosCollisionBatch batch(centers(),{.01,.01},field);
    SymbolicFunctionManager functions;
    auto x=MX::sym("x",6),u=MX::sym("u",6);
    functions.addExpression("dyn",{x,u},x+u,false);
    auto native=functions.addNative("batch",batch.function());
    functions.addResidual("collision",{x,u},native(x));
    // The full step leaves the finite SDF; a shorter step is evaluable.
    functions.addResidual("target",{x,u},100*(x-MX::ones(6,1)*1.5));
    functions.addResidual("effort",{x,u},.01*u);
    horizon::IterativeLQR solver(functions,"dyn",2);
    solver.addEvaluationBatch(batch);
    solver.setResidual({0,1,2},functions,"collision");
    solver.setResidual({2},functions,"target");solver.setResidual({0,1},functions,"effort");
    const Vector start=Vector::Constant(6,.1);
    Matrix lb=Matrix::Constant(6,3,-10.),ub=-lb;lb.col(0)=ub.col(0)=start;
    solver.setStateBounds(lb,ub);solver.setStateInitialGuess(start.replicate(1,3));
    solver.setInputInitialGuess(Matrix::Zero(6,2));
    EXPECT_NO_THROW(solver.solve(1));
    ASSERT_FALSE(solver.getIterationHistory().empty());
    const auto& step=solver.getIterationHistory().back();
    EXPECT_GT(step.alpha,0.);EXPECT_LT(step.alpha,1.);
    EXPECT_TRUE(solver.getStateTrajectory().allFinite());
    EXPECT_LE(solver.state(2).maxCoeff(),1.);
    EXPECT_GT(solver.state(2).minCoeff(),start[0]);
}

TEST(KokkosCollision, FullSolveMatchesScalarWithNodeWorkers)
{
    using casadi::MX;
    auto field=grid();KokkosCollisionBatch batch(centers(),{.1,.15},field,{{0,1}});
    Matrix reference;
    for(int mode=0;mode<3;++mode)
    {
        SymbolicFunctionManager functions;
        auto x=MX::sym("x",6),u=MX::sym("u",6);
        std::shared_ptr<const Function> kernel=mode?batch.function():
            std::make_shared<SphereDistanceFunction>(std::vector<double>{.1,.15},
                std::vector<std::shared_ptr<const DistanceField>>{field},std::vector<SphereDistanceFunction::Pair>{{0,1}});
        auto native=functions.addNative("collision",kernel);
        functions.addExpression("dyn",{x,u},x+.1*u,false);
        functions.addResidual("collision_cost",{x,u},10*native(x));
        functions.addResidual("effort",{x,u},u);
        horizon::IterativeLQR solver(functions,"dyn",5,{{"ilqr.n_threads",mode==2?2:0}});
        solver.setResidual({1,2,3,4,5},functions,"collision_cost");
        solver.setResidual({0,1,2,3,4},functions,"effort");
        if(mode)solver.addEvaluationBatch(batch);
        Vector start(6);start<<-.1,.1,0,0,-.05,-.05;
        Matrix lb=Matrix::Constant(6,6,-1.),ub=-lb;
        lb.col(0)=ub.col(0)=start;solver.setStateBounds(lb,ub);
        solver.setStateInitialGuess(start.replicate(1,6));solver.setInputInitialGuess(Matrix::Zero(6,5));
        solver.solve(20);
        if(mode==0)reference=solver.getStateTrajectory();
        else EXPECT_LT((solver.getStateTrajectory()-reference).norm(),1e-8);
    }
}

int main(int argc,char** argv)
{
    ::testing::InitGoogleTest(&argc,argv);
    Kokkos::ScopeGuard runtime(Kokkos::InitializationSettings().set_num_threads(2));
    return RUN_ALL_TESTS();
}
