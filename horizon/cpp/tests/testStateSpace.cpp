#include <gtest/gtest.h>
#include "ilqr/state_space.h"
#include <Eigen/Cholesky>

using namespace horizon::ilqr;

TEST(StateSpace, QuaternionRetractionAndDifference)
{
    QuaternionSpace s;
    Vector q(4), from(4), to(4), v(3), back(3); s.neutral(q);
    v<<.4,-.2,.6; s.integrate(q,v,from);
    v<<-.3,.6,.2; s.integrate(from,v,to); s.difference(from,to,back);
    EXPECT_LT((v-back).norm(),1e-14);
    Vector negative=-to; s.difference(from,negative,back);
    EXPECT_LT((v-back).norm(),1e-14);
    EXPECT_NEAR(to.norm(),1,1e-14);
}

TEST(StateSpace, QuaternionAnalyticalDerivatives)
{
    QuaternionSpace s;
    Vector identity(4), from(4), to(4), v(3), w(4); s.neutral(identity);
    v<<.4,-.2,.6; s.integrate(identity,v,from);
    v<<-.3,.6,.2; s.integrate(from,v,to);
    w<<.2,.3,-.5,1;
    Matrix R(4,3), H(3,3), A(3,3), B(3,3), numR(4,3), numH(3,3), numA(3,3), numB(3,3);
    s.retractionJacobian(from,R); s.retractionHessian(from,w,H);
    s.differenceJacobians(from,to,A,B);
    double eps=1e-6;
    for(int i=0;i<3;++i)
    {
        Vector d=Vector::Zero(3), plus(4), minus(4), gp(3), gm(3);
        d(i)=eps; s.integrate(from,d,plus); s.integrate(from,-d,minus);
        numR.col(i)=(plus-minus)/(2*eps);
        s.difference(plus,to,gp); s.difference(minus,to,gm);
        numA.col(i)=(gp-gm)/(2*eps);
        s.integrate(to,d,plus); s.integrate(to,-d,minus);
        s.difference(from,plus,gp); s.difference(from,minus,gm);
        numB.col(i)=(gp-gm)/(2*eps);
        // Scalar second derivative, including mixed directions.
        for(int j=0;j<3;++j)
        {
            Vector e=Vector::Zero(3), pp(4), pm(4), mp(4), mm(4);
            const double hs=1e-4; d.setZero(); d(i)=hs; e(j)=hs;
            s.integrate(from,d+e,pp); s.integrate(from,d-e,pm);
            s.integrate(from,-d+e,mp); s.integrate(from,-d-e,mm);
            numH(i,j)=w.dot(pp-pm-mp+mm)/(4*hs*hs);
        }
    }
    EXPECT_LT((R-numR).norm(),1e-8);
    EXPECT_LT((A-numA).norm(),1e-8); EXPECT_LT((B-numB).norm(),1e-8);
    EXPECT_LT((H-numH).norm(),1e-7);
}

TEST(StateSpace, EuclideanDefault)
{
    EuclideanSpace s(2);
    Vector x(2), dx(2), y(2), gap(2); x<<1,2; dx<<-.2,.3;
    s.integrate(x,dx,y); s.difference(x,y,gap);
    EXPECT_LT((gap-dx).norm(),1e-14);
    Matrix a(2,2),b(2,2); s.differenceJacobians(x,y,a,b);
    EXPECT_LT((a+Matrix::Identity(2,2)).norm(),1e-14);
    EXPECT_LT((b-Matrix::Identity(2,2)).norm(),1e-14);
}

TEST(StateSpace, CustomOplusOminusAndProduct)
{
    auto custom=std::make_shared<CallbackStateSpace>(Vector::Ones(1),1,
        [](Input x,Input dx,VectorOutput y){y(0)=x(0)*std::exp(dx(0));},
        [](Input a,Input b,VectorOutput d){d(0)=std::log(b(0)/a(0));});
    ProductSpace s({custom,std::make_shared<QuaternionSpace>(),std::make_shared<EuclideanSpace>(2)});
    Vector x(7),dx(6),y(7),back(6); s.neutral(x); dx<<.4,.2,-.1,.3,-.2,.7;
    s.integrate(x,dx,y); s.difference(x,y,back);
    EXPECT_LT((dx-back).norm(),1e-12);
    Matrix j(7,6),h(6,6),a(6,6),b(6,6);
    s.retractionJacobian(y,j); s.retractionHessian(y,Vector::Ones(7),h);
    s.differenceJacobians(x,y,a,b);
    EXPECT_NEAR(j(0,0),y(0),1e-8); EXPECT_NEAR(h(0,0),y(0),1e-7);
    EXPECT_NEAR(a(0,0),-1,1e-8); EXPECT_NEAR(b(0,0),1,1e-8);
    Matrix projection(6,7);s.tangentProjection(y,projection);
    Matrix reference=(j.transpose()*j).ldlt().solve(j.transpose());
    EXPECT_LT((projection-reference).norm(),1e-12);
    EXPECT_LT((projection*j-Matrix::Identity(6,6)).norm(),1e-12);
    Matrix storage=Matrix::Random(7,5),directions(6,5);s.projectDirections(y,storage,directions);
    EXPECT_LT((directions-projection*storage).norm(),1e-12);
}

TEST(StateSpace, NonorthogonalCustomProjectionFallback)
{
    Matrix R(3,2),P(2,3);R<<1,0,1,1,0,1;P<<2,1,-1,-1,1,2;P/=3;
    auto custom=std::make_shared<CallbackStateSpace>(Vector::Zero(3),2,
        [R](Input x,Input d,VectorOutput y){y=x+R*d;},
        [P](Input a,Input b,VectorOutput d){d=P*(b-a);},
        [R](Input,MatrixOutput jac){jac=R;});
    ProductSpace product({custom,std::make_shared<EuclideanSpace>(1)});
    Vector x=Vector::Zero(4);Matrix projection(3,4),expected=Matrix::Zero(3,4);
    expected.topLeftCorner(2,3)=P;expected(2,3)=1;
    product.tangentProjection(x,projection);EXPECT_LT((projection-expected).norm(),1e-14);
    Matrix seed=Matrix::Random(4,5),actual(3,5);product.projectDirections(x,seed,actual);
    EXPECT_LT((actual-expected*seed).norm(),1e-14);
}
