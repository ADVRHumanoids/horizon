#include <gtest/gtest.h>
#include "ilqr/casadi_function.h"
#include "ilqr/state_space.h"
#include <future>
#include <atomic>

using namespace horizon::ilqr;

namespace
{
class ProductOnly final : public Function
{
public:
    ProductOnly():Function(2,1) {}
    bool hasJacobianProduct() const override { return true; }
    std::unique_ptr<Function> clone() const override { return std::make_unique<ProductOnly>(); }
private:
    void evaluateImpl(Input x,VectorOutput y) override { y[0]=x.squaredNorm(); }
    void jacobianImpl(Input,MatrixOutput) override { throw std::logic_error("dense Jacobian must not be requested"); }
    void valueAndJacobianProductImpl(Input x,MatrixInput d,VectorOutput y,MatrixOutput p) override
    { evaluateImpl(x,y);p.noalias()=2*x.transpose()*d; }
};
}

TEST(IlqrFunctions, SharedCachePreservesNativeDirectionalKernel)
{
    auto context=std::make_shared<EvaluationContext>();
    auto f=context->share(std::make_shared<ProductOnly>());
    Vector x(2),v(1);x<<.3,-.2;
    Matrix directions(2,3),p(1,3);directions<<1,2,3,4,5,6;
    f->valueAndJacobianProduct(x,directions,v,p);
    EXPECT_NEAR(v[0],x.squaredNorm(),1e-14);
    EXPECT_LT((p-2*x.transpose()*directions).norm(),1e-14);
    x[0]=.6;context->invalidate();
    f->valueAndJacobianProduct(x,directions,v,p);
    EXPECT_LT((p-2*x.transpose()*directions).norm(),1e-14);
    f->evaluate(x,v);EXPECT_NEAR(v[0],x.squaredNorm(),1e-14);
}

TEST(IlqrFunctions, NativeSymbolicCompositionDerivatives)
{
    CallbackFunction inner(2, 2,
        [](Input x, VectorOutput y) { y << x(0)*x(1), x(0)*x(0) + x(1); },
        [](Input x, MatrixOutput j) { j << x(1), x(0), 2*x(0), 1; },
        [](Input, Input w, MatrixOutput h) { h << 2*w(1), w(0), w(0), 0; });
    auto y = casadi::MX::sym("y", 2);
    CasadiFunction outer(casadi::Function("outer", {y},
        {casadi::MX::vertcat({sin(y(0)) + y(1)*y(1), y(0)*y(1)})}));
    ComposedFunction f(outer, inner);
    Vector x(2), w(2), value(2); x << .4, -.7; w << .8, -.3;
    Matrix j(2,2), h(2,2), numericalJ(2,2), numericalH(2,2);
    f.evaluate(x, value); f.jacobian(x, j); f.weightedHessian(x, w, h);
    const double eps = 1e-6;
    for(int i=0; i<2; ++i)
    {
        Vector a=x, b=x, va(2), vb(2); Matrix ja(2,2), jb(2,2);
        a(i)+=eps; b(i)-=eps;
        f.evaluate(a,va); f.evaluate(b,vb);
        numericalJ.col(i)=(va-vb)/(2*eps);
        f.jacobian(a,ja); f.jacobian(b,jb);
        numericalH.col(i)=(ja-jb).transpose()*w/(2*eps);
    }
    EXPECT_LT((j-numericalJ).norm(),1e-8);
    EXPECT_LT((h-numericalH).norm(),1e-8);
    auto clone=f.clone();
    auto task=std::async(std::launch::async,[&] {
        Vector result(2);
        for(int i=0;i<100;++i) clone->evaluate(x,result);
        return result;
    });
    for(int i=0;i<100;++i) f.jacobian(x,j);
    EXPECT_LT((task.get()-value).norm(),1e-14);
}

TEST(IlqrFunctions, CasadiConcatenationAndClone)
{
    auto x=casadi::SX::sym("x",2), u=casadi::SX::sym("u"), p=casadi::SX::sym("p");
    CasadiFunction f(casadi::Function("sparse_output",{x,u,p},
        {casadi::SX::vertcat({x(0)*p,casadi::SX(0),u*x(1)})}));
    Vector z(4), value(3), w(3); z<<2,3,4,5; w<<1,2,3;
    Matrix j(3,4), h(4,4);
    f.evaluate(z,value); f.jacobian(z,j); f.weightedHessian(z,w,h);
    EXPECT_EQ(value(0),10); EXPECT_EQ(value(1),0); EXPECT_EQ(value(2),12);
    EXPECT_EQ(j(0,3),2); EXPECT_EQ(j(2,1),4); EXPECT_EQ(j(1,1),0);
    EXPECT_EQ(h(0,3),1); EXPECT_EQ(h(1,2),3);
    f.clone()->evaluate(z,value);
    EXPECT_EQ(value(0),10);
}

TEST(IlqrFunctions, ValidationAndMissingHessian)
{
    auto f=std::make_shared<CallbackFunction>(2,1,
        [](Input x,VectorOutput y){ y(0)=x.sum(); },
        [](Input,MatrixOutput j){ j.setOnes(); });
    Vector x=Vector::Ones(2), w=Vector::Ones(1), value(1); Matrix h(2,2);
    EXPECT_THROW(f->weightedHessian(x,w,h),std::logic_error);
    EXPECT_THROW(f->evaluate(w,value),std::invalid_argument);
    FunctionManager manager;
    manager.add("f",f,{{"p",1}});
    EXPECT_THROW(manager.add("f",f),std::invalid_argument);
    EXPECT_THROW(manager.add("bad",f,{{"p",1},{"p",1}}),std::invalid_argument);
    EXPECT_THROW(manager.at("missing"),std::out_of_range);
}

TEST(IlqrFunctions, SymbolicNativeCallsWithBranchesParametersAndExactHessian)
{
    SymbolicFunctionManager manager;
    auto native=std::make_shared<CallbackFunction>(2,2,
        [](Input x,VectorOutput y){y<<x(0)*x(1),x(0)*x(0)+x(1);},
        [](Input x,MatrixOutput j){j<<x(1),x(0),2*x(0),1;},
        [](Input,Input w,MatrixOutput h){h<<2*w(1),w(0),w(0),0;});
    auto fk=manager.addNative("fk",native);
    auto q=casadi::MX::sym("q",2),u=casadi::MX::sym("u"),ref=casadi::MX::sym("ref",2);
    auto a=fk(q);
    auto b=fk(a+u);
    auto residual=sin(b-ref)+a;
    auto expression=casadi::MX::sumsqr(residual)+u*u;
    manager.addCost("tracking",{q,u,ref},expression);
    auto function=manager.at("tracking").function->clone();
    function->setEvaluationContext(std::make_shared<EvaluationContext>());
    Vector z(5),w=Vector::Ones(1),value(1); z<<.3,-.4,.2,.1,.7;
    Matrix j(1,5),h(5,5),nj(1,5),nh(5,5);
    function->evaluate(z,value); function->jacobian(z,j); function->weightedHessian(z,w,h);
    const double eps=1e-6;
    for(int k=0;k<5;++k)
    {
        Vector plus=z,minus=z,vp(1),vm(1); Matrix jp(1,5),jm(1,5);
        plus(k)+=eps; minus(k)-=eps;
        function->evaluate(plus,vp); function->evaluate(minus,vm);
        nj(0,k)=(vp(0)-vm(0))/(2*eps);
        function->jacobian(plus,jp); function->jacobian(minus,jm);
        nh.col(k)=(jp-jm).transpose()/(2*eps);
    }
    EXPECT_LT((j-nj).norm(),1e-7); EXPECT_LT((h-nh).norm(),1e-7);
    EXPECT_EQ(manager.at("tracking").parameters.at(0).name,"ref");
    manager.addResidual("residual",{q,u,ref},residual);
    EXPECT_FALSE(manager.at("residual").function->hasHessian());
    auto clone=function->clone(); clone->evaluate(z,value);
    EXPECT_LT((h-h.transpose()).norm(),1e-12);
}

namespace
{
class QuaternionVector final : public TangentFunction
{
public:
    QuaternionVector():TangentFunction(std::make_shared<QuaternionSpace>(),3) {}
    std::unique_ptr<Function> clone() const override {return std::make_unique<QuaternionVector>();}
    bool hasHessian() const override {return true;}
private:
    void evaluateImpl(Input q,VectorOutput y) override {y=q.head(3);}
    void tangentJacobian(Input q,MatrixOutput j) override
    {
        QuaternionSpace space; Matrix R(4,3); space.retractionJacobian(q,R); j=R.topRows(3);
    }
    void tangentWeightedHessian(Input q,Input w,MatrixOutput h) override
    {h.setIdentity(); h*=-.25*q.head(3).dot(w);}
};
}

TEST(IlqrFunctions, NativeTangentJacobianThroughSymbolicGraph)
{
    SymbolicFunctionManager manager;
    auto fk=manager.addNative("fk",std::make_shared<QuaternionVector>());
    auto q=casadi::MX::sym("q",4),u=casadi::MX::sym("u"),ref=casadi::MX::sym("ref",3);
    auto error=fk(q)-ref;
    manager.addCost("cost",{q,u,ref},casadi::MX::sumsqr(error));
    auto function=manager.at("cost").function->clone();
    QuaternionSpace space; Vector identity(4),z=Vector::Zero(8),rotation(3),w=Vector::Ones(1);
    space.neutral(identity); rotation<<.3,-.5,.8; space.integrate(identity,rotation,z.head(4));
    z.tail(3)<<.2,-.1,.3;
    Matrix j(1,8),h(8,8),R(4,3),curvature(3,3),localH(3,3),numH(3,3);
    function->jacobian(z,j); function->weightedHessian(z,w,h);
    space.retractionJacobian(z.head(4),R);
    space.retractionHessian(z.head(4),j.leftCols(4).transpose(),curvature);
    localH=R.transpose()*h.topLeftCorner(4,4)*R+curvature;
    const double eps=1e-4;
    for(int a=0;a<3;++a)
        for(int b=0;b<3;++b)
        {
            Vector da=Vector::Zero(3),db=da; da(a)=eps; db(b)=eps;
            Vector zz=z,v(1); double values[4];
            space.integrate(z.head(4),da+db,zz.head(4));function->evaluate(zz,v);values[0]=v(0);
            space.integrate(z.head(4),da-db,zz.head(4));function->evaluate(zz,v);values[1]=v(0);
            space.integrate(z.head(4),-da+db,zz.head(4));function->evaluate(zz,v);values[2]=v(0);
            space.integrate(z.head(4),-da-db,zz.head(4));function->evaluate(zz,v);values[3]=v(0);
            numH(a,b)=(values[0]-values[1]-values[2]+values[3])/(4*eps*eps);
        }
    EXPECT_LT((localH-numH).norm(),1e-7);
}

TEST(IlqrFunctions, JointLinearizationEvaluatesSharedNativeCallOnce)
{
    auto calls=std::make_shared<std::atomic<int>>(0);
    SymbolicFunctionManager manager;
    auto fk=manager.addNative("fk",std::make_shared<CallbackFunction>(1,1,
        [calls](Input q,VectorOutput y){++*calls;y(0)=q(0)*q(0);},
        [](Input q,MatrixOutput j){j(0,0)=2*q(0);}));
    auto q=casadi::MX::sym("q"),u=casadi::MX::sym("u");auto result=fk(q);
    manager.addResidual("res",{q,u},result+sin(result)+u);
    auto f=manager.at("res").function->clone();
    Vector z(2),value(1);z<<.3,.2;Matrix jac(1,2);
    f->valueAndJacobian(z,value,jac);
    EXPECT_EQ(calls->load(),1);
    EXPECT_NEAR(jac(0,0),.6*(1+std::cos(.09)),1e-12);
}

TEST(IlqrFunctions, SeededTangentGraphMatchesManifoldFiniteDifferences)
{
    SymbolicFunctionManager manager;
    auto fk=manager.addNative("fk",std::make_shared<QuaternionVector>());
    auto q=casadi::MX::sym("q",4),u=casadi::MX::sym("u"),p=casadi::MX::sym("p",3);
    manager.addCost("cost",{q,u,p},casadi::MX::sumsqr(sin(fk(q)-p))+u*u);
    auto f=manager.at("cost").function->clone();f->setEvaluationContext(std::make_shared<EvaluationContext>());
    QuaternionSpace space;Vector identity(4),z=Vector::Zero(8),rotation(3),value(1);
    space.neutral(identity);rotation<<.2,-.3,.4;space.integrate(identity,rotation,z.head(4));z(4)=.7;z.tail(3)<<.1,.2,-.1;
    Matrix R(4,3),seed=Matrix::Zero(8,4),product(1,4),numeric(1,4),h(8,8),j(1,8);
    space.retractionJacobian(z.head(4),R);seed.topLeftCorner(4,3)=R;seed(4,3)=1;
    f->valueAndJacobianProduct(z,seed,value,product);
    for(int k=0;k<4;++k)
    {
        Vector a=z,b=z,va(1),vb(1),d=Vector::Zero(3);
        if(k<3) {d(k)=1e-6;space.integrate(z.head(4),d,a.head(4));space.integrate(z.head(4),-d,b.head(4));}
        else {a(4)+=1e-6;b(4)-=1e-6;}
        f->evaluate(a,va);f->evaluate(b,vb);numeric.col(k)=(va-vb)/2e-6;
    }
    EXPECT_LT((product-numeric).norm(),1e-8);
    // Mixing seeded, full-Jacobian, and exact-Hessian calls must retain the
    // original ambient-coordinate API and its independently sized buffers.
    f->jacobian(z,j);f->weightedHessian(z,Vector::Ones(1),h);
    EXPECT_LT((product-j*seed).norm(),1e-12);EXPECT_TRUE(h.allFinite());
    Matrix shorter=seed.leftCols(1),one(1,1);f->valueAndJacobianProduct(z,shorter,value,one);
    EXPECT_NEAR(one(0,0),product(0,0),1e-12);
}

TEST(IlqrFunctions, SharedNativeCallsAcrossGraphsInvalidateExactly)
{
    auto values=std::make_shared<std::atomic<int>>(0),jacs=std::make_shared<std::atomic<int>>(0);
    SymbolicFunctionManager manager;
    auto native=manager.addNative("kernel",std::make_shared<CallbackFunction>(2,2,
        [values](Input x,VectorOutput y){++*values;y<<x(0)*x(0)+x(1),x(0)*x(1);},
        [jacs](Input x,MatrixOutput j){++*jacs;j<<2*x(0),1,x(1),x(0);}));
    auto x=casadi::MX::sym("x"),u=casadi::MX::sym("u"),p=casadi::MX::sym("p");
    auto y=native(casadi::MX::vertcat({x,u}));
    manager.addResidual("r",{x,u,p},y+p);manager.addExpression("c",{x,u},y(0)-3*y(1),false);
    auto a=manager.at("r").function->clone(),b=manager.at("c").function->clone();
    auto context=std::make_shared<EvaluationContext>();a->setEvaluationContext(context);b->setEvaluationContext(context);
    Vector z(3),v(2),w(1);z<<.3,.2,.7;
    Matrix seed=Matrix::Zero(3,2),ja(2,2),dir(2,1),jb(1,1);seed.topRows(2).setIdentity();dir<<2,-1;
    a->valueAndJacobianProduct(z,seed,v,ja);b->valueAndJacobianProduct(z.head(2),dir,w,jb);
    EXPECT_EQ(values->load(),1);EXPECT_EQ(jacs->load(),1);
    EXPECT_NEAR(v(0),.99,1e-14);EXPECT_NEAR(w(0),.11,1e-14);EXPECT_NEAR(jb(0,0),-.1,1e-14);
    z(2)+=.1;a->valueAndJacobianProduct(z,seed,v,ja);
    EXPECT_EQ(values->load(),1);EXPECT_NEAR(v(0),1.09,1e-14);
    z(0)=std::nextafter(z(0),1.);a->valueAndJacobianProduct(z,seed,v,ja);
    EXPECT_EQ(values->load(),2);EXPECT_EQ(jacs->load(),2);
    context->invalidate();b->valueAndJacobianProduct(z.head(2),dir,w,jb);
    EXPECT_EQ(values->load(),3);
    auto independent=manager.at("r").function->clone();independent->setEvaluationContext(std::make_shared<EvaluationContext>());
    independent->valueAndJacobianProduct(z,seed,v,ja);EXPECT_EQ(values->load(),4);
    a->setEvaluationContext(nullptr);a->valueAndJacobianProduct(z,seed,v,ja);EXPECT_EQ(values->load(),5);
}

TEST(IlqrFunctions, NumericalFailureDoesNotPoisonSharedNativeCache)
{
    auto calls=std::make_shared<int>(0),attempts=std::make_shared<int>(0);
    auto source=std::make_shared<CallbackFunction>(1,1,
        [calls](Input x,VectorOutput y){++*calls;y(0)=x(0)*x(0);},
        [attempts](Input x,MatrixOutput j){j(0,0)=123;if(++*attempts==1) throw EvaluationError("failed derivative");j(0,0)=2*x(0);});
    EvaluationContext context;auto a=context.share(source),b=context.share(source);
    Vector x=Vector::Constant(1,.3),value(1);Matrix jac(1,1);
    EXPECT_THROW(a->valueAndJacobian(x,value,jac),EvaluationError);
    b->valueAndJacobian(x,value,jac);EXPECT_NEAR(value(0),.09,1e-14);EXPECT_NEAR(jac(0,0),.6,1e-14);
    EXPECT_EQ(*calls,2);EXPECT_EQ(*attempts,2);
}

TEST(IlqrFunctions, SharedNativeCacheKeysFunctionIdentityNotSharedOwnership)
{
    struct Pair {std::shared_ptr<Function> a,b;};
    auto pair=std::make_shared<Pair>();
    auto make=[](double scale) {return std::make_shared<CallbackFunction>(1,1,
        [scale](Input x,VectorOutput y){y(0)=scale*x(0);},
        [scale](Input,MatrixOutput j){j(0,0)=scale;});};
    pair->a=make(1);pair->b=make(2);
    std::shared_ptr<const Function> a(pair,pair->a.get()),b(pair,pair->b.get());
    EvaluationContext context;auto ca=context.share(a),cb=context.share(b);
    Vector x=Vector::Ones(1),va(1),vb(1);Matrix ja(1,1),jb(1,1);
    ca->valueAndJacobian(x,va,ja);cb->valueAndJacobian(x,vb,jb);
    EXPECT_DOUBLE_EQ(va(0),1);EXPECT_DOUBLE_EQ(vb(0),2);
    EXPECT_DOUBLE_EQ(ja(0,0),1);EXPECT_DOUBLE_EQ(jb(0,0),2);
}

TEST(IlqrFunctions, SparseCasadiDirectionalOutputsAndStructuralZeros)
{
    auto x=casadi::SX::sym("x",3);
    CasadiFunction f(casadi::Function("sparse",{x},{casadi::SX::vertcat({x(0)*x(2),casadi::SX(0),x(1)+2*x(2)})}));
    Vector z(3),value(3);z<<2,3,4;Matrix seed(3,2),product(3,2);seed<<1,2,3,4,5,6;
    f.valueAndJacobianProduct(z,seed,value,product);
    Matrix expected(3,2);expected<<14,20,0,0,13,16;
    EXPECT_LT((product-expected).norm(),1e-14);EXPECT_DOUBLE_EQ(value(1),0);
    seed.setZero();f.valueAndJacobianProduct(z,seed,value,product);EXPECT_DOUBLE_EQ(product.norm(),0);
}
