#include <gtest/gtest.h>
#include "../src/wrapped_function.h"
#include "typedefs.h"

namespace{

using Real=horizon::Real;
using MatrixXr=horizon::MatrixXr;
using VectorXr=horizon::VectorXr;

class testCasadiUtils: public ::testing::Test
{
protected:
    testCasadiUtils(){

    }

    virtual ~testCasadiUtils() {

    }

    virtual void SetUp() {

    }

    virtual void TearDown() {

    }

};

TEST_F(testCasadiUtils, testSparseHessian)
{
//    // This compile
    MatrixXr A, B;
    A.resize(8,8);
    A.setRandom(8,8);
    B.resize(8, 8);
    B.triangularView<Eigen::Upper>() = A.transpose()*A;

    //This does not compile
    Eigen::SparseMatrix<Real> J, H;
    J.resize(8, 8);
    J.setIdentity();
    H.resize(8, 8);
    H.setZero();
    H.selfadjointView<Eigen::Lower>().rankUpdate(J.transpose());
}





void EXPECT_EQUAL(const Eigen::SparseMatrix<Real>& E, const casadi::DM& C)
{
    EXPECT_EQ(E.rows(), C.rows());
    EXPECT_EQ(E.cols(), C.columns());
    EXPECT_EQ(E.nonZeros(), C.nnz());

    std::vector<Real> e;
    e.assign(E.valuePtr(), E.valuePtr() + E.nonZeros());
    std::vector<Real> c;
    c.assign(C->data(), C->data() + C.nnz());

    for(unsigned int i = 0; i < e.size(); ++i)
        EXPECT_Real_EQ(e[i], c[i]);
}


TEST_F(testCasadiUtils, toCasadiSparse)
{

    std::default_random_engine gen;
    std::uniform_real_distribution<Real> dist(0.0,1.0);

    int rows=100;
    int cols=100;

    std::vector<Eigen::Triplet<Real> > tripletList;
    for(int i=0;i<rows;++i)
        for(int j=0;j<cols;++j)
        {
           auto v_ij=dist(gen);                         //generate random number
           if(v_ij < 0.1)
           {
               tripletList.push_back(Eigen::Triplet<Real>(i,j,v_ij));      //if larger than treshold, insert it
           }
        }
    Eigen::SparseMatrix<Real> E(rows,cols);
    E.setFromTriplets(tripletList.begin(), tripletList.end());

    //std::cout<<"E: "<<E<<std::endl;

    auto tic = std::chrono::high_resolution_clock::now();
    casadi_utils::WrappedSparseMatrix<Real> C(E);
    auto toc = std::chrono::high_resolution_clock::now();
    std::cout<<"Constructor: "<<(toc-tic).count()*1E-9<<"   [s]"<<std::endl;

    //std::cout<<"C: "<<C.get()<<std::endl;
    EXPECT_EQUAL(E, C.get());



    E.coeffRef(0, 6) = 2.;

    //std::cout<<"E: "<<E<<std::endl;



    tic = std::chrono::high_resolution_clock::now();
    C.update_values(E);
    toc = std::chrono::high_resolution_clock::now();
    std::cout<<"update_values: "<<(toc-tic).count()*1E-9<<"   [s]"<<std::endl;

    //std::cout<<"C: "<<C.get()<<std::endl;
    EXPECT_EQUAL(E, C.get());
}

TEST_F(testCasadiUtils, testToCasadiMatrix)
{
    MatrixXr E;
    E.setRandom(5,6);

    casadi::DM C;
    casadi_utils::toCasadiMatrix(E, C);

    EXPECT_EQ(E.rows(), C.rows());
    EXPECT_EQ(E.cols(), C.columns());

    for(unsigned int i = 0; i < E.rows(); ++i)
    {
        for(unsigned int j = 0; j < E.cols(); ++j)
            EXPECT_Real_EQ(E(i,j), Real(C(i,j)));
    }

    MatrixXr EE;
    casadi_utils::toEigen(C, EE);

    EXPECT_EQ(EE.rows(), C.rows());
    EXPECT_EQ(EE.cols(), C.columns());

    for(unsigned int i = 0; i < EE.rows(); ++i)
    {
        for(unsigned int j = 0; j < EE.cols(); ++j)
            EXPECT_Real_EQ(EE(i,j), Real(C(i,j)));
    }

    std::cout<<"E: \n"<<E<<std::endl;
    std::cout<<"EE: \n"<<EE<<std::endl;
    std::cout<<"C: "<<C<<std::endl;

    std::cout<<"C: "<<C<<std::endl;

    VectorXr e;
    e.setRandom(7);

    casadi::DM c;
    casadi_utils::toCasadiMatrix(e, c);

    EXPECT_EQ(e.rows(), c.rows());
    EXPECT_EQ(e.cols(), c.columns());

    for(unsigned int i = 0; i < e.size(); ++i)
        EXPECT_Real_EQ(e[i], Real(c(i)));

    VectorXr ee;
    casadi_utils::toEigen(c, ee);

    EXPECT_EQ(ee.rows(), c.rows());
    EXPECT_EQ(ee.cols(), c.columns());

    for(unsigned int i = 0; i < ee.size(); ++i)
        EXPECT_Real_EQ(ee[i], Real(c(i)));


    std::cout<<"e: "<<e.transpose()<<std::endl;
    std::cout<<"ee: "<<ee.transpose()<<std::endl;
    std::cout<<"c: "<<c<<std::endl;

}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
