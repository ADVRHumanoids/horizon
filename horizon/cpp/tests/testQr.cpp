#include <Eigen/Dense>
#include <iostream>
#include <gtest/gtest.h>

#include <chrono>
#include <numeric>
#include "typedefs.h"

using Real=horizon::Real;
using MatrixXr=horizon::MatrixXr;
using VectorXr=horizon::VectorXr;
using Matrix2r=horizon::Matrix2r;
using Vector2r=horizon::Vector2r;

TEST(testQr, rank)
{
    MatrixXr A, R;
    Eigen::HouseholderQR<MatrixXr> qr;

    for(int i = 0; i < 100000; i++)
    {
        A = MatrixXr::Random(10, 3)*MatrixXr::Random(3, 5);
        qr.compute(A);
        R = qr.matrixQR().triangularView<Eigen::Upper>();

        EXPECT_LT(std::fabs(R(3, 3)), 1e-9);
        EXPECT_LT(std::fabs(R(4, 4)), 1e-9);

        A = MatrixXr::Random(5, 3)*MatrixXr::Random(3, 10);
        qr.compute(A);
        R = qr.matrixQR().triangularView<Eigen::Upper>();

        EXPECT_LT(std::fabs(R(3, 3)), 1e-9);
        EXPECT_LT(std::fabs(R(4, 4)), 1e-9);
    }
}

TEST(testDecomp, compare)
{
    int nu = 18 + 12;
    int nc = 6 + 12;

    MatrixXr K, Huu, D;
    K.setZero(nu+nc, nu+nc);

    Huu.setIdentity(nu, nu);
    D.setRandom(nc, nu);
    K.topLeftCorner(nu, nu) = Huu;
    K.bottomLeftCorner(nc, nu) = D;
    K.topRightCorner(nu, nc) = D.transpose();

    Eigen::ColPivHouseholderQR<MatrixXr> qr(K);
    Eigen::BDCSVD<MatrixXr> bdcsvd(K);
    Eigen::JacobiSVD<MatrixXr> jsvd(K);
    Eigen::CompleteOrthogonalDecomposition<MatrixXr> cod(K);
    Eigen::FullPivLU<MatrixXr> lu(K);

    int n_trials = 1000;
    std::map<std::string, std::vector<Real>> times;
    using hrc = std::chrono::high_resolution_clock;
    hrc::time_point tic, toc;
    for(int i = 0; i < n_trials; i++)
    {
        Huu.setRandom(nu, nu);
        Huu = Huu*Huu.transpose();
        D.setRandom(nc, nu);
        K.topLeftCorner(nu, nu) = Huu;
        K.bottomLeftCorner(nc, nu) = D;
        K.topRightCorner(nu, nc) = D.transpose();
        K = D;


        tic = hrc::now();
        qr.compute(K);
        MatrixXr q = qr.householderQ();
        toc = hrc::now();
        times["qr"].push_back((toc - tic).count()*1e-3);

        tic = hrc::now();
        bdcsvd.compute(K);
        toc = hrc::now();
        times["bdcsvd"].push_back((toc - tic).count()*1e-3);

        tic = hrc::now();
        jsvd.compute(K);
        toc = hrc::now();
        times["jsvd"].push_back((toc - tic).count()*1e-3);

        tic = hrc::now();
        cod.compute(K);
        toc = hrc::now();
        times["cod"].push_back((toc - tic).count()*1e-3);

        tic = hrc::now();
        lu.compute(K);
        toc = hrc::now();
        times["lu"].push_back((toc - tic).count()*1e-3);
    }

    for(auto& item : times)
    {
        int avg = std::accumulate(item.second.begin(), item.second.end(), .0);
        std::cout << item.first << ": " << avg/item.second.size() << " us \n";
    }

}


TEST(testLdlt, basic)
{
#if false
    MatrixXr L;
    L.setZero(3, 3);
    L.triangularView<Eigen::Lower>() = L.Random(3, 3);
    L.diagonal().setConstant(1);

    VectorXr d = d.Random(3);

    MatrixXr K = L*d.asDiagonal()*L.transpose();

    std::cout << "L=\n" << L << std::endl;
    std::cout << "d = " << d.transpose() << std::endl;


    int n = K.rows();
    VectorXr AP = AP.Zero(n*(n+1)/2);
    Eigen::VectorXi ipiv(n);


    int ap_idx = 0;
    for(int j = 0; j < n; j++)
    {
        AP.segment(ap_idx, n-j) = K.col(j).tail(n-j);
        ap_idx += n-j;
    }

    std::cout << "K=\n" << K << std::endl;
    std::cout << "AP = \n" << AP.transpose() << std::endl;

    int info = LAPACKE_dsptrf(LAPACK_COL_MAJOR, 'L', n, AP.data(), ipiv.data());

    std::cout << "AP = \n" << AP.transpose() << std::endl;

    std::cout << "ipiv = " << ipiv.transpose() << std::endl;
#endif
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
