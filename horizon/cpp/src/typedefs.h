#ifndef TYPDEFS_H
#define TYPDEFS_H

#include <Eigen/Dense>

namespace horizon
{

#ifdef HORIZON_FLOAT32
typedef float Real;
#else
typedef double Real;
#endif

typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> VectorXr;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixXr;

typedef Eigen::Matrix<Real, 2, 2> Matrix2r;
typedef Eigen::Matrix<Real, 2, 1> Vector2r;

typedef Eigen::Matrix<Real, 1, Eigen::Dynamic> RowVectorXr;

const Real RInfinity = std::numeric_limits<Real>::infinity();

}

#endif // TYPDEFS_H