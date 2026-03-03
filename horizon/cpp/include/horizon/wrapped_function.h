#ifndef WRAPPED_FUNCTION_H
#define WRAPPED_FUNCTION_H

#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace casadi_utils
{

template <typename T, int r_size, int c_size>
void toCasadiMatrix(const Eigen::Matrix<T, r_size, c_size>& E, casadi::Matrix<T>& C)
{
    C.resize(E.rows(), E.cols());
    C = casadi::Matrix<T>::zeros(E.rows(), E.cols());
    std::memcpy(C.ptr(), E.data(), sizeof(T)*E.rows()*E.cols());
}

template <typename T, int r_size, int c_size>
void toEigen(const casadi::Matrix<T>& C, Eigen::Matrix<T, r_size, c_size>& E)
{
    E.setZero(C.rows(), C.columns());
    std::memcpy(E.data(), C.ptr(), sizeof(T)*C.rows()*C.columns());
}


template <typename T>
class WrappedSparseMatrix
{
public:
    WrappedSparseMatrix(){}

    WrappedSparseMatrix(const Eigen::SparseMatrix<T>& E)
    {
        to_triplets(E, _r, _c);
        _s = casadi::Sparsity::triplet(E.rows(), E.cols(), _r, _c);
        _values = casadi::DM::zeros(E.nonZeros());
        std::memcpy(_values.ptr(), E.valuePtr(), sizeof(double)*E.nonZeros());
        _C = casadi::Matrix<T>(_s, _values);
    }

    const casadi::Matrix<T>& get(){ return _C;}

    void update_values(const Eigen::SparseMatrix<T>& E)
    {
        if(E.rows() != _C.rows()) throw std::runtime_error("E.rows() != C.rows()");
        if(E.cols() != _C.columns()) throw std::runtime_error("E.cols() != C.cols()");
        if(E.nonZeros() != _C.nnz()) throw std::runtime_error("E.nnz() != C.nnz()");


        std::memcpy(_C->data(), E.valuePtr(), sizeof(T)*E.nonZeros());
    }

    static void to_triplets(const Eigen::SparseMatrix<T> & M,
                           std::vector<long long int>& rows,
                           std::vector<long long int>& cols);

    bool is_init()
    {
        return (_C.rows() > 0) ? true : false;
    }

private:


    std::vector<long long int> _r, _c;

    casadi::Matrix<T> _values;

    casadi::Matrix<T> _C;
    casadi::Sparsity _s;
};

template <typename T>
void WrappedSparseMatrix<T>::to_triplets(const Eigen::SparseMatrix<T> & M,
                       std::vector<long long int>& rows,
                       std::vector<long long int>& cols)
{
    rows.resize(M.nonZeros(), 0.);
    cols.resize(M.nonZeros(), 0.);

    unsigned int outer = M.outerSize();
    long long int c = 0;
    for (unsigned int k = 0; k < outer; ++k)
    {
        for (typename Eigen::SparseMatrix<T, Eigen::ColMajor>::InnerIterator it(M,k); it; ++it)
        {
            rows[c] = it.row();
            cols[c] = it.col();
            c++;
        }
    }

}


class WrappedFunction
{

public:

    WrappedFunction() = default;
    WrappedFunction(casadi::Function f);
    WrappedFunction& operator=(casadi::Function f);

    WrappedFunction(const WrappedFunction&);
    WrappedFunction& operator=(const WrappedFunction&);

    void setInput(int i, Eigen::Ref<const Eigen::VectorXd> xi);
    void call(bool sparse = false);
    void call_accumulate(std::vector<Eigen::Ref<Eigen::MatrixXd>>& out);
    const Eigen::MatrixXd& getOutput(int i) const;
    const Eigen::SparseMatrix<double>& getSparseOutput(int i) const;
    casadi::Function& functionRef();
    const casadi::Function& function() const;
    Eigen::MatrixXd& out(int i);

    bool is_valid() const;

private:

    void csc_to_matrix(const casadi::Sparsity& sp,
                       const std::vector<casadi_int>&  sp_rows,
                       const std::vector<casadi_int>&  sp_cols,
                       const std::vector<double>& data,
                       Eigen::MatrixXd& matrix);

    void csc_to_matrix_accu(const casadi::Sparsity& sp,
                            const std::vector<casadi_int>&  sp_rows,
                            const std::vector<casadi_int>&  sp_cols,
                            const std::vector<double>& data,
                            Eigen::Ref<Eigen::MatrixXd> matrix);

    void csc_to_sparse_matrix(const casadi::Sparsity& sp,
                              const std::vector<casadi_int>&  sp_rows,
                              const std::vector<casadi_int>&  sp_cols,
                              const std::vector<double>& data,
                              Eigen::SparseMatrix<double>& matrix);

    std::vector<const double *> _in_buf;
    std::vector<std::vector<double>> _out_data;
    std::vector<Eigen::MatrixXd> _out_matrix;
    std::vector<Eigen::SparseMatrix<double> > _out_matrix_sparse;
    std::vector<double *> _out_buf;
    std::vector<casadi_int> _iw;
    std::vector<double> _dw;
    std::vector<std::vector<casadi_int>> _rows;
    std::vector<std::vector<casadi_int>> _cols;

    casadi::Function _f;
};

}

#endif // WRAPPED_FUNCTION_H
