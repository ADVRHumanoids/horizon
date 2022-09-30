#include "wrapped_function.h"
#include "profiling.h"

using namespace casadi_utils;

extern horizon::utils::Timer::TocCallback on_timer_toc;


WrappedFunction::WrappedFunction(casadi::Function f)
{
    *this = f;
}

WrappedFunction &WrappedFunction::operator=(casadi::Function f)
{
    if(f.is_null())
    {
        return *this;
    }

    _f = f;

    // resize work vectors
    _iw.assign(_f.sz_iw(), 0);
    _dw.assign(_f.sz_w(), 0.);

    // resize input buffers (note: sz_arg might be > n_in!!)
    _in_buf.assign(_f.sz_arg(), nullptr);

    // create memory for output data
    for(int i = 0; i < _f.n_out(); i++)
    {
        const auto& sp = _f.sparsity_out(i);

        // allocate memory for all nonzero elements of this output
        _out_data.emplace_back(sp.nnz(), 0.);

        // push the allocated buffer address to a vector
        _out_buf.push_back(_out_data.back().data());

        // allocate a zero dense matrix to store the output
        _out_matrix.emplace_back(Eigen::MatrixXd::Zero(sp.size1(), sp.size2()));

        //allocate a zero sparse matrix to store the output
        _out_matrix_sparse.emplace_back(Eigen::SparseMatrix<double>(sp.size1(), sp.size2()));

        // save sparsity pattern for i-th output
        std::vector<casadi_int> rows, cols;
        sp.get_triplet(rows, cols);
        _rows.push_back(rows);
        _cols.push_back(cols);
    }

    for(int i = _f.n_out(); i < _f.sz_res(); i++)
    {
        _out_buf.push_back(nullptr);
    }

    return *this;
}

WrappedFunction::WrappedFunction(const WrappedFunction & other)
{
    *this = other._f;
}

void WrappedFunction::setInput(int i, Eigen::Ref<const Eigen::VectorXd> xi)
{
    if(xi.size() != _f.size1_in(i))
    {
        throw std::invalid_argument(_f.name() + ": input size mismatch");
    }

    if(xi.hasNaN() || !xi.allFinite())
    {
        std::ostringstream oss;
        oss << _f.name() << " input " << i << " contains invalid values: \n" <<
               xi.transpose().format(3);
        throw std::runtime_error(oss.str());
    }

    _in_buf[i] = xi.data();
}

void WrappedFunction::call(bool sparse)
{
    // call function (allocation-free)
    casadi_int mem = _f.checkout();
    _f(_in_buf.data(), _out_buf.data(), _iw.data(), _dw.data(), mem);

    // copy all outputs to dense matrices
    for(int i = 0; i < _f.n_out(); ++i)
    {

        if(sparse)
        {
            csc_to_sparse_matrix(_f.sparsity_out(i),
                                 _rows[i], _cols[i],
                                 _out_data[i],
                                 _out_matrix_sparse[i]);
        }
        else
        {
            csc_to_matrix(_f.sparsity_out(i),
                          _rows[i], _cols[i],
                          _out_data[i],
                          _out_matrix[i]);

            if(_out_matrix[i].hasNaN() || !_out_matrix[i].allFinite())
            {
                std::ostringstream oss;
                oss << _f.name() << " output " << i << " contains invalid values: \n";
                oss << _out_matrix[i].format(3) << "\n";
                for(int j = 0; j < _f.n_in(); j++)
                {
                    auto u = Eigen::VectorXd::Map(_in_buf[j],
                                                  _f.size1_in(j));
                    oss << _f.name() << " input " << j <<
                           " = " << u.transpose().format(3) << "\n";
                }
                throw std::runtime_error(oss.str());
            }
        }

    }

    // release mem (?)
    _f.release(mem);
}

void WrappedFunction::call_accumulate(std::vector<Eigen::Ref<Eigen::MatrixXd>> &out)
{
    // call function (allocation-free)
    casadi_int mem = _f.checkout();

    {

#ifdef HORIZON_PROFILING
        horizon::utils::Timer tm("call_accumulate_" + _f.name() + "_inner",
                                 on_timer_toc);
#endif

        _f(_in_buf.data(), _out_buf.data(), _iw.data(), _dw.data(), mem);
    }

#ifdef HORIZON_PROFILING
    horizon::utils::Timer tm("csc_to_matrix_accu_" + _f.name() + "_inner",
                             on_timer_toc);
#endif

    // sum all outputs to out matrices
    for(int i = 0; i < _f.n_out(); ++i)
    {
        csc_to_matrix_accu(_f.sparsity_out(i),
                           _rows[i], _cols[i],
                           _out_data[i],
                           out[i]);
    }

    // release mem (?)
    _f.release(mem);
}

const Eigen::MatrixXd& WrappedFunction::getOutput(int i) const
{
    if(_out_matrix[i].hasNaN() || !_out_matrix[i].allFinite())
    {
        std::cout << "invalid value in output of " << _f.name() << "\n";
        std::cout << "output #" << i << ": \n" << _out_matrix[i].format(3) << "\n";
        for(int j = 0; j < _f.n_in(); j++)
        {
            auto in = Eigen::VectorXd::Map(_in_buf[j], _f.size1_in(j));
            std::cout << "input #" << j << ": \n" << in.transpose().format(3) << "\n";
        }
    }

    return _out_matrix[i];
}

const Eigen::SparseMatrix<double>& WrappedFunction::getSparseOutput(int i) const
{
    return _out_matrix_sparse[i];
}

Eigen::MatrixXd& WrappedFunction::out(int i)
{
    return _out_matrix[i];
}

casadi::Function& WrappedFunction::functionRef()
{
    return _f;
}

const casadi::Function &WrappedFunction::function() const
{
    return _f;
}

bool WrappedFunction::is_valid() const
{
    return !_f.is_null();
}

void WrappedFunction::csc_to_sparse_matrix(const casadi::Sparsity& sp,
                                           const std::vector<casadi_int>&  sp_rows,
                                           const std::vector<casadi_int>&  sp_cols,
                                           const std::vector<double>& data,
                                           Eigen::SparseMatrix<double>& matrix)
{
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(data.size());
    for(unsigned int i = 0; i < data.size(); ++i)
        triplet_list.push_back(Eigen::Triplet<double>(sp_rows[i], sp_cols[i], data[i]));

    matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

void WrappedFunction::csc_to_matrix(const casadi::Sparsity& sp,
                                    const std::vector<casadi_int>&  sp_rows,
                                    const std::vector<casadi_int>&  sp_cols,
                                    const std::vector<double>& data,
                                    Eigen::MatrixXd& matrix)
{
    // if dense output, do copy assignment which should be
    // faster
    if(sp.is_dense())
    {
        matrix = Eigen::MatrixXd::Map(data.data(),
                                      matrix.rows(),
                                      matrix.cols());

        return;
    }


    for(int k = 0; k < sp.nnz(); k++)
    {
        // current elem row index
        int row_i = sp_rows[k];
        int col_j = sp_cols[k];

        // copy data
        matrix(row_i, col_j) =  data[k];
    }
}

void WrappedFunction::csc_to_matrix_accu(const casadi::Sparsity &sp,
                                         const std::vector<casadi_int> &sp_rows,
                                         const std::vector<casadi_int> &sp_cols,
                                         const std::vector<double> &data,
                                         Eigen::Ref<Eigen::MatrixXd> matrix)
{
    // if dense output, do copy assignment which should be
    // faster
    if(sp.is_dense())
    {
        matrix += Eigen::MatrixXd::Map(data.data(),
                                       matrix.rows(),
                                       matrix.cols());

        return;
    }


    for(int k = 0; k < sp.nnz(); k++)
    {
        // current elem row index
        int row_i = sp_rows[k];
        int col_j = sp_cols[k];

        // copy data
        matrix(row_i, col_j) +=  data[k];
    }
}

