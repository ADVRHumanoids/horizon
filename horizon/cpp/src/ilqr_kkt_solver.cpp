#include "ilqr.h"
#include "ilqr_impl.h"

// #include <mkl/mkl.h>
// #include <Eigen/src/PardisoSupport/PardisoSupport.h>

namespace {

void fill_block(Eigen::Ref<const Eigen::MatrixXd> block,
                int offset_i,
                int offset_j,
                const Eigen::SparseMatrix<double>& m,
                std::vector<Eigen::Triplet<double>>& tr)
{
    if(block.rows() + offset_i > m.rows() || block.cols() + offset_j > m.cols())
    {
        throw std::runtime_error("fill block index error");
    }

    for(int j = 0; j < block.cols(); j++)
    {
        for(int i = 0; i < block.rows(); i++)
        {
            tr.emplace_back(offset_i + i, offset_j + j, block(i, j));

            if(offset_i != offset_j)
            {
                tr.emplace_back(offset_j + j, offset_i + i,  block(i, j));
            }
        }
    }
}

void fill_m_eye(int n,
                double value,
                int offset_i,
                int offset_j,
                const Eigen::SparseMatrix<double>& m,
                std::vector<Eigen::Triplet<double>>& tr)
{
    if(n + offset_i > m.rows() || n + offset_j > m.cols())
    {
        throw std::runtime_error("fill block index error");
    }

    for(int i = 0; i < n; i++)
    {
        tr.emplace_back(offset_i + i, offset_j + i, value);
        tr.emplace_back(offset_j + i, offset_i + i, value);
    }
}
}

void IterativeLQR::kkt_solve()
{
    // w = [dx_0 dx_1 ... dx_N du_0 .... du_N-1 lam_0 ... lam_N-1 mu_0 ... mu_nc]

    // fill kkt matrix and rhs
    TIC(fill_kkt)

    _kkt_triplets.clear();

    // compute size (_N+1 states, _N controls, _N dynamic constraints)
    int kkt_size = _nx*(2*_N+1) + _nu*_N;

    // add constraints
    for(auto& c : _constraint)
    {
        kkt_size += c.size();
    }

    // add bounds if lb == ub
    for(int k = 0; k < _N+1; k++)
    {
        for(int i = 0; i < _nx; i++)
        {
            if(_x_lb(i, k) == _x_ub(i, k))
            {
                _kkt_rhs.conservativeResize(kkt_size + 1);
                _kkt_triplets.emplace_back(kkt_size, k*_nx + i, 1);
                _kkt_triplets.emplace_back(k*_nx + i, kkt_size, 1);
                _kkt_rhs(kkt_size) = _x_lb(i, k) - _xtrj(i, k);
                kkt_size += 1;
            }
        }

        if(k == _N)
        {
            break;
        }

        for(int i = 0; i < _nu; i++)
        {
            if(_u_lb(i, k) == _u_ub(i, k))
            {
                _kkt_rhs.conservativeResize(kkt_size + 1);
                _kkt_triplets.emplace_back(kkt_size, (_N+1)*_nx + k*_nu + i, 1);
                _kkt_triplets.emplace_back((_N+1)*_nx + k*_nu + i, kkt_size, 1);
                _kkt_rhs(kkt_size) = _u_lb(i, k) - _utrj(i, k);
                kkt_size += 1;
            }
        }
    }

    _kkt_mat.resize(kkt_size, kkt_size);
    _kkt_rhs.resize(kkt_size);

    // fill cost
    for(int i = 0; i < _N+1; i++)
    {
        ::fill_block(_cost[i].Q(),
                     i*_nx,
                     i*_nx,
                     _kkt_mat,
                     _kkt_triplets);

        ::fill_m_eye(_nx,
                     _hxx_reg,
                     i*_nx,
                     i*_nx,
                     _kkt_mat,
                     _kkt_triplets);

        _kkt_rhs.segment(i*_nx, _nx) = -_cost[i].q();

    }

    for(int i = 0; i < _N; i++)
    {
        int row_offset = _nx*(_N+1);

        ::fill_block(_cost[i].R(),
                     row_offset + i*_nu,
                     row_offset + i*_nu,
                     _kkt_mat,
                     _kkt_triplets);

        ::fill_block(_cost[i].P(),
                     row_offset + i*_nu,
                     i*_nx,
                     _kkt_mat,
                     _kkt_triplets);

        _kkt_rhs.segment(row_offset + i*_nu, _nu) = -_cost[i].r();
    }

    // fill dynamics
    for(int i = 0; i < _N; i++)
    {
        int row_offset = _nx*(_N+1) + _nu*_N;

        ::fill_block(_dyn[i].A(),
                     row_offset + i*_nx,
                     i*_nx,
                     _kkt_mat,
                     _kkt_triplets);

        ::fill_m_eye(_nx,
                     -1,
                     row_offset + i*_nx,
                     (i+1)*_nx,
                     _kkt_mat,
                     _kkt_triplets);

        ::fill_block(_dyn[i].B(),
                     row_offset + i*_nx,
                     _nx*(_N+1) + i*_nu,
                     _kkt_mat,
                     _kkt_triplets);

        _kkt_rhs.segment(row_offset + i*_nx, _nx) = -_dyn[i].d;
    }

    // fill constraints
    int row_offset = (2*_N + 1)*_nx + _N*_nu;

    for(int i = 0; i < _N+1; i++)
    {
        auto& c = _constraint[i];

        if(c.size() == 0)
        {
            continue;
        }

        ::fill_block(c.C(),
                     row_offset,
                     i*_nx,
                     _kkt_mat,
                     _kkt_triplets);


        _kkt_rhs.segment(row_offset, c.size()) = -c.h();

        if(i == _N)
        {
            break;
        }

        ::fill_block(c.D(),
                     row_offset,
                     (_N+1)*_nx + i*_nu,
                     _kkt_mat,
                     _kkt_triplets);

        row_offset += c.size();

    }

    _kkt_mat.setFromTriplets(_kkt_triplets.begin(), _kkt_triplets.end());

    _kkt_mat.makeCompressed();

    // std::cout << _kkt_mat.toDense().format(2) << std::endl;

    // std::cout << _kkt_rhs << std::endl;

    TOC(fill_kkt)


    // solve kkt
    TIC(solve_kkt)

    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::COLAMDOrdering<int>> _kkt_solver;

    // _kkt_solver.analyzePattern(_kkt_mat);

    // _kkt_solver.factorize(_kkt_mat);

    auto& kkt_solver = _kkt_lu_solver;

    // Eigen::PardisoLU<Eigen::SparseMatrix<double>> kkt_solver;

    // kkt_solver.isSymmetric(true);

    kkt_solver.compute(_kkt_mat);

    if(kkt_solver.info() != Eigen::ComputationInfo::Success)
    {
        std::cerr << "kkt solve failed! \n";
        throw std::runtime_error("aaa");
    }

    Eigen::VectorXd sol = kkt_solver.solve(_kkt_rhs);

    for(int i = 0; i < 0; i++)
    {
        _kkt_rhs = _kkt_rhs - _kkt_mat * sol;

        sol += kkt_solver.solve(_kkt_rhs);
    }

    TOC(solve_kkt)

    // save newton step and multipliers
    TIC(kkt_save_newton_step);

    row_offset = 0;

    Eigen::VectorXd::Map(_dx.data(), _dx.size()) = sol.segment(row_offset, _dx.size());
    row_offset += _dx.size();

    Eigen::VectorXd::Map(_du.data(), _du.size()) = sol.segment(row_offset, _du.size());
    row_offset += _du.size();

    Eigen::VectorXd::Map(_lam_x.data(), _lam_x.size()) = sol.segment(row_offset, _lam_x.size());
    row_offset += _lam_x.size();

    for(int i = 0; i < _N+1; i++)
    {
        auto& c = _constraint[i];

        _lam_g[i].resize(c.size());

        if(c.size() == 0)
        {
            continue;
        }

        _lam_g[i] = sol.segment(row_offset, c.size());
        row_offset += c.size();
    }
}
