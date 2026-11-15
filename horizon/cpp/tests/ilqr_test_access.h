#pragma once
#include "ilqr/ilqr_impl.h"
#include <iostream>

namespace horizon
{
/** Independent, dense condensed-QP oracle for tests and diagnostics only. */
struct IterativeLQRTestAccess
{
    static void diagnose(IterativeLQR& solver,std::ostream& out)
    {
        auto& s=solver;
        if(!s.fixed_initial_state()) throw std::invalid_argument("diagnostic requires fixed initial state");
        s.linearize_quadratize();
        std::string failure;
        try {s.backward_pass();} catch(const std::exception& e) {failure=e.what();}
        const int nx=s._nx,nu=s._nu,N=s._N,m=N*nu;
        Eigen::MatrixXd H=Eigen::MatrixXd::Zero(m,m),X=Eigen::MatrixXd::Zero(nx,m);
        Eigen::VectorXd g=Eigen::VectorXd::Zero(m),c(nx);
        s._space->difference(s.state(0),s._x_lb.col(0),c);
        std::vector<Eigen::MatrixXd> eq;
        std::vector<Eigen::VectorXd> rhs;
        std::vector<Eigen::MatrixXd> roots;
        std::vector<Eigen::VectorXd> residuals;
        for(int k=0;k<=N;++k)
        {
            Eigen::MatrixXd Q=s._cost[k].Q();Q.diagonal().array()+=s._hxx_reg;
            H.noalias()+=X.transpose()*Q*X;
            g.noalias()+=X.transpose()*(s._cost[k].q()+Q*c);
            auto append=[&](MatConstRef C,VecConstRef h,MatConstRef D) {
                if(!h.size()) return;
                Eigen::MatrixXd E=C*X;
                if(k<N && D.cols()) E.middleCols(k*nu,nu)+=D;
                eq.push_back(std::move(E));rhs.emplace_back(-h-C*c);
            };
            append(s._constraint[k].C(),s._constraint[k].h(),s._constraint[k].D());
            if(k>0)
            {
                Eigen::MatrixXd C;Eigen::VectorXd h;s.stateEquality(k,C,h);
                append(C,h,Eigen::MatrixXd::Zero(h.size(),nu));
            }
            const int local_size=nx+(k<N?nu:0);
            Eigen::MatrixXd G=Eigen::MatrixXd::Zero(local_size,local_size),T=Eigen::MatrixXd::Zero(local_size,m);
            Eigen::VectorXd local_g(local_size),affine=Eigen::VectorXd::Zero(local_size);
            G.topLeftCorner(nx,nx)=Q;T.topRows(nx)=X;local_g.head(nx)=s._cost[k].q();affine.head(nx)=c;
            if(k<N)
            {
                G.bottomRightCorner(nu,nu)=s._cost[k].R();G.bottomRightCorner(nu,nu).diagonal().array()+=s._huu_reg;
                G.bottomLeftCorner(nu,nx)=s._cost[k].P();G.topRightCorner(nx,nu)=s._cost[k].P().transpose();
                T.block(nx,k*nu,nu,nu).setIdentity();local_g.tail(nu)=s._cost[k].r();
            }
            Eigen::LLT<Eigen::MatrixXd> llt(G);
            if(llt.info()!=Eigen::Success) throw std::runtime_error("non-positive local GN Hessian in QP oracle");
            Eigen::MatrixXd F=llt.matrixU();
            roots.push_back(F*T);residuals.emplace_back(llt.matrixL().solve(local_g)+F*affine);
            if(k==N) break;
            auto& cost=s._cost[k];
            H.block(k*nu,k*nu,nu,nu)+=cost.R();
            H.block(k*nu,k*nu,nu,nu).diagonal().array()+=s._huu_reg;
            H.middleRows(k*nu,nu).noalias()+=cost.P()*X;
            H.middleCols(k*nu,nu).noalias()+=X.transpose()*cost.P().transpose();
            g.segment(k*nu,nu)+=cost.r()+cost.P()*c;
            c=s._dyn[k].A()*c+s._dyn[k].d;
            X=(s._dyn[k].A()*X).eval();X.middleCols(k*nu,nu)+=s._dyn[k].B();
        }
        int rows=0;for(const auto& e:eq) rows+=e.rows();
        Eigen::MatrixXd E(rows,m);Eigen::VectorXd b(rows);int off=0;
        for(size_t i=0;i<eq.size();++i) {E.middleRows(off,eq[i].rows())=eq[i];b.segment(off,rhs[i].size())=rhs[i];off+=eq[i].rows();}
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(E,Eigen::ComputeThinU|Eigen::ComputeFullV);
        svd.setThreshold(s._svd_threshold);int rank=svd.rank();
        Eigen::VectorXd p=svd.solve(b);Eigen::MatrixXd Z=svd.matrixV().rightCols(m-rank);
        int residual_size=0;for(const auto& r:residuals) residual_size+=r.size();
        Eigen::MatrixXd design(residual_size,m);Eigen::VectorXd residual(residual_size);off=0;
        for(size_t i=0;i<roots.size();++i) {design.middleRows(off,roots[i].rows())=roots[i];residual.segment(off,residuals[i].size())=residuals[i];off+=residuals[i].size();}
        Eigen::MatrixXd reduced=design*Z;
        Eigen::VectorXd scale=reduced.colwise().norm().transpose().cwiseMax(1e-30).cwiseInverse();
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(reduced*scale.asDiagonal());qr.setThreshold(1e-14);
        Eigen::VectorXd v=qr.solve(-(residual+design*p));
        Eigen::VectorXd reference=p+Z*(scale.asDiagonal()*v);
        Eigen::VectorXd actual=Eigen::Map<Eigen::VectorXd>(s._du.data(),m);
        auto model=[&](const Eigen::VectorXd& u) {return .5*(design*u+residual).squaredNorm()-.5*residual.squaredNorm();};
        out<<"N="<<N<<" controls="<<nu<<" eq_rows="<<rows<<" rank="<<rank
           <<" singular_max="<<svd.singularValues()(0)<<" smallest_kept="<<(rank?svd.singularValues()(rank-1):0)
           <<" reference_cost="<<model(reference)<<" reference_eq="<<(E*reference-b).norm()
           <<" reference_stationarity="<<(Z.transpose()*design.transpose()*(design*reference+residual)).norm()/(1+g.norm())
           <<" reduced_rank="<<qr.rank()<<" free_controls="<<Z.cols();
        if(failure.empty())
            out<<" riccati_cost="<<model(actual)<<" riccati_eq="<<(E*actual-b).norm()
               <<" riccati_stationarity="<<(Z.transpose()*design.transpose()*(design*actual+residual)).norm()/(1+g.norm())
               <<" control_relative_error="<<(actual-reference).norm()/(1+reference.norm());
        else out<<" riccati_error="<<failure;
        out<<"\n";
    }
};
}
