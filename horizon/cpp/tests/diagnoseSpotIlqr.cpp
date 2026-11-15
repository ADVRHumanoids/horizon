#include "spot_problem.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv)
{
    try
    {
        int N=argc>1?std::stoi(argv[1]):16;
        auto form=argc>2 && std::string(argv[2])=="torque"?spot_test::Formulation::Torque:spot_test::Formulation::AccelerationForce;
        spot_test::ProblemOptions options;
        std::string terminal=argc>4?argv[4]:"full";
        if(terminal=="base") options.terminal_velocity=spot_test::ProblemOptions::Base;
        else if(terminal=="none") options.terminal_velocity=spot_test::ProblemOptions::None;
        else if(terminal!="full") throw std::invalid_argument("terminal policy must be full, base, or none");
        if(argc>5) options.rank_threshold=std::stod(argv[5]);
        if(argc>6) options.kkt_decomposition=argv[6];
        if(argc>7) options.project_contacts=std::stoi(argv[7])!=0;
        if(argc>8) options.scale_regularization=std::stoi(argv[8])!=0;
        spot_test::Problem problem(form,N,0,options);
        if(argc>3 && std::string(argv[3])=="solve")
        {
            auto m=problem.solve();
            std::cout<<std::setprecision(12)<<"RESULT N="<<N<<" form="<<spot_test::name(form)
                <<" terminal="<<terminal<<" rank_threshold="<<options.rank_threshold
                <<" kkt="<<options.kkt_decomposition
                <<" projected="<<options.project_contacts
                <<" scaled_reg="<<options.scale_regularization
                <<" converged="<<m.converged<<" iterations="<<m.iterations<<" ms="<<m.solve_ms
                <<" objective="<<m.objective<<" end_v="<<m.endpoint_velocity
                <<" gap="<<m.dynamics_gap<<" contact_acc="<<m.contact_acceleration
                <<" foot_drift="<<m.foot_drift<<" normal="<<m.min_normal
                <<" friction="<<m.min_friction_margin<<"\n";
        }
        else problem.diagnoseLqStep(std::cout,argc>3?std::stoi(argv[3]):0);
        return 0;
    }
    catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
