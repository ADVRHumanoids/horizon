#include "ilqr/ilqr.h"
#include <phase_manager/phase_manager.h>
#include <phase_manager/timeline.h>
#include <phase_manager/phase.h>
#include <phase_manager/horizon_interface.h>
#include "casadi/casadi.hpp"
#include "horizon/functions.h"

int main()
{
    int n_nodes = 50;
    PhaseManager pm(n_nodes);

    auto timeline_1 = pm.createTimeline("first_timeline");

    int stance_duration = 5;
    auto stance_1 = timeline_1->createPhase(stance_duration, "stance_1");
    auto stance_2 = timeline_1->createPhase(stance_duration, "stance_2");


//    for (auto phase_i : timeline_1->getPhases())
//        {
//            std::cout <<" phase: " << phase_i->getName() << std::endl;
//        }

////    std::cout << has_set_bounds<ClassWithoutBounds>() << std::endl;
////    std::cout << has_set_bounds<Constraint>() << std::endl;

//////     all of this comes from outside
    auto v1 = std::make_shared<horizon::Variable>("v1", 3, n_nodes);
    auto v2 = std::make_shared<horizon::Variable>("v2", 3, n_nodes);

    auto par1 = std::make_shared<horizon::Parameter>("par1", 1, n_nodes);


    Eigen::MatrixXd mat(3, 50);
    int val = 0;

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            mat(i, j) = val++;
        }
    }

    std::cout << mat << std::endl;
    par1->setValues(mat);

    std::cout << par1->getValues() << std::endl;

    auto casadi_fun_1 = casadi::Function("f_1", {v1->getSym(), v2->getSym()}, {v1->getSym() + v2->getSym()});

    auto casadi_fun_2 = casadi::Function("f_2", {v1->getSym(), v2->getSym(), par1->getSym()}, {v1->getSym()(0) + v2->getSym()(1) / par1->getSym()});

    horizon::Cost::Ptr cost_1 = std::make_shared<horizon::Cost>(casadi_fun_1, n_nodes);
    auto constr_1 = std::make_shared<horizon::Constraint>(casadi_fun_2, n_nodes);

    ItemBase::Ptr item_cost_1 = std::make_shared<Wrapper<horizon::Cost>>(cost_1);
    ItemWithBoundsBase::Ptr item_constr_1 = std::make_shared<WrapperWithBounds<horizon::Constraint>>(constr_1);

    stance_1->addItem(item_cost_1);
    stance_2->addItem(item_constr_1);

    timeline_1->addPhase(stance_1);
    timeline_1->addPhase(stance_2);
//    timeline_1->addPhase(stance);
//    timeline_1->addPhase(stance);



    pm.update();

    std::cout << "cost_1: " << cost_1->getNodes() << std::endl;
    std::cout << "constr_1: " << constr_1->getNodes() << std::endl;

    pm.shift(); pm.update();
    pm.shift(); pm.update();
    pm.shift(); pm.update();
    pm.shift(); pm.update();

    std::cout << "cost_1: " << cost_1->getNodes() << std::endl;
    std::cout << "constr_1: " << constr_1->getNodes() << std::endl;



////    // ==================================================================================================================

////    // convert variable to something usable by phase_manager

////    ItemWithBoundsBase::Ptr var_1 = std::make_shared<WrapperWithBounds<Variable>>(fake_var_1);
////    ItemWithBoundsBase::Ptr stance_c_1 = std::make_shared<WrapperWithBounds<Constraint>>(fake_stance_c_1);
////    ItemWithBoundsBase::Ptr stance_c_2 = std::make_shared<WrapperWithBounds<Constraint>>(fake_stance_c_2);
////    ItemWithBoundsBase::Ptr flight_c_1 = std::make_shared<WrapperWithBounds<Constraint>>(fake_flight_c_1);
////    ItemWithValuesBase::Ptr flight_p_1 = std::make_shared<WrapperWithValues<Parameter>>(fake_flight_p_1);
//    ItemWithValuesBase::Ptr flight_p_1 = std::make_shared<WrapperWithValues<Item>>(fake_item_1);
////    Phase::Ptr flight = std::make_shared<Phase>(5, "flight");


////    stance->addConstraint(stance_c_1); //, my_nodes);
////    stance->addConstraint(stance_c_2);



////    Eigen::MatrixXd bounds_var_1 = Eigen::MatrixXd::Zero(1,5);
////    std::vector<int> var_phase_nodes;
////    for (int i = 0; i < 5; ++i) {
////            var_phase_nodes.push_back(i);
////        }


////    stance->addVariableBounds(var_1, bounds_var_1, bounds_var_1, var_phase_nodes);


////    flight->addVariableBounds(var_1, bounds_var_1, bounds_var_1);

////    flight->addConstraint(flight_c_1);

////    flight->addParameterValues(flight_p_1, values);

////    std::cout << "variable var_1 has bounds: " << std::endl;
////    std::cout << std::get<0>(fake_var_1->getBounds()) << std::endl;
////    std::cout << std::endl;

////    //  without registering, cannot link horizon constraints to updater
//    timeline_1->registerPhase(stance);
////    timeline_1->registerPhase(flight);

//////    auto start_time = std::chrono::high_resolution_clock::now();
//    for (int phase_num = 0; phase_num < 10; phase_num++)
//    {
//        timeline_1->addPhase(stance);
//    }
////    std::cout << "ADD PHASE: " << std::endl;
////    std::cout << std::get<0>(fake_var_1->getBounds()) << std::endl;
////    std::cout << std::endl;


//    //    this creates phasetokens
////    auto start_time = std::chrono::high_resolution_clock::now();
////    for (int i =0; i < 1000; i++)
////    {
////        timeline_1->addPhase(stance);
////    }

////    std::chrono::duration<double> elapsed_time = std::chrono::system_clock::now() - start_time;

////    std::cout << elapsed_time.count() << std::endl; //0.00026454 // 0.002 // 0.0008

//    //    std::cout << "fake_item_1 current nodes: ";

////    for (auto node : fake_item_1->getNodes())
////    {
////         std::cout << node << " ";
////    }
////    std::cout << std::endl;

////    std::cout << "fake_item_1 current values: " << fake_item_1->getValues() << std::endl;
//////    timeline_1->addPhase(stance);

//////    std::cout << "ADD PHASE: " << std::endl;
//////    std::cout << std::get<0>(fake_var_1->getBounds()) << std::endl;
//////    std::cout << std::endl;

////    for (auto phase_i : timeline_1->getPhases())
////    {
////        std::cout <<" active nodes of phase '" << phase_i->getName() << "': ";
////        for (auto node_i : phase_i->getActiveNodes())
////        {
////            std::cout << node_i << " ";
////        }
////        std::cout << std::endl;
////    }


////    for (auto pair : stance->getItemsReferenceInfo())
////    {
////        std::cout << pair.first << ": " << pair.second << std::endl;
////    }



//    Eigen::MatrixXd new_values(1, stance_duration);
//    new_values << 7, 7, 7, 7, 7;

//    Eigen::MatrixXd new_values_1(1, stance_duration);
//    new_values_1 << 4, 4, 4, 4, 4;

//    Eigen::MatrixXd new_values_2(1, stance_duration);
//    new_values_2 << 5, 5, 5, 5, 5;

//    Eigen::MatrixXd new_values_3(1, stance_duration);
//    new_values_3 << 6, 6, 6, 6, 6;


//    std::cout << "fake_item_1 current values: " << fake_item_1->getValues() << std::endl;
//    timeline_1->getPhases()[0]->setItemReference(flight_p_1->getName(), new_values);
//    timeline_1->getPhases()[1]->setItemReference(flight_p_1->getName(), new_values_1);
//    timeline_1->getPhases()[4]->setItemReference(flight_p_1->getName(), new_values_2);
//    timeline_1->getPhases()[7]->setItemReference(flight_p_1->getName(), new_values_3);

//    timeline_1->getPhases()[0]->update();
//    timeline_1->getPhases()[1]->update();
//    timeline_1->getPhases()[4]->update();
//    timeline_1->getPhases()[7]->update();

//    std::cout << "fake_item_1 current values: " << fake_item_1->getValues() << std::endl;
////    timeline_1->getPhases()[10]->setItemReference(flight_p_1, new_values);


////    std::cout << "fake_item_1 current values: " << fake_item_1->getValues() << std::endl;
////    for (int i =0; i < 6; i++)
////    {
////        timeline_1->shift();
////    }
////    std::cout << "fake_item_1 current values: " << fake_item_1->getValues() << std::endl;


////    timeline_1->addPhase(stance);
////    timeline_1->addPhase(flight, 10);


//////    std::cout << "active Phases" << std::endl;
//////    for (auto active_phase: timeline_1->getActivePhase())
//////    {
//////        std::cout << active_phase << std::endl;
//////    }


////    std::cout << "constraint stance_c_1 has nodes: ";
////    for (int i : fake_stance_c_1->getNodes())
////    {
////        std::cout << i << " ";
////    }
////    std::cout << std::endl;

////    std::cout << "constraint stance_c_2 has nodes: ";
////    for (int i : fake_stance_c_2->getNodes())
////    {
////        std::cout << i << " ";
////    }
////    std::cout << std::endl;

////    std::cout << "constraint flight_c_1 has nodes: ";
////    for (int i : fake_flight_c_1->getNodes())
////    {
////        std::cout << i << " ";
////    }
////    std::cout << std::endl;

////    std::cout << "parameter flight_p_1 has values: " << std::endl;
////    std::cout << fake_flight_p_1->getValues() << std::endl;

////    std::cout << "variable var_1 has bounds: " << std::endl;
////    std::cout << std::get<0>(fake_var_1->getBounds()) << std::endl;
////    std::cout << std::endl;

//////    std::chrono::duration<double> elapsed_time = std::chrono::system_clock::now() - start_time;
//////    std::cout << "elapsed time: " << elapsed_time.count() << std::endl;

//// bounds not set to zero

////    timeline_1->_shift_phases();

////    std::cout << "SHIFT ONCE: " << std::endl;
////    std::cout << std::get<0>(fake_var_1->getBounds()) << std::endl;
////    std::cout << std::endl;

////    timeline_1->_shift_phases();

////    std::cout << "SHIFT ONCE: " << std::endl;
////    std::cout << std::get<0>(fake_var_1->getBounds()) << std::endl;
////    std::cout << std::endl;

////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();
////    timeline_1->_shift_phases();

////    std::cout << "constraint stance_c_1 has nodes: ";
////    for (int i : fake_stance_c_1->getNodes())
////    {
////        std::cout << i << " ";
////    }
////    std::cout << std::endl;

////    std::cout << "constraint stance_c_2 has nodes: ";
////    for (int i : fake_stance_c_2->getNodes())
////    {
////        std::cout << i << " ";
////    }
////    std::cout << std::endl;

////    std::cout << "constraint flight_c_1 has nodes: ";
////    for (int i : fake_flight_c_1->getNodes())
////    {
////        std::cout << i << " ";
////    }
////    std::cout << std::endl;

////    std::cout << "parameter flight_p_1 has values: " << std::endl;
////    std::cout << fake_flight_p_1->getValues() << std::endl;

////    std::cout << "variable var_1 has bounds: " << std::endl;
////    std::cout << std::get<0>(fake_var_1->getBounds()) << std::endl;
////    std::cout << std::endl;
}
