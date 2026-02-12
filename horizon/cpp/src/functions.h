#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "variables.h"

namespace horizon
{

class Function
{
 public:

     typedef std::shared_ptr<Function> Ptr;

     Function(casadi::Function fun, int n_nodes);

     virtual bool setNodes(std::vector<int> nodes, bool erasing) = 0;


     std::vector<int> getNodes();
     std::string getName();
     int getDim();
     casadi::Function getFunction();

 protected:

     int _n_nodes;
     std::string _name;
     std::vector<int> _nodes;
     int _dim;
     casadi::Function _fun;

 };

class Constraint: public Function
{
 public:

    typedef std::shared_ptr<Constraint> Ptr;

    Constraint(casadi::Function fun, int n_nodes);

    bool setNodes(std::vector<int> nodes, bool erasing) override;

    bool setLowerBounds(Eigen::MatrixXd lb, std::vector<int> nodes = {});
    bool setUpperBounds(Eigen::MatrixXd ub, std::vector<int> nodes = {});

    bool setBounds(Eigen::MatrixXd lb, Eigen::MatrixXd ub, std::vector<int> nodes = {});


    Eigen::MatrixXd getLowerBounds();
    Eigen::MatrixXd getUpperBounds();
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getBounds();

private:

    horizon::Bounds::Ptr _bounds;


};

class Cost: public Function
{
 public:

    typedef std::shared_ptr<Cost> Ptr;

    Cost(casadi::Function fun, int n_nodes);

    bool setNodes(std::vector<int> nodes, bool erasing) override;

};

class Residual: public Function
{
 public:

    typedef std::shared_ptr<Residual> Ptr;

    Residual(casadi::Function fun, int n_nodes);

    bool setNodes(std::vector<int> nodes, bool erasing) override;

};


}
#endif // FUNCTIONS_H
