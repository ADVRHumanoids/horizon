#include "horizon_parser.h"

int main()
{

    int dim = 10;
    int n_nodes = 34;
    auto v1 = std::make_shared<horizon::Variable>("v1", dim, n_nodes);
    auto v2 = std::make_shared<horizon::Variable>("v2", dim, n_nodes);

    auto casadi_fun = casadi::Function("f", {v1->getSym(), v2->getSym()}, {v1->getSym() + v2->getSym()});

    auto fun = std::make_shared<horizon::Constraint>(casadi_fun, n_nodes);



    std::vector<int> nodes;
    for (int i = 0; i <= 43; ++i)
    {
        nodes.push_back(i);
    }

    v1->setNodes(nodes, true);

    std::cout << v1->getNodes() << std::endl;
    std::cout << v1->getLowerBounds() << std::endl;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(nodes.begin(), nodes.end(), gen);

    std::vector<int> result(nodes.begin(), nodes.begin() + 10);
    std::sort(result.begin(), result.end());


    fun->setNodes(result, true);

    std::cout << fun->getNodes() << std::endl;
    std::cout << fun->getLowerBounds() << std::endl;


}
