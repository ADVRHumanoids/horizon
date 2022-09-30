#include "codegen_function.h"
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "wrapped_function.h"

namespace
{

class RestoreCwd
{
public:
    RestoreCwd(const char * cwd)
    {
        _cwd = cwd;
    }
    ~RestoreCwd()
    {
        chdir(_cwd);
        free((void *)_cwd);
    }
private:
    const char * _cwd;

};

bool check_function_consistency(const casadi::Function &f, const casadi::Function &g)
{
    casadi_utils::WrappedFunction fw = f;
    casadi_utils::WrappedFunction gw = g;

    std::vector<Eigen::VectorXd> input(f.n_in());

    for(int iter = 0; iter < 10; iter++)
    {

        for(int i = 0; i < f.n_in(); i++)
        {
            Eigen::VectorXd u;
            u.setRandom(f.size1_in(i));
            input[i] = 100*u;
        }

        for(int i = 0; i < f.n_in(); i++)
        {
            fw.setInput(i, input[i]);
            gw.setInput(i, input[i]);
        }

        fw.call();
        gw.call();

        for(int i = 0; i < f.n_out(); i++)
        {
            double err = (fw.getOutput(i) - gw.getOutput(i)).lpNorm<Eigen::Infinity>();

            if(err > 1e-16)
            {
                std::cout << "outputs of functions " << f.name() << " and " << g.name() <<
                             " differ by " << err << "\n";
                return false;
            }
        }

    }

    return true;
}

}



casadi::Function horizon::utils::codegen(const casadi::Function &f, std::string dir)
{    
    // save cwd
    RestoreCwd rcwd = get_current_dir_name();

    // make working directory
    mkdir(dir.c_str(), 0777);

    // cd to it
    if(chdir(dir.c_str()) != 0)
    {
        throw std::runtime_error("could not open codegen directory '" + dir + "': " + strerror(errno));
    }

    // serialize to string a compute hash
    auto f_str = f.serialize();
    std::size_t hash = std::hash<std::string>()(f_str);

    // check if .so already exists
    std::string fname = f.name() + "_generated_" + std::to_string(hash);

    if(access((fname + ".so").c_str(), F_OK) == 0)
    {
        std::cout << "exists: loading " << fname << "... \n";

        auto handle = dlopen(("./" + fname + ".so").c_str(), RTLD_NOW);

        if(!handle)
        {
            std::cout << "failed to load generated function " << fname <<
                         ": " << dlerror() << "\n";
            return f;
        }

        auto fext = casadi::external(f.name(),
                                     "./" + fname + ".so");

        if(!check_function_consistency(f, fext))
        {
            throw std::runtime_error("inconsistent generated function :(");
        }

        std::cout << "consistency check passed \n";

        return fext;
    }

    // else, generate and compile
    f.generate(fname + ".c");

    std::cout << "not found: compiling " << fname << "... \n";

    int ret = system(("clang -fPIC -shared -O2 " + fname + ".c -o " + fname + ".so").c_str());

    if(ret != 0)
    {
        std::cout << "failed to compile generated function " << fname << "\n";
        return f;
    }

    std::cout << "loading " << fname << "... \n";

    auto handle = dlopen(("./" + fname + ".so").c_str(), RTLD_NOW);

    if(!handle)
    {
        std::cout << "failed to load generated function " << fname <<
                     ": " << dlerror() << "\n";
        return f;
    }

    auto fext = casadi::external(f.name(),
                                 "./" + fname + ".so");

    if(!check_function_consistency(f, fext))
    {
        throw std::runtime_error("inconsistent generated function :(");
    }

    std::cout << "consistency check passed \n";

    return fext;

}
