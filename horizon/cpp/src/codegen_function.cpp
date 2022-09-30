#include "codegen_function.h"
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

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
        std::cout << "loading " << fname << "... \n";

        auto handle = dlopen(("./" + fname + ".so").c_str(), RTLD_NOW);

        if(!handle)
        {
            std::cout << "failed to load generated function " << fname <<
                         ": " << dlerror() << "\n";
            return f;
        }

        return casadi::external(f.name(),
                                "./" + fname + ".so");
    }

    // else, generate and compile
    f.generate(fname + ".c");

    std::cout << "compiling " << fname << "... \n";

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

    return casadi::external(f.name(),
                            "./" + fname + ".so");

}
