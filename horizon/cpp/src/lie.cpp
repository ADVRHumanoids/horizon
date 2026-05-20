#include "lie.h"

std::string replaceSubstring(const std::string& original, const std::string& toReplace, const std::string& replacement)
{
    std::string result = original;
    size_t pos = 0;

    // Find and replace all occurrences of `toReplace` with `replacement`
    while ((pos = result.find(toReplace, pos)) != std::string::npos) {
        result.replace(pos, toReplace.length(), replacement);
        pos += replacement.length(); // Move past the replacement
    }

    return result;
}

casadi::Function horizon::lie::jacobian(casadi::Function f,
                                        casadi::Function xsum,
                                        casadi::Function xdiff,
                                        std::vector<std::string> out,
                                        std::string name,
                                        bool lie_output)
{
    if(xsum.is_null() != xdiff.is_null())
    {
        throw std::invalid_argument("xsum and xdiff must be either both valid or null");
    }

    // euclidean case
    if(xsum.is_null())
    {
        return f.factory(name,
                         f.name_in(),
                         out);
    }

    // lie derivative
    // F(x, u, dx) = f(x [+] dx, u) or, if output is in lie group, =>
    // F(x, u, dx) = f(x [+] dx, u) [-] f(x, 0, 0, ...)
    // A(x, u) = dF/ddx | dx = 0
    // B(x, u) = dF/du  | dx = 0

    // create input variables for f(x, u, p1, p2, ...)
    std::vector<casadi::MX> var_in;
    for(int i = 0; i < f.n_in(); i++)
    {
        auto v = casadi::MX::sym(f.name_in(i), f.size1_in(i));
        var_in.push_back(v);
    }

    // x if var[0]
    auto x = var_in[0];

    // tangent space dim
    const int ndx = xdiff.size1_out(0);

    // dx
    auto dx = casadi::MX::sym("dx", ndx);

    // x [+] dx
    auto x_plus_dx = xsum(std::vector{x, dx})[0];

    // y = f(x [+] dx, u, ...)
    std::vector<casadi::MX> y_var_in = var_in;
    y_var_in[0] = x_plus_dx;

    auto y = f(y_var_in);

    // if needed, y = f(x [+] dx, u, ...) [-] f(x, 0, 0, ...)
    if(lie_output)
    {
        auto y0_var_in = var_in;

        for(int i = 1; i < f.n_in(); i++)
        {
            y0_var_in[i] = casadi::MX::zeros(f.size1_in(i));
        }

        auto y0 = f(y0_var_in)[0];
        y[0] = xdiff(std::vector{y[0], y0})[0];
    }

    // add dx as input variable
    auto F_var_in = var_in;
    F_var_in.push_back(dx);

    auto F_name_in = f.name_in();
    F_name_in.push_back("dx");

    // create F and compute dF/ddx, dF/du
    casadi::Function F("F", F_var_in, y, F_name_in, f.name_out());

    // replace derivatives w.r.t. x with dx
    for(auto& o : out)
    {
        o = replaceSubstring(o, ":x", ":dx");
    }

    auto dF = F.factory("dF", F_name_in, out);

    // evaluate at dx = 0
    F_var_in.back() = casadi::MX::zeros(ndx);

    auto jac_A_B = dF(F_var_in);

    // replace output names
    auto dF_name_out = dF.name_out();

    for(auto& o : dF_name_out)
    {
        o = replaceSubstring(o, "_dx", "_x");
    }

    return casadi::Function(name,
                            var_in,
                            jac_A_B,
                            f.name_in(),
                            dF_name_out);

}
