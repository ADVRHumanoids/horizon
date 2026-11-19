#ifndef PROFILING_H
#define PROFILING_H

#include <chrono>
#include <functional>
#include <map>
#include <vector>
#include <string>

namespace horizon { namespace utils {


struct Timer
{
    typedef std::chrono::high_resolution_clock hrc;

    typedef std::function<void(const char *, double)> TocCallback;

    Timer(const char* name, TocCallback& cb);

    Timer(std::string name, TocCallback& cb);

    void toc();

    ~Timer();

private:

    std::string _name_str;
    const char * _name;
    hrc::time_point _t0;
    bool _done;
    TocCallback& _on_toc;

};

struct ProfilingInfo
{
    std::map<std::string, std::vector<double>> timings;
};


} }


#ifdef HORIZON_PROFILING
#define TIC(name) ::horizon::utils::Timer t_##name(#name, on_timer_toc);
#define TOC(name) t_##name.toc();
#else
#define TIC(name)
#define TOC(name)
#endif

#endif // PROFILING_H
