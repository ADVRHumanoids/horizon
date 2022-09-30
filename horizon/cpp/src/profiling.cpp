#include "profiling.h"

using namespace horizon::utils;

Timer::Timer(const char *name, TocCallback& cb):
    _on_toc(cb),
    _name(name),
    _t0(hrc::now()),
    _done(false)
{
}

Timer::Timer(std::string name, Timer::TocCallback &cb):
    _on_toc(cb),
    _name_str(name),
    _t0(hrc::now()),
    _done(false),
    _name(_name_str.c_str())
{
}

void Timer::toc()
{
    if(_done) return;

    double usec = (hrc::now() - _t0).count() * 1e-3;
    _on_toc(_name, usec);
    _done = true;
}

Timer::~Timer()
{
    toc();
}
