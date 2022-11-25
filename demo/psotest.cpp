#include "PSOoptimizer.h"
#include <climits>
#include "nucleotides.h"
struct valueWithID
{
    double value;
    valueWithID()
    {
        value = -(__DBL_MAX__);
    }
    valueWithID& operator=(valueWithID other)
    {
        value = other.value;
        return *this;
    }
};

struct point
{
    vector<double> vars;
    point()
    {
        vars.resize(2,0);
    }
    point& operator=(point other)
    {
        vars = other.vars;
        return *this;
    }
};

valueWithID concave(point x)
{
    valueWithID result;
    double value = x.vars[0] * x.vars[0] + x.vars[1] * x.vars[1];
    result.value = -value;
    return result;
}

int main(int argc, char const *argv[])
{
    PSOoptimizer<valueWithID, point, double>::hArgs args;
    args.dimension = 2;
    args.dt = 0.001;
    args.ego = 15;
    args.evaluateFunc = concave;
    args.gFuncArgs = point();
    args.maxIter = 1000;
    args.omega = 0.95;
    args.particleNum = 90;
    args.rangeMax = vector<double>{100.0,100.0};
    args.rangeMin = vector<double>{-100.0, -100.0};
    args.vMax = vector<double>{10.0,10.0};
    args.vMin = vector<double>{0.0,0.0};
    PSOoptimizer<valueWithID, point, double> env(args);
    valueWithID result = env.run();
    std::cout<<result.value<<std::endl;
    return 0;
}
