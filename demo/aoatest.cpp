#include "AOAoptimizer.h"
#include "nucleotides.h"
#include <climits>

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
    result.value = value;
    return result;
}

int main(int argc, char const *argv[])
{
    AOAoptimizer<valueWithID, point, double>::args arg;
    arg.dimensionNum = 2;
    arg.optimizerNum= 90;
    arg.evaluateFunc = concave;
    arg.gFuncArgs = point();
    arg.iterNum= 1000;
    arg.lowerBounds= vector<double>{100.0,100.0};
    arg.upperBounds= vector<double>{-100.0, -100.0};
    AOAoptimizer<valueWithID, point, double> env(arg);
    valueWithID result = env.optimize();
    std::cout<<result.value<<std::endl;
    return 0;
}