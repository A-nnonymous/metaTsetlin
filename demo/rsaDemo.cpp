#include "RSAoptimizer.h"
#include <climits>
#include "nucleotides.h"

/// @brief Correlated to template argument "output", must contain member "value"
struct utility
{
    double value;
    double x;
    double y;
    utility()
    {
        value = -(__DBL_MAX__);
    }
    utility(double valueIn)
    {
        value = valueIn;
    }
};

/// @brief Correlated to template argument "funcArgs", must contain member vector "vars" in type of <rangDtype>.
struct point
{
    std::vector<double> vars;
    point()
    {
        vars.resize(2,0);
    }
    point(double xin, double yin)
    {
        vars.resize(2,0);
        vars[0] = xin;
        vars[1] = yin;
    }
};

utility concave(point p)
{
    double num = p.vars[0]*p.vars[0] + p.vars[1]*p.vars[1];
    //double num =0;
    auto result = utility(num);
    result.x = p.vars[0];
    result.y = p.vars[1];
    return result;
}


int main(int argc, char const *argv[])
{

    std::vector<double> min{-10,-10};
    std::vector<double> max{10,10};
    auto limits = Predator<utility,point,double>::rangeLimits(min,max);
    auto search = Predator<utility,point,double>::searchArgs(2,200,0.1,0.005,limits);
    auto test = RSAoptimizer<utility,point,double>(50,concave,point(), search);
    auto result = test.optimize();
    test.snapShot();
    std::cout<<"function:f(x,y) = (x^2 + y^2) is optimized to: "<< result.value<<std::endl;
    std::cout<<"With x = "<<result.x <<" , y = "<< result.y << std::endl;
    return 0;
}
