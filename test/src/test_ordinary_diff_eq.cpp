
#include <iostream>
#include <iomanip>

#include <emsr/ode.h>
#include <test_functions.h>

int
main()
{
  linear<double> lin;
  double x_0 = 3.0;
  double h = 5.0;

  //emsr::runge_kutta_4([lin](double x)->auto{return lin.rhs(x);}, lin.f, lin.dfdx, x_0, h);
  //emsr::runge_kutta_4(&lin.rhs, lin.f, lin.dfdx, x_0, h);
}
