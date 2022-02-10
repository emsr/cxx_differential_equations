#ifndef ODE_H
#define ODE_H 1

#include <vector>

namespace emsr
{

template<typename Deriv, typename StateVec, typename Real>
  StateVec
  runge_kutta_4(Deriv deriv, const StateVec& y, const StateVec& dydx,
        	Real x, Real h);

template<typename Deriv, typename StateVec, typename Real,
	 typename RealOutIter, typename StateVecOutIter>
  void
  step_runge_kutta(Deriv deriv, const StateVec& y1, Real x1, Real x2, int n_step,
                   RealOutIter x_tab, StateVecOutIter y_tab);

template<typename Deriv, typename StateVec, typename Real>
  void
  quad_runge_kutta(Deriv deriv, StateVec& y, StateVec& dydx, Real& x, Real h_try,
                   Real eps, StateVec & y_scale, Real& h_final, Real& h_next);

template<typename Deriv, typename StateVec, typename Real>
  void
  quad_cash_karp_rk(Deriv deriv, StateVec& y, StateVec& dydx, Real& x, Real h_try,
                    Real eps, StateVec& y_scale, Real& h_final, Real& h_next);

template<typename Deriv, typename StateVec, typename Real>
  void
  cash_karp_rk(Deriv deriv, StateVec& y, StateVec& dydx,
               Real x, Real h, StateVec& y_out, StateVec& y_err);

/**
 * Ordinary differential equation integrator.
 */
template<typename Deriv, typename StateVec, typename Real>
  class ode_integrator
  {
  public:

    ode_integrator(void (*stepper)(StateVec&, StateVec&, Real&, Real,
				   Real, Real&, Real&, Real&))
    : m_stepper(stepper)
    { }

    void integrate(Deriv deriv, StateVec y1, Real x1, Real x2,
		   Real eps, Real h1, Real hmin,
		   int& n_ok, int& n_bad);
  private:

    int m_max = 0;
    int m_count = 0;
    std::vector<Real> m_xp;
    std::vector<StateVec> m_yp;
    double m_dxsave = 0;
    void (*m_stepper)(StateVec&, StateVec&, Real&, Real,
		      Real, Real&, Real&, Real&);
  };

/**
 * Modified midpoint step.  At xs, input the dependent variable vector y,
 * and its derivative dydx.  Also input is h_tot, the total step to be made,
 * and n_step, the number of interior steps to be used.  The output is returned as 
 * y_out, which need not be distinct from y; if it is distinct
 * however, then y and dydx will be returned undamaged.  Derivs is the user-supplied
 * routine for calculating the right-hand side derivative.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  modified_midpoint(StateVec& y, StateVec& dydx, Real xs,
                    Real h_tot, int n_step, StateVec& y_out);

template<typename StateVec, typename Real>
  void
  stoermer(StateVec& y, StateVec& dy, StateVec& d2y, Real xs,
           Real h_tot, int n_step, StateVec& y_out, StateVec& dy_out);

template<typename Deriv, typename StateVec, typename Real>
  class BulirschStoer
  {
  public:

    BulirschStoer()
    : m_xx(KMAXX),
      m_yy(KMAXX)
    { }

    void step(Deriv deriv, const StateVec& y, const StateVec& dydx, Real& xx,
	      Real h_try, Real eps, StateVec& y_scale,
	      Real& h_final, Real& h_next);
  private:


    void
    m_rational_extrap(int i_est, Real x_est, StateVec& y_est,
		      StateVec& yz, StateVec& dy);

    void
    m_polynomial_extrap(int i_est, Real x_est, StateVec& y_est,
			StateVec& yz, StateVec& dy);

    static constexpr int KMAXX = 9;
    static constexpr int IMAXX = KMAXX + 1;
    static constexpr Real SAFE1 = 0.25;
    static constexpr Real SAFE2 = 0.7;
    static constexpr Real REDMAX = 1.0e-5;
    static constexpr Real REDMIN = 0.7;
    static constexpr Real TINY = 1.0e-30;
    static constexpr Real SCALEMAX = 0.1;

    bool m_first = true;
    int m_kmax;
    int m_kopt;
    static constexpr int m_nseq[IMAXX + 1] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 };

    Real m_epsold = -1.0;
    Real m_xnew;
    Real m_a[IMAXX + 1];
    Real m_alf[KMAXX + 1][KMAXX + 1];

    std::vector<Real> m_xx;
    std::vector<StateVec> m_yy;
  };

} // namespace emsr

#include <emsr/ode.tcc>

#endif  //  ODE_H
