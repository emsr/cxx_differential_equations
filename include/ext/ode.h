#ifndef ODE_H
#define ODE_H 1

#include <vector>

template<typename _Deriv, typename _StateVec, typename _Real>
  _StateVec
  runge_kutta_4(_Deriv deriv, const _StateVec& y, const _StateVec& dydx,
        	_Real x, _Real h);

template<typename _Deriv, typename _StateVec, typename _Real,
	 typename _RealOutIter, typename _StateVecOutIter>
  void
  step_runge_kutta(_Deriv deriv, const _StateVec& y1, _Real x1, _Real x2, int n_step,
                   _RealOutIter x_tab, _StateVecOutIter y_tab);

template<typename _Deriv, typename _StateVec, typename _Real>
  void
  quad_runge_kutta(_Deriv deriv, _StateVec& y, _StateVec& dydx, _Real& x, _Real h_try,
                   _Real eps, _StateVec & y_scale, _Real& h_final, _Real& h_next);

template<typename _Deriv, typename _StateVec, typename _Real>
  void
  quad_cash_karp_rk(_Deriv deriv, _StateVec& y, _StateVec& dydx, _Real& x, _Real h_try,
                    _Real eps, _StateVec& y_scale, _Real& h_final, _Real& h_next);

template<typename _Deriv, typename _StateVec, typename _Real>
  void
  cash_karp_rk(_Deriv deriv, _StateVec& y, _StateVec& dydx,
               _Real x, _Real h, _StateVec& y_out, _StateVec& y_err);

/**
 * Ordinary differential equation integrator.
 */
template<typename _Deriv, typename _StateVec, typename _Real>
  class ode_integrator
  {
  public:

    ode_integrator(void (*stepper)(_StateVec&, _StateVec&, _Real&, _Real,
				   _Real, _Real&, _Real&, _Real&))
    : m_stepper(stepper)
    { }

    void integrate(_Deriv deriv, _StateVec y1, _Real x1, _Real x2,
		   _Real eps, _Real h1, _Real hmin,
		   int& n_ok, int& n_bad);
  private:

    int m_max = 0;
    int m_count = 0;
    std::vector<_Real> m_xp;
    std::vector<_StateVec> m_yp;
    double m_dxsave = 0;
    void (*m_stepper)(_StateVec&, _StateVec&, _Real&, _Real,
		      _Real, _Real&, _Real&, _Real&);
  };

/**
 * Modified midpoint step.  At xs, input the dependent variable vector y,
 * and its derivative dydx.  Also input is h_tot, the total step to be made,
 * and n_step, the number of interior steps to be used.  The output is returned as 
 * y_out, which need not be distinct from y; if it is distinct
 * however, then y and dydx will be returned undamaged.  Derivs is the user-supplied
 * routine for calculating the right-hand side derivative.
 */
template<typename _Deriv, typename _StateVec, typename _Real>
  void
  modified_midpoint(_StateVec& y, _StateVec& dydx, _Real xs,
                    _Real h_tot, int n_step, _StateVec& y_out);

template<typename _StateVec, typename _Real>
  void
  stoermer(_StateVec& y, _StateVec& dy, _StateVec& d2y, _Real xs,
           _Real h_tot, int n_step, _StateVec& y_out, _StateVec& dy_out);

template<typename _Deriv, typename _StateVec, typename _Real>
  class BulirschStoer
  {
  public:

    BulirschStoer()
    : m_xx(KMAXX),
      m_yy(KMAXX)
    { }

    void step(_Deriv deriv, const _StateVec& y, const _StateVec& dydx, _Real& xx,
	      _Real h_try, _Real eps, _StateVec& y_scale,
	      _Real& h_final, _Real& h_next);
  private:


    void
    m_rational_extrap(int i_est, _Real x_est, _StateVec& y_est,
		      _StateVec& yz, _StateVec& dy);

    void
    m_polynomial_extrap(int i_est, _Real x_est, _StateVec& y_est,
			_StateVec& yz, _StateVec& dy);

    static constexpr int KMAXX = 9;
    static constexpr int IMAXX = KMAXX + 1;
    static constexpr _Real SAFE1 = 0.25;
    static constexpr _Real SAFE2 = 0.7;
    static constexpr _Real REDMAX = 1.0e-5;
    static constexpr _Real REDMIN = 0.7;
    static constexpr _Real TINY = 1.0e-30;
    static constexpr _Real SCALEMAX = 0.1;

    bool m_first = true;
    int m_kmax;
    int m_kopt;
    static constexpr int m_nseq[IMAXX + 1] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 };

    _Real m_epsold = -1.0;
    _Real m_xnew;
    _Real m_a[IMAXX + 1];
    _Real m_alf[KMAXX + 1][KMAXX + 1];

    std::vector<_Real> m_xx;
    std::vector<_StateVec> m_yy;
  };

#include <ext/ode.tcc>

#endif  //  ODE_H
