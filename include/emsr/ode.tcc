#ifndef ODE_TCC
#define ODE_TCC 1

#include <cmath>
#include <stdexcept>

namespace emsr
{


/**
 * Given values for n dependent variables y and their derivatives dydx
 * known at x, use fourth-order Runge-Kutta method to advance the solution
 * over an interval h and return the incremented variables.
 *
 * @param deriv Function taking the dependent variable state vector and the independent variable
 *              and returning the state vector containing the derivative.
 */
template<typename Deriv, typename StateVec, typename Real>
  StateVec
  runge_kutta_4(Deriv deriv, const StateVec& y, const StateVec& dydx,
        	Real x, Real h)
  {
    auto h2 = h / 2;
    auto h6 = h / 6;
    auto xh = x + h2;

    auto yt = y + h2 * dydx;
    auto dyt = deriv(yt, xh);

    yt = y + h2 * dyt;
    auto dym = deriv(yt, xh);

    yt = y + h * dym;
    dym += dyt;

    dyt = deriv(yt, x + h);

    return y + h6 * (dydx + dyt + Real{2} * dym);
  }


/**
 * Starting from initial state vector y1 known at x1,
 * use 4th order Runge-Kutta to advance n_step equal increments to x2.
 * The user supplied routine deriv supplies derivatives.
 * The results are stored in the output iterators x_tab, y_tab
 * which must either be perallocated for n_step + 1 values or be accessed
 * through a back inserter.
 */
template<typename Deriv, typename StateVec, typename Real,
	 typename RealOutIter, typename StateVecOutIter>
  void
  step_runge_kutta(Deriv deriv, const StateVec& y1, Real x1, Real x2,
                    int n_step, RealOutIter x_tab, StateVecOutIter y_tab)
  {
    //  Load starting values of dependant variables.
    StateVec y = y1;
    y_tab[0] = y;
    x_tab[0] = x1;
    auto x = x1;
    auto h = (x2 - x1) / n_step;

    //  Take n_step steps in the independant variable x.
    for (int k = 1; k <= n_step; ++k)
      {
	auto y_out = runge_kutta_4(deriv, y, deriv(y, x), x, h);
	if (x + h == x)
          throw std::runtime_error("Step size too small in step_runge_kutta.");
	x += h;
	//  Store intermediate results.
	x_tab[k] = x;
	y = y_out;
	y_tab[k] = y;
      }
  }


template<typename StateVec, typename Real>
  Real
  max_error(const StateVec& y_temp, const StateVec& y_scale)
  {
    static constexpr Real TINY = 1.0e-30;
    auto errmax = TINY;
    for (int i = 0; i < y_temp.size(); ++i)
      {
	auto yt = std::abs(y_temp[i] / y_scale[i]);
	if (yt > errmax)
	  errmax = yt;
      }
    return errmax;
  }

/**
 * Fifth-order Runge-Kutta step with monitoring of local truncation error to ensure
 * accuracy and adjust stepsize.  Input are the dependent variable state vector y
 * and its derivative dydx at the starting value of the independent variable x.
 * Also input are the first guess for the stepsize h_try, the requred accuracy eps, and the
 * vector y_scale[1..n] against which the error is scaled independently for each dependent variable.
 * On output, x and y are replaced by thier new values, h_final is the stepsize which was actually
 * accomplished, and h_next is the estimated next stepsize.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  quad_runge_kutta(Deriv deriv, StateVec& y, StateVec& dydx, Real& x,
                   Real h_try, Real eps, StateVec& y_scale, Real& h_final,
		   Real& h_next)
  {
    constexpr Real POW_GROW = 10000;
    constexpr Real POW_SHRINK = 1.0e-30;
    constexpr Real F_CORR = 1.0 / 15.0;
    constexpr Real F_SAFETY = 0.9;
    constexpr Real ERR_COND = std::pow((4.0 / F_SAFETY), (1.0 / POW_GROW));

    auto dydx_save = dydx;
    auto y_save = y;
    auto x_save = x;

    StateVec y_temp;
    //  Set stepsize to the initial trial value.
    auto h = h_try;
    while (true)
      {
	//  Take two half steps.
	auto h2 = h / 2;
	y_temp = runge_kutta_4(y_save, dydx_save, x_save, h2);
	x = x_save + h2;
	dydx = deriv(y_temp, x);
	y = runge_kutta_4(y_temp, dydx, x, h2);
	x = x_save + h;
	if (x == x_save)
          throw std::runtime_error("Step size too small in quad_runge_kutta.");

	//  Take the large step.
	y_temp = runge_kutta_4(deriv, y_save, dydx_save, x_save, h);

	//  Evaluate accuracy.  Put the error estimate into y_temp.
	y_temp = y - y_temp;
	auto errmax = max_error(y_temp, y_scale);

	//  Scale relative to required tolerance.
	errmax /= eps;
	if (errmax <= 1.0)
          {
            //  Step succeeded.  Compute size of next step.
            h_final = h;
            h_next = (errmax > ERR_COND ? F_SAFETY * std::exp(POW_GROW * std::log(errmax)) : 4 * h);
            break;
          }
	//  Truncation error too large, reduce stepsize.
	h = F_SAFETY * h * std::exp(POW_SHRINK * std::log(errmax));
      }

    //  Mop up fifth order truncation error.
    y += F_CORR * y_temp;
  }



/**
 * Cash-Carp Runge-Kutta algorithm.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  cash_karp_rk(Deriv deriv, StateVec& y, StateVec& dydx,
               Real x, Real h, StateVec& y_out, StateVec& y_err)
  {
    static constexpr Real
      a2 = 0.2, a3 = 0.3, a4 = 0.6, a5 = 1.0, a6 = 0.875,
      b21 = 0.2,
      b31 = 3.0/40.0, b32 = 9.0/40.0,
      b41 = 0.3, b42 = -0.9, b43 = 1.2,
      b51 = -11.0/54.0, b52 = 2.5, b53 = -70.0/27.0, b54 = 35.0/27.0,
      b61 = 1631.0/55296.0, b62 = 175.0/512.0, b63 = 575.0/13824.0, b64 = 44275.0/110592.0, b65 = 253.0/4096.0,
      c1 = 37.0/378.0, c3 = 250.0/621.0, c4 = 125.0/594.0, c6 = 512.0/1771.0,
      dc5 = -277.0/14336.0;
    static constexpr Real dc1 = c1 - 2825.0/27648.0, dc3 = c3 - 18575.0/48384.0,
                            dc4 = c4 - 13525.0/55296.0, dc6 = c6 - 0.25;

    auto y_temp = y + h * b21 * dydx;

    auto ak2 = deriv(y_temp, x + a2 * h);
    y_temp = y + h * (b31 * dydx + b32 * ak2);

    auto ak3 = deriv(y_temp, x + a3 * h);
    y_temp = y + h * (b41 * dydx + b42 * ak2 + b43 * ak3);

    auto ak4 = deriv(y_temp, x + a4 * h);
    y_temp = y + h * (b51 * dydx + b52 * ak2 + b53 * ak3 + b54 * ak4);

    auto ak5 = deriv(y_temp, x + a5 * h);
    y_temp = y + h * (b61 * dydx + b62 * ak2 + b63 * ak3 + b64 * ak4 + b65 * ak5);

    auto ak6 = deriv(y_temp, x + a6 * h);
    y_out = y + h * (c1 * dydx + c3 * ak3 + c4 * ak4 + c6 * ak6);
    y_err = h * (dc1 * dydx + dc3 * ak3 + dc4 * ak4 + dc5 * ak5 + dc6 * ak6);
  }


/**
 * Fifth-order Runge-Kutta step with monitoring of local truncation error to ensure
 * accuracy and adjust stepsize.  Input are the dependent variable vector y
 * and its derivative dydx at the starting value of the independent variable x.
 * Also input are the first guess for the stepsize h_try, the requred accuracy eps, and the
 * vector y_scale against which the error is scaled independently for each dependent variable.
 * On output, x and y are replaced by thier new values, h_final is the stepsize which was actually
 * accomplished, and h_next is the estimated next stepsize.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  quad_cash_karp_rk(Deriv deriv, StateVec& y, StateVec& dydx, Real& x,
                    Real h_try, Real eps, StateVec& y_scale, Real& h_final,
		    Real& h_next)
  {
    constexpr Real POW_GROW = -0.20;
    constexpr Real POW_SHRINK = -0.25;
    constexpr Real F_SAFETY = 0.9;
    constexpr Real ERR_COND = std::pow(5 / F_SAFETY, 1 / POW_GROW);

    StateVec y_temp, y_err;
    Real errmax = 0;

    auto h = h_try;
    while (true)
      {
	cash_karp_rk(deriv, y, dydx, x, h, y_temp, y_err);
	errmax = max_error(y_err, y_scale);
	errmax /= eps;
	if (errmax <= 1.0)
          break;
	auto h_temp = F_SAFETY * h * std::pow(errmax, POW_SHRINK);
	h = (h > 0 ? std::max(h_temp, 0.1 * h) : std::min(h_temp, 0.1 * h));
	auto x_new= x + h;
	if (x_new == x)
          throw std::runtime_error("quad_cash_karp_rk: Stepsize underflow.");
      }

    if (errmax > ERR_COND)
      h_next = F_SAFETY * h * std::pow(errmax, POW_GROW);
    else
      h_next = 5 * h;
    x += h_final = h;

    y = y_temp;
  }


/**
 * ODE driver with adaptive stepsize control.  Integrate starting with values
 * y1 from x1 to x2 with accuracy eps, storing intermediate results
 * in global variables m_xp, m_yp, m_max, m_count, m_dxsave.  If m_max == 0 no intermediate results
 * will be stored and the pointers m_xp and m_yp can be set to zero.
 * h1 should be set as a first guess initial stepsize, hmin is the minimum stepsize (can be zero).
 * On output nok and nbad are the numbers of good and bad (but retried and fixed) steps taken.
 * y1 is replaced by stepped values at the end of the integration interval.
 * stepper is the name of the integration stepper to be used (e.g. quad_runge_kutta or bulirsch_stoer).
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  ode_integrator<Deriv, StateVec, Real>::
  integrate(Deriv deriv, StateVec y1, Real x1, Real x2,
	    Real eps, Real h1, Real hmin,
	    int& n_ok, int& n_bad)
  {
    Real xsave, h_next, h_final;

    constexpr int MAXSTEP = 10000;
    constexpr Real TINY = 1.0e-30;

    StateVec y_scale;
    StateVec y;
    StateVec dydx;

    auto x = x1;
    auto h = (x2 > x1) ? std::abs(h1) : -std::abs(h1);
    n_ok = n_bad = this->m_count = 0;
    y = y1;
    if (this->m_max > 0)
      xsave = x - 2 * this->m_dxsave;
    for (int n_step = 0; n_step < MAXSTEP; ++n_step)
     {
	dydx = deriv(y, x);
	y_scale = std::abs(y) + std::abs(dydx) + TINY;
	if (this->m_max)
          if (std::abs(x - xsave) > std::abs(this->m_dxsave))
            if (this->m_count < this->m_max - 1)
              {
        	this->m_xp[++this->m_count] = x;
        	this->m_yp[this->m_count] = y;
        	xsave = x;
              }
	if ((x + h - x2) * (x + h - x1) > 0.0)
          h = x2 - x;
	this->m_stepper(y, dydx, x, h, eps, y_scale, h_final, h_next);
	if (h_final == h)
          ++n_ok;
	else
          ++n_bad;
	if ((x - x2) * (x2 - x1) >= 0.0)
          {
            y1 = y; 
            if (this->m_max)
              {
        	this->m_xp[++this->m_count] = x;
        	this->m_yp[this->m_count] = y;
              }
            return;
          }
	if (std::abs(h_next) <= hmin)
          throw std::runtime_error("ode_integrate: Step size to small.");
	h = h_next;
    }
    throw std::runtime_error("ode_integrate: Too many steps in routine.");
  }


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
  modified_midpoint(Deriv deriv, StateVec& y, StateVec& dydx, Real xs,
                    Real h_tot, int n_step, StateVec& y_out)
  {
    auto h = h_tot / n_step;
    auto ym = y;
    auto yn = y + h * dydx;
    auto x = xs + h;
    y_out = deriv(yn, x);
    auto h2 = 2.0 * h;
    for (int n = 1; n < n_step; ++n)
      {
	ym = std::exchange(yn, ym + h2 * y_out);
	x += h;
	y_out = deriv(yn, x);
      }
    y_out = (ym + yn + h * y_out) / 2;
  }


/**
 * Stoermer's rule for integrating second order conservative systems of the form
 * y'' = f(x,y) for a system of n = nv/2 equations.  On input y[1..nv] contains
 * y in the first n elements and y' in the second n elements all evaluated at xs.
 * d2y[1..nv] contains the right hand side function f (also evaluated at xs) in
 * its first n elements (the second n elements are not referenced).  Also input
 * is h_tot, the total step to be taken and n_step, the number of substeps to be used.
 * The output is returned as y_out[1..nv], with the same storage arrangement as y.
 * derivvs is the user-supplied routine that calculates f.
 *
 * Nope: This routine can replace modified_midpoint above.
 * Originally, they folded y and y' into one vector to fake he same API as above.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  stoermer(Deriv deriv, StateVec& y, StateVec& dy, StateVec& d2y, Real xs,
           Real h_tot, int n_step, StateVec& y_out, StateVec& dy_out)
  {
    auto h = h_tot / n_step;
    auto hh = h / 2;

    auto dy_temp = h * (dy + hh * d2y);
    auto y_temp = y + dy_temp;

    auto x = xs + h;
    y_out = deriv(y_temp, x);
    auto h2 = 2 * h;
    for (int nn = 1; nn < n_step; ++nn)
      {
	dy_temp += h2 * y_out;
	y_temp += dy_temp;
	x += h;
	y_out = deriv(y_temp, x);
      }
    dy_out = y_temp / h + hh * y_out;
    y_out = y_temp;
  }


/**
 * Bulirsch-Stoer step with monitoring of local truncation error to ensure accuracy
 * and adjust stepsize.  Input are the dependent variables y and the derivatives dydx
 * at the starting value of the independent variable x.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  BulirschStoer<Deriv, StateVec, Real>::
  step(Deriv deriv, const StateVec& y, const StateVec& dydx, Real& xx,
       Real h_try, Real eps, StateVec& y_scale,
       Real& h_final, Real& h_next)
  {
    int km;
    bool exitflag = false;
    Real errmax, fact, red, scale, work, workmin, x_est;

    std::vector<Real> err(KMAXX);
    StateVec y_err;
    StateVec y_save;
    StateVec y_seq;

    if (eps != m_epsold)
      {
	h_next = m_xnew = -1.0e29;
	auto eps1 = SAFE1 * eps;
	m_a[0] = m_nseq[0] + 1;
	for (int k = 0; k < KMAXX; ++k)
          m_a[k + 1] = m_a[k] + m_nseq[k + 1];
	for (int iq = 1; iq < KMAXX; ++iq)
          for (int k = 1; k < iq; ++k) // Didn't touch!
            m_alf[k][iq] = std::pow(eps1, (m_a[k + 1] - m_a[iq + 1])
                                       / ((m_a[iq + 1] - m_a[1]) * (2 * k + 1)));
	m_epsold = eps;
	for (int m_kopt = 2; m_kopt < KMAXX; ++m_kopt)
          if (m_a[m_kopt + 1] > m_a[m_kopt] * m_alf[m_kopt - 1][m_kopt])
            break;
	m_kmax = m_kopt;
      }
    auto h = h_try;
    y_save = y;
    if (xx != m_xnew || h != h_next)
      {
	m_first = true;
	m_kopt = m_kmax;
      }
    bool reduct = false;
    int k = 0;
    while (true)
      {
	for (; k < m_kmax; ++k)
          {
            m_xnew = xx + h;
            if (m_xnew == xx)
              throw std::runtime_error("Step size underflow in BulirschStoer::step.");
            y_seq = modified_midpoint(deriv, y_save, dydx, xx, h, m_nseq[k]);
            x_est = dsqr(h / m_nseq[k]);
            m_polynomial_extrap(k, x_est, y_seq, y, y_err);
            if (k != 0)
              {
        	errmax = max_error(y_err, y_scale);
        	errmax /= eps;
        	km = k - 1;
        	err[km] = std::pow(errmax / SAFE1, 1.0 / (2 * k + 1));
        	if (k >= m_kopt - 1 || m_first)
        	  {
        	    if (errmax < 1.0)
                      {
                	exitflag = true;
                	break;
                      }
        	    if (k == m_kmax || k == m_kopt + 1)
                      {
                	red = SAFE2 / err[km];
                	break;
                      }
        	    else if (k == m_kopt && m_alf[m_kopt - 1][m_kopt] < err[km])
                      {
                	red = 1.0 / err[km];
                	break;
                      }
        	    else if (k == m_kopt && m_alf[km][m_kmax - 1] < err[km])
                      {
                	red = m_alf[km][m_kopt - 1] / err[km];
                	break;
                      }
        	  }
              }
          }
	if (exitflag)
          break;
	red = std::min(red, REDMIN);
	red = std::max(red, REDMAX);
	h *= red;
	reduct = true;
      }
    xx = m_xnew;
    h_final = h;
    m_first = false;
    workmin = 1.0e35;
    for (int kk = 0; kk < km; ++kk)
      {
	fact = std::max(err[kk], SCALEMAX);
	work = fact * m_a[kk + 1];
	if (work < workmin)
          {
            scale = fact;
            workmin = work;
            m_kopt = kk + 1;
          }
      }
    h_next = h / scale;
    if (m_kopt >= k && m_kopt != m_kmax && !reduct)
      {
	fact = std::max(scale / m_alf[m_kopt - 1][m_kopt], SCALEMAX);
	if (fact * m_a[m_kopt + 1] <= workmin)
          {
            h_next = h / fact;
            ++m_kopt;
          }
      }
  }


/**
 * Routine used by bulirsch_stoer to perform rational function extrapolation.
 * FIXME: These extrapolators don't depend on Deriv and could be a param.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  BulirschStoer<Deriv, StateVec, Real>::
  m_rational_extrap(int i_est, Real x_est, StateVec& y_est, 
		    StateVec& yz, StateVec& dy)
  {
    std::vector<Real> fx(i_est);

    this->m_xx[i_est] = x_est;
    if (i_est == 0)
      dy = this->m_yy[0] = yz = y_est;
    else
      {
	//  Evaluate next diagonal in the tableau.
	for (int k = 0; k < i_est; ++k)
	  fx[k + 1] = this->m_xx[i_est - k] / x_est;
	for (int j = 0; j < y_est.size(); ++j)
          {
            auto v = this->m_yy[0][j];
            this->m_yy[0][j] = y_est[j];
            auto yy = y_est[j];
            auto c = yy;
            Real ddy;
            for (int k = 1; k < i_est; ++k)
              {
        	auto b1 = fx[k] * v;
        	auto b = b1 - c;
        	//  Watch division by zero.
        	if (b)
        	  {
                    b = (c - v) / b;
                    ddy = c * b;
                    c = b1 * b;
        	  }
        	else
        	  ddy = v;
        	if (k != i_est)
        	  v = this->m_yy[k][j];
        	this->m_yy[k][j] = ddy;
        	yy += ddy;
              }
            dy[j] = ddy;
            yz[j] = yy;
          }
      }
  }


/**
 * Routine used by bulirsch_stoer to perform polynomial function extrapolation.
 * FIXME: These extrapolators don't depend on Deriv and could be a param.
 */
template<typename Deriv, typename StateVec, typename Real>
  void
  BulirschStoer<Deriv, StateVec, Real>::
  m_polynomial_extrap(int i_est, Real x_est, StateVec& y_est, 
		      StateVec& yz, StateVec& dy)
  {
    this->m_xx[i_est] = x_est;
    dy = yz = y_est;
    if (i_est == 0)
      this->m_yy[0] = y_est;
    else
      {
	auto c = y_est;
	for (int k = 0; k < i_est; ++k)
          {
            auto delta = Real{1} / (this->m_xx[i_est - k] - x_est);
            auto f1 = delta * x_est;
            auto f2 = delta * this->m_xx[i_est - k];
	    delta = c - std::exchange(this->m_yy[k], dy);
	    dy = f1 * delta;
	    c = f2 * delta;
	    yz += dy;
          }
	this->m_yy[i_est] = dy;
      }
  }

} // namespace emsr

#endif  //  ODE_TCC
