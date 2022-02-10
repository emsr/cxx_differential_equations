/*
$HOME/bin/bin/g++ -std=c++17 -DSTANDALONE -o test_ode test_functions.h
*/

#include <array>
#include <cmath>

/**
 * RHS for f = 2.
 * Solution: y = 2 * x + x_0
 */
template<typename Tp>
  struct linear
  {
    static constexpr std::size_t dimension = 1;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp);
    void jacobian(Tp);
  };

template<typename Tp>
  void
  linear<Tp>::rhs(Tp)
  {
    f[0] = Tp{2};
  }

template<typename Tp>
  void
  linear<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = Tp{0};

    dfdx[0] = Tp{0};
  }

/**
 * RHS for f = y.
 * Solution: y = exp(x) with y_0 = 1.
 */
template<typename Tp>
  struct exptest
  {
    static constexpr std::size_t dimension = 1;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp);
    void jacobian(Tp);
  };

template<typename Tp>
  void
  exptest<Tp>::rhs(Tp)
  {
    f[0] = y[0];
  }

template<typename Tp>
  void
  exptest<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = y[0];

    dfdx[0] = Tp{0};
  }

/**
 * Derivative change at x=0, small derivative
 */
template<typename Tp>
  struct stepfn
  {
    static constexpr std::size_t dimension = 1;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp);
    void jacobian(Tp);
  };

template<typename Tp>
  void
  stepfn<Tp>::rhs(Tp x)
  {
    if (x >= Tp{1})
      f[0] = Tp{1};
    else
      f[0] = Tp{0};
  }

template<typename Tp>
  void
  stepfn<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = Tp{0};

    dfdx[0] = Tp{0};
  }

template<typename Tp>
  struct sintest
  {
    static constexpr std::size_t dimension = 2;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp);
    void jacobian(Tp);
  };

/**
 * RHS for f_0 = -y_1, f_1 = y_0
 * equals y = [cos(x), sin(x)] with initial values [1, 0]
 */
template<typename Tp>
  void
  sintest<Tp>::rhs(Tp)
  {
    f[0] = -y[1];
    f[1] = y[0];
  }

template<typename Tp>
  void
  sintest<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = Tp{0};
    dfdy[0][1] = Tp{-1};
    dfdy[1][0] = Tp{1};
    dfdy[1][1] = Tp{0};

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
  }

template<typename Tp>
  class xsin
  {
  public:

    static constexpr std::size_t dimension = 2;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp);
    void jacobian(Tp);

  private:

    static constexpr Tp NaN = std::numeric_limits<Tp>::quiet_NaN();

    int n = 0;
    int m = 0;

    bool rhs_reset = false;
    bool jacobian_reset = false;
  };

/**
 * Sine/cosine with random failures
 */
template<typename Tp>
  void
  xsin<Tp>::rhs(Tp)
  {
    if (rhs_reset)
      {
	rhs_reset = false;
	n = 0;
	m = 1;
      }
    ++n;

    if (n >= m)
      { 
	m = n * 1.3;
	return; 
      }

    if (n > 40 && n < 65)
      {
	f[0] = NaN;
	f[1] = NaN;

	return;
      }

    f[0] = -y[1];
    f[1] = y[0];
  }

template<typename Tp>
  void
  xsin<Tp>::jacobian(Tp)
  {
    if (jacobian_reset)
      {
	jacobian_reset = false;
	n = 0;
      }
    ++n;

    if (n > 50 && n < 55)
      {
	dfdy[0][0] = NaN;
	dfdy[0][1] = NaN;
	dfdy[1][0] = NaN;
	dfdy[1][1] = NaN;

	dfdx[0] = NaN;
	dfdx[1] = NaN;

	return;
      }

    dfdy[0][0] = Tp{0};
    dfdy[0][1] = Tp{-1};
    dfdy[1][0] = Tp{1};
    dfdy[1][1] = Tp{0};

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
  }

/**
 * RHS for classic stiff example
 *  dy_0 / dt =  998 * y_0 + 1998 * y_1    y_0(0) = 1
 *  dy_1 / dt = -999 * y_0 - 1999 * y_1    y_1(0) = 0
 *
 *  Solution is
 *  y_0 = 2 * \exp(-x) - \exp(-1000 x)
 *  y_1 = - \exp(-x) + \exp(-1000 x)
 */
template<typename Tp>
  struct stiff
  {
    static constexpr std::size_t dimension = 2;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);
  };

template<typename Tp>
  void
  stiff<Tp>::rhs(Tp)
  {
    f[0] = Tp{998} * y[0] + Tp{1998} * y[1];
    f[1] = -Tp{999} * y[0] - Tp{1999} * y[1];
  }

template<typename Tp>
  void
  stiff<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = Tp{998};
    dfdy[0][1] = Tp{1998};
    dfdy[1][0] = Tp{-999};
    dfdy[1][1] = Tp{-1999};

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
  }

/**
 * van Der Pol oscillator:
 *  f_0 = y_1                             y_0(0) = 1
 *  f_1 = -y_0 + \mu * y_1 * (1 - y_0^2)  y_1(0) = 0
 */
template<typename Tp>
  class vanderpol
  {
  public:

    static constexpr std::size_t dimension = 2;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);

  private:

    static constexpr Tp mu = Tp{10};
  };

template<typename Tp>
  void
  vanderpol<Tp>::rhs(Tp)
  {
    f[0] = y[1];
    f[1] = -y[0] + mu * y[1] * (Tp{1} - y[0] * y[0]); 
  }

template<typename Tp>
  void
  vanderpol<Tp>::jacobian(Tp)
  {

    dfdy[0][0] = Tp{0};
    dfdy[0][1] = Tp{1};
    dfdy[1][0] = Tp{-2} * mu * y[0] * y[1] - Tp{1};
    dfdy[1][1] = mu * (Tp{1} - y[0] * y[0]);

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
  }

/**
 * The Oregonator - chemical Belusov-Zhabotinskii reaction 
 *  y_0(0) = 1
 *  y_1(0) = 2
 *  y_2(0) = 3
 */
template<typename Tp>
  class oregonator
  {
  public:

    static constexpr std::size_t dimension = 3;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);

  private:

    static constexpr Tp c1 = Tp{77.27};
    static constexpr Tp c2 = Tp{8.375e-6};
    static constexpr Tp c3 = Tp{0.161};
  };

template<typename Tp>
  void
  oregonator<Tp>::rhs(Tp)
  {
    f[0] = c1 * (y[1] + y[0] * (1 - c2 * y[0] - y[1]));
    f[1] = (y[2] - y[1] * (1 + y[0])) / c1;
    f[2] = c3 * (y[0] - y[2]);
  }

template<typename Tp>
  void
  oregonator<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = c1 * (1 - 2 * c2 * y[0] - y[1]);
    dfdy[0][1] = c1 * (1 - y[0]);
    dfdy[0][2] = Tp{0};

    dfdy[1][0] = 1 / c1 * (-y[1]);
    dfdy[1][1] = 1 / c1 * (-1 - y[0]);
    dfdy[1][2] = 1 / c1;

    dfdy[2][0] = c3;
    dfdy[2][1] = Tp{0};
    dfdy[2][2] = -c3;

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
    dfdx[2] = Tp{0};
  }

/**
 * Volterra-Lotka predator-prey model:
 *  f_0 = (a - b y_1) y_0     y_0(0) = 3
 *  f_1 = (-c + d y_0) y_1    y_1(0) = 1
 */
template<typename Tp>
  class volterra_lotka
  {
  public:

    static constexpr std::size_t dimension = 2;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);

  private:

    static constexpr Tp a = Tp{-1};
    static constexpr Tp b = Tp{-1};
    static constexpr Tp c = Tp{-2};
    static constexpr Tp d = Tp{-1};
  };

template<typename Tp>
  void
  volterra_lotka<Tp>::rhs(Tp)
  {
    f[0] = (a - b * y[1]) * y[0];
    f[1] = (-c + d * y[0]) * y[1];
  }

template<typename Tp>
  void
  volterra_lotka<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = a - b * y[1];
    dfdy[0][1] = -b * y[0];
    dfdy[1][0] = d * y[1];
    dfdy[1][1] = -c + d * y[0];

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
  }

/**
 * Stiff trigonometric example 
 *
 *  f_0 = -50 * (y_0 - cos(x))    y_0(0) = 0
 */
template<typename Tp>
  class stifftrig
  {
  public:
    static constexpr std::size_t dimension = 1;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);
  };

template<typename Tp>
  void
  stifftrig<Tp>::rhs(Tp x)
  {
    f[0] = Tp{-50} * (y[0] - std::cos(x));
  }

template<typename Tp>
  void
  stifftrig<Tp>::jacobian(Tp x)
  {
    dfdy[0][0] = Tp{-50};

    dfdx[0] = Tp{-50} * std::sin(x);
  }

/**
 * E5 - a stiff badly scaled chemical problem by Enright, Hull & Lindberg (1975):
 * Comparing numerical methods for stiff systems of
 * ODEs. BIT, vol. 15, pp. 10-48.
 *
 *  f_0 = -a * y_0 - b * y_0 * y_2                               y_0(0) = 1.76e-3
 *  f_1 = a * y_0 - m * c * y_1 * y_2                            y_1(0) = 0.0
 *  f_2 = a * y_0 - b * y_0 * y_2 - m * c * y_1 * y_2 + c * y_3  y_2(0) = 0.0
 *  f_3 = b * y_0 * y_2 - c * y_3                                y_3(0) = 0.0
 */
template<typename Tp>
  class e5
  {
  public:

    static constexpr std::size_t dimension = 4;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);

  private:

    static constexpr Tp a = 7.89e-10;
    static constexpr Tp b = 1.1e7;
    static constexpr Tp c = 1.13e3;
    static constexpr Tp m = 1.0e6;
  };

template<typename Tp>
  void
  e5<Tp>::rhs(Tp)
  {
    f[0] = -a * y[0] - b * y[0] * y[2];
    f[1] = a * y[0] - m * c * y[1] * y[2];
    f[3] = b * y[0] * y[2] - c * y[3];
    f[2] = f[1] - f[3];
  }

template<typename Tp>
  void
  e5<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = -a - b * y[2];
    dfdy[0][1] = Tp{0};
    dfdy[0][2] = -b * y[0];
    dfdy[0][3] = Tp{0};

    dfdy[1][0] = a;
    dfdy[1][1] = -m * c * y[2];
    dfdy[1][2] = -m * c * y[1];
    dfdy[1][3] = Tp{0};

    dfdy[2][0] = a - b * y[2];
    dfdy[2][1] = -m * c * y[2];
    dfdy[2][2] = -b * y[0] - m * c * y[1];
    dfdy[2][3] = c;

    dfdy[3][0] = b * y[2];
    dfdy[3][1] = Tp{0};
    dfdy[3][2] = b * y[0];
    dfdy[3][3] = -c;

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
    dfdx[2] = Tp{0};
    dfdx[3] = Tp{0};
  }

/**
 * Chemical reaction system of H.H. Robertson (1966): The solution of
 * a set of reaction rate equations. In: J. Walsh, ed.: Numer.
 * Anal., an Introduction, Academ. Press, pp. 178-182.
 *
 *  f_0 = -a * y_0 + b * y_1 * y_2              y_0(0) = 1
 *  f_1 = a * y_0 - b * y_1 * y_2 - c * y_1^2   y_1(0) = 0
 *  f_2 = c * y_1^2                             y_2(0) = 0
 */
template<typename Tp>
  class robertson
  {
  public:

    static constexpr std::size_t dimension = 3;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);

  private:

    static constexpr Tp a = Tp{0.04};
    static constexpr Tp b = Tp{1.0e4};
    static constexpr Tp c = Tp{3.0e7};
  };

template<typename Tp>
  void
  robertson<Tp>::rhs(Tp)
  {
    f[0] = -a * y[0] + b * y[1] * y[2];
    f[2] = c * y[1] * y[1];
    f[1] = -f[0] - f[2];
  }

template<typename Tp>
  void
  robertson<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = -a;
    dfdy[0][1] = b * y[2];
    dfdy[0][2] = b * y[1];

    dfdy[1][0] = a;
    dfdy[1][1] = -b * y[2] - 2 * c * y[1];
    dfdy[1][2] = -b * y[1];

    dfdy[2][0] = Tp{0};
    dfdy[2][1] = 2 * c * y[1];
    dfdy[2][2] = Tp{0};

    dfdx[0] = Tp{0};
    dfdx[1] = Tp{0};
    dfdx[2] = Tp{0};
  }

/**
 * A two-dimensional oscillating Brusselator system.
 *
 * f0 = a + y0^2 * y1 - (b + 1) * y0      y0(0) = 1.5
 * f1 = b * y0 - y0^2 * y1                y1(0) = 3.0
 */
template<typename Tp>
  class brusselator
  {
  public:

    static constexpr std::size_t dimension = 2;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);

  private:

    static constexpr Tp a = 1;
    static constexpr Tp b = 3;
  };

template<typename Tp>
  void
  brusselator<Tp>::rhs(Tp)
  {
    f[0] = a + y[0] * y[0] * y[1] - (b + 1.0) * y[0];
    f[1] = b * y[0] - y[0] * y[0] * y[1];
  }

template<typename Tp>
  void
  brusselator<Tp>::jacobian(Tp)
  {
    dfdy[0][0] = 2 * y[0] * y[1] - (b + 1.0);
    dfdy[0][1] = y[0] * y[0];

    dfdy[1][0] = b - 2 * y[0] * y[1];
    dfdy[1][1] = -y[0] * y[0];

    dfdx[0] = 0;
    dfdx[1] = 0;
  }

/**
 * Ring Modulator, stiff ODE of dimension 15.
 *
 * Reference: Walter M. Lioen, Jacques J.B. de Swart, Test Set for
 * Initial Value Problem Solvers, Release 2.1 September 1999,
 * http://ftp.cwi.nl/IVPtestset/software.htm
 */

template<typename Tp>
  class ring_modulator
  {
  public:

    static constexpr std::size_t dimension = 15;

    std::array<Tp, dimension> y;
    std::array<Tp, dimension> f;
    std::array<Tp, dimension> dfdx;
    std::array<std::array<Tp, dimension>, dimension> dfdy;

    void rhs(Tp x);
    void jacobian(Tp x);

  private:

    static constexpr Tp pi = 3.141592653589793238462643383L;

    static constexpr Tp c = 1.6e-8;
    static constexpr Tp cs = 2e-12;
    static constexpr Tp cp = 1e-8;
    static constexpr Tp r = 25e3;
    static constexpr Tp rp = 50e0;
    static constexpr Tp lh = 4.45e0;
    static constexpr Tp ls1 = 2e-3;
    static constexpr Tp ls2 = 5e-4;
    static constexpr Tp ls3 = 5e-4;
    static constexpr Tp rg1 = 36.3;
    static constexpr Tp rg2 = 17.3;
    static constexpr Tp rg3 = 17.3;
    static constexpr Tp ri = 5e1;
    static constexpr Tp rc = 6e2;
    static constexpr Tp gamma = 40.67286402e-9;
    static constexpr Tp delta = 17.7493332;
  };

template<typename Tp>
  void
  ring_modulator<Tp>::rhs(Tp x)
  {
    const auto uin1 = 0.5 * std::sin(2e3 * pi * x);
    const auto uin2 = 2 * std::sin(2e4 * pi * x);
    const auto ud1 = +y[2] - y[4] - y[6] - uin2;
    const auto ud2 = -y[3] + y[5] - y[6] - uin2;
    const auto ud3 = +y[3] + y[4] + y[6] + uin2;
    const auto ud4 = -y[2] - y[5] + y[6] + uin2;

    const auto qud1 = gamma * (std::exp(delta * ud1) - 1.0);
    const auto qud2 = gamma * (std::exp(delta * ud2) - 1.0);
    const auto qud3 = gamma * (std::exp(delta * ud3) - 1.0);
    const auto qud4 = gamma * (std::exp(delta * ud4) - 1.0);

    f[0] = (y[7] - 0.5 * y[9] + 0.5 * y[10] + y[13] - y[0] / r) / c;
    f[1] = (y[8] - 0.5 * y[11] + 0.5 * y[12] + y[14] - y[1] / r) / c;
    f[2] = (y[9] - qud1 + qud4) / cs;
    f[3] = (-y[10] + qud2 - qud3) / cs;
    f[4] = (y[11] + qud1 - qud3) / cs;
    f[5] = (-y[12] - qud2 + qud4) / cs;
    f[6] = (-y[6] / rp + qud1 + qud2 - qud3 - qud4) / cp;
    f[7] = -y[0] / lh;
    f[8] = -y[1] / lh;
    f[9] = (0.5 * y[0] - y[2] - rg2 * y[9]) / ls2;
    f[10] = (-0.5 * y[0] + y[3] - rg3 * y[10]) / ls3;
    f[11] = (0.5 * y[1] - y[4] - rg2 * y[11]) / ls2;
    f[12] = (-0.5 * y[1] + y[5] - rg3 * y[12]) / ls3;
    f[13] = (-y[0] + uin1 - (ri + rg1) * y[13]) / ls1;
    f[14] = (-y[1] - (rc + rg1) * y[14]) / ls1;
  }

template<typename Tp>
  void
  ring_modulator<Tp>::jacobian(Tp x)
  {
    const auto uin2 = 2 * std::sin(2e4 * pi * x);
    const auto ud1 = +y[2] - y[4] - y[6] - uin2;
    const auto ud2 = -y[3] + y[5] - y[6] - uin2;
    const auto ud3 = +y[3] + y[4] + y[6] + uin2;
    const auto ud4 = -y[2] - y[5] + y[6] + uin2;
    const auto qpud1 = gamma * delta * std::exp(delta * ud1);
    const auto qpud2 = gamma * delta * std::exp(delta * ud2);
    const auto qpud3 = gamma * delta * std::exp(delta * ud3);
    const auto qpud4 = gamma * delta * std::exp(delta * ud4);

    for (size_t i = 0; i < dimension; ++i)
      for (size_t j = 0; j < dimension; ++j)
	dfdy[i][j] = Tp{0};

    dfdy[0][0] = -1 / (c * r);
    dfdy[0][7] = 1 / c;
    dfdy[0][9] = -0.5 / c;
    dfdy[0][10] = -dfdy[0][9];
    dfdy[0][13] = dfdy[0][7];
    dfdy[1][1] = dfdy[0][0];
    dfdy[1][8] = dfdy[0][7];
    dfdy[1][11] = dfdy[0][9];
    dfdy[1][12] = dfdy[0][10];
    dfdy[1][14] = dfdy[0][13];
    dfdy[2][2] = (-qpud1 - qpud4) / cs;
    dfdy[2][4] = qpud1 / cs;
    dfdy[2][5] = -qpud4 / cs;
    dfdy[2][6] = (qpud1 + qpud4) / cs;
    dfdy[2][9] = 1 / cs;
    dfdy[3][3] = (-qpud2 - qpud3) / cs;
    dfdy[3][4] = -qpud3 / cs;
    dfdy[3][5] = qpud2 / cs;
    dfdy[3][6] = (-qpud2 - qpud3) / cs;
    dfdy[3][10] = -1 / cs;
    dfdy[4][2] = qpud1 / cs;
    dfdy[4][3] = -qpud3 / cs;
    dfdy[4][4] = (-qpud1 - qpud3) / cs;
    dfdy[4][6] = (-qpud1 - qpud3) / cs;
    dfdy[4][11] = 1 / cs;
    dfdy[5][2] = -qpud4 / cs;
    dfdy[5][3] = qpud2 / cs;
    dfdy[5][5] = (-qpud2 - qpud4) / cs;
    dfdy[5][6] = (qpud2 + qpud4) / cs;
    dfdy[5][12] = -1 / cs;
    dfdy[6][2] = (qpud1 + qpud4) / cp;
    dfdy[6][3] = (-qpud2 - qpud3) / cp;
    dfdy[6][4] = (-qpud1 - qpud3) / cp;
    dfdy[6][5] = (qpud2 + qpud4) / cp;
    dfdy[6][6] = (-qpud1 - qpud2 - qpud3 - qpud4 - 1 / rp) / cp;
    dfdy[7][0] = -1 / lh;
    dfdy[8][1] = dfdy[7][0];
    dfdy[9][0] = 0.5 / ls2;
    dfdy[9][2] = -1 / ls2;
    dfdy[9][9] = -rg2 / ls2;
    dfdy[10][0] = -0.5 / ls3;
    dfdy[10][3] = 1 / ls3;
    dfdy[10][10] = -rg3 / ls3;
    dfdy[11][1] = dfdy[9][0];
    dfdy[11][4] = dfdy[9][2];
    dfdy[11][11] = dfdy[9][9];
    dfdy[12][1] = dfdy[10][0];
    dfdy[12][5] = dfdy[10][3];
    dfdy[12][12] = dfdy[10][10];
    dfdy[13][0] = -1 / ls1;
    dfdy[13][13] = -(ri + rg1) / ls1;
    dfdy[14][1] = dfdy[13][0];
    dfdy[14][14] = -(rc + rg1) / ls1;

    for (size_t i = 0; i < dimension; ++i)
      dfdx[i] = Tp{0};
  }

#ifdef STANDALONE
int
main()
{
}
#endif
