




  template< typename T > inline constexpr
  std::array<int, 2> beschb_nuse_v = std::array<int, 2>{6, 7};
  
  template<> inline constexpr
  std::array<int, 2> beschb_nuse_v<float> = std::array<int, 2>{5, 5};


  /*! Evaluate Gamma_1 and Gamma_2 by Chebyshev expansion
   *
   * \Gamma_1(nu) = [1/Gamma(1-nu) - 1/Gamma(1+nu)] / (2*nu)
   * \Gamma_2(nu) = [1/Gamma(1-nu) + 1/Gamma(1+nu)] / 2
   *
   * Input x must satisfy |x| <= 0.5
   * Modified from "Numerical Recipes in C" (p 245).
   */
  template< typename T = double >
  void beschb(
    const T x,
    T& gam1, T& gam2,   /*!< Return values */
    T& gampl, T& gammi  /*!< 1/Gamma(1+x) and 1/Gamma(1-x) */
  ) {
    assert( std::abs(x) <= (T)0.5 && "beschb: input |x| > 0.5" );
    static const std::array<T, 7> coefs_1{
      -1.142022680371168e0, 6.5165112670737e-3,
	3.087090173086e-4, -3.4706269649e-6, 6.9437664e-9,
	3.67795e-11, -1.356e-13
    };
    static const std::array<T, 8> coefs_2{
      1.843740587300905e0, -7.68528408447867e-2,
	1.2719271366546e-3, -4.9717367042e-6, -3.31261198e-8,
	2.423096e-10, -1.702e-13, -1.49e-15
    };
    static const std::array<int, 2> n = beschb_nuse_v<T>;
    const T xx = 8 * x * x - 1;
    gam1 = abseil::chebev<T>(-1, 1, coefs_1.cbegin(),
			     coefs_1.cbegin() + n[0], xx);
    gam2 = abseil::chebev<T>(-1, 1, coefs_2.cbegin(),
			     coefs_2.cbegin() + n[1], xx);
    gampl = gam2 - x * gam1;
    gammi = gam2 + x * gam1;
  };






  template< typename T >
  void bessik(
    const T x, const T nu,
    T& ri, T& rk, T& rip, T& rkp,
    const int maxit = 1e4,
    const T eps = std::numeric_limits<T>::epsilon()/2,
    const T xpiv = 2
  ) {
    assert( x > 0 && "bessik: x must be > 0" );
    assert( nu >= 0 && "bessik: nu must be >= 0" );
    const T fpmin = std::numeric_limits<T>::min();
    const int nl = static_cast<int>(nu + 0.5);
    /* ^^ Number of downward recurrences of I's/ upward recurrences
     * of the K's */
    T xmu = nu - nl;
    T xmu2 = xmu * xmu;
    T xi = 1 / x;
    T xi2 = 2 * xi;
    T h = nu * xi;  h = (h < fpmin) ? fpmin : h;
    T b = xi2 * nu;
    T d = 0;
    T c = h;
    T del;
    int i;
    for ( i = 0; i < maxit; i++ ) {
      b += xi2;
      d = 1 / (b + d);
      c = b + 1/c;
      del = c * d;
      h = del * h;
      if ( std::abs(del - 1) < eps ) break;
    }
    if ( i == (maxit - 1) )
      std::cerr << "\t*** WARNING: bessik: maxit reached\n";
    T ril = fpmin;
    T ripl = h * ril;
    T ril1 = ril;
    T rip1 = ripl;
    T fact = nu * xi;
    T ritemp;
    int l;
    for ( l = nl; l >= 1; l-- ) {
      ritemp = fact * ril + ripl;
      fact -= xi;
      ripl = fact * ritemp + ril;
      ril = ritemp;
    }
    T f = ripl / ril;  /* Now have unnormalized I */
    T x2, pimu, fact2, e, ff, sum, p, q, sum1;
    T gam1, gam2, gampl, gammi;
    T del1, delh, rkmu, rk1;
    if ( x < xpiv ) {  /* Use series */
      x2 = 0.5 * x;
      pimu = num::pi_v<T> * xmu;
      fact = (std::abs(pimu) < eps) ? 1 : pimu / std::sin(pimu);
      d = -std::log(x2);
      e = xmu * d;
      fact2 = (std::abs(e) < eps) ? 1 : std::sin(e) / e;
      gampl = 1 / std::tgamma(1 + xmu);
      gammi = 1 / std::tgamma(1 - xmu);
      gam1 = (gammi - gampl) / (2 * xmu + eps);
      gam2 = (gammi + gampl) / 2;
      ff = fact * (gam1 * std::cosh(e) + gam2 * fact2 * d);  /* f0 */
      sum = ff;
      e = std::exp(e);
      p = 0.5 * e / gampl;
      q = 0.5 / (e * gammi);
      c = 1;
      d = x2 * x2;
      sum1 = p;
      for ( i = 1; i <= maxit; i++ ) {
	ff = (i * ff + p + q) / (i * i - xmu2);
	c *= (d / i);
	p /= (i - xmu);
	q /= (i + xmu);
	del = c * ff;
	sum += del;
	del1 = c * (p - i * ff);
	sum1 += del1;
	if ( std::abs(del) < (std::abs(sum) * eps)) break;
      }
      if (i == (maxit + 1))
	std::cerr << "\t*** WARNING: bessik: series didn't converge\n";
      rkmu = sum;
      rk1 = sum1 * xi2;
    }
    else {
      b = 2 * (1 + x);
      d = 1 / b;
      h = delh = d;
      T q1 = 0;
      T q2 = 1;
      T a1 = 0.25 - xmu2;
      q = c = a1;
      T a = -a1;
      T s = 1 + q * delh;
      for ( i = 2; i <= maxit; i++ ) {
	a -= 2 * (i - 1);
	c = -a * c / i;
	T qnew = (q1 - b * q2) / a;
	q1 = q2;
	q2 = qnew;
	q += c * qnew;
	b += 2;
	d = 1 / (b + a * d);
	delh = (b * d - 1) * delh;
	h += delh;
	T dels = q * delh;
	s += dels;
	if ( std::abs(dels/s) < eps ) break;
      }
      if ( i > maxit )
	std::cerr << "\t*** WARNING: bessik: CF2 did not converge\n";
      h = a1 * h;
      rkmu = std::sqrt(num::pi_v<T> / (2 * x)) * std::exp(-x) / s;
      rk1 = rkmu * (xmu + x + 0.5 - h) * xi;
    }
    //
    const T rkmup = xmu * xi * rkmu - rk1;
    const T rimu = xi / (f * rkmu - rkmup);
    ri = (rimu * ril1) / ril;
    rip = (rimu * rip1) / ril;
    for ( i = 1; i <= nl; i++ ) {
      T rktemp = (xmu + i) * xi2 * rk1 + rkmu;
      rkmu = rk1;
      rk1 = rktemp;
    }
    rk = rkmu;
    rkp = nu * xi * rkmu - rk1;
  };
