
#include <array>
#include <iostream>

// #include <type_traits>


#ifndef _ABSEIL_COVARIANCE_FUNCTORS_
#define _ABSEIL_COVARIANCE_FUNCTORS_


namespace abseil {


  /* TODO: write better/more complete set of constructors */
  

  /* ****************************************************************/
  /*! Covariance functor
   *
   * Parent covariance functor class. Equivalent to uniform/constant
   * covariance
   */
  template< typename T = double, size_t M = 2 >
  class covariance_functor {
  public:
    typedef T result_type;

    class param_type {
    private:
      std::array<T, M> _theta;
      
    public:
      typedef size_t size_type;
      
      param_type() { ; }
      param_type( const param_type& other );
      // explicit param_type( const int n );
      
      template< typename InputIt >
      param_type( InputIt first, InputIt last );

      param_type & operator=( const param_type & other );
      
      T& operator[]( const int pos );
      const T& operator[]( const int pos ) const;
      constexpr size_t size() const { return M; };

      friend std::ostream& operator<<(
        std::ostream& os,
	const param_type& param
      ) {
	os << "\u03B8" << " = (";
	for ( unsigned i = 0; i < param.size(); i++ ) {
	  os << param[i] << ", ";
	}
	os << "\b\b)'";
	return os;
      };
      // friend bool operator== (const param_type& lhs, const param_type& rhs);
    };
    // class param_type

    covariance_functor() { ; }
    covariance_functor( const covariance_functor<T, M> & other );

    explicit covariance_functor( const param_type & par ) :
      _par( par )
    { ; }
    
    template< typename InputIt >
    covariance_functor( InputIt first, InputIt last ) :
      _par( first, last )
    { ; }
    // Can put SFINAE here ^^
    //   typename std::enable_if_t<std::is_floating_point<T>::value, bool> = true

    virtual ~covariance_functor() { ; }
    
    virtual T operator() ( const T val ) const;
    virtual T inverse( const T cov ) const;
    virtual T fwhm() const;
    // friend bool operator== (const covariance_functor<T>& lhs, const covariance_functor<T>& rhs);

    virtual std::array<T, M> gradient( const T val ) const;
    virtual std::array<T, M> param_lower_bounds() const;
    virtual std::array<T, M> param_upper_bounds() const;

    virtual void param( const param_type & par );
    
    param_type param() const;
    constexpr size_t param_size() const noexcept { return M; };

  protected:
    param_type _par;
  };
  // class covariance_functor
  /* ****************************************************************/


  

  /* ****************************************************************/
  /*! Radial basis covariance functor
   *
   * Three parameter model:
   *   k(d) = sigma^2 * exp( -psi * d^nu )
   */
  template< typename T = double >
  class radial_basis :
    public covariance_functor<T, 3> {
  public:
    typedef T result_type;
    using param_type = typename covariance_functor<T, 3>::param_type;
    using covariance_functor<T, 3>::param_size;
    using covariance_functor<T, 3>::param;

    radial_basis() : covariance_functor<T, 3>() { ; }

    explicit radial_basis( const param_type& par );
    
    template< typename InputIt >
    radial_basis( InputIt first, InputIt last );

    T operator() ( const T val ) const;
    T inverse( const T val ) const;
    T fwhm() const;

    std::array<T, 3> gradient( const T val ) const;
    std::array<T, 3> param_lower_bounds() const;
    std::array<T, 3> param_upper_bounds() const;

    T variance() const;
    T bandwidth() const;
    T exponent() const;

    void param( const param_type& par );
    void variance( const T val );
    void bandwidth( const T val );
    void exponent( const T val );
    

  private:
    void _validate_parameters() const;
  };
  /* ****************************************************************/



  
  /* ****************************************************************/
  /*! Matern covariance
   * 
   * Parameters are (sigma^2, rho, nu), all of which should be > 0
   *
   * C_nu(d) = sigma^2 * 2^(1 - nu) / Gamma(nu) *
   *             (sqrt(2 * nu) * d / rho)^nu * 
   *             K_nu( sqrt(2 * nu) * d / rho ),
   *
   * where K_nu(*) is the modified Bessel function of the second kind,
   * with order nu.
   *
   */
  template< typename T = double >
  class matern :
    public covariance_functor<T, 3> {
  public:
    typedef T result_type;
    using param_type = typename covariance_functor<T, 3>::param_type;
    using covariance_functor<T, 3>::param_size;
    using covariance_functor<T, 3>::param;

    matern();

    explicit matern( const param_type& par );
    
    template< typename InputIt >
    matern( InputIt first, InputIt last );

    T operator() ( const T val ) const;
    T inverse( const T val ) const;
    T fwhm() const;

    std::array<T, 3> gradient( const T val ) const;
    std::array<T, 3> param_lower_bounds() const;
    std::array<T, 3> param_upper_bounds() const;

    T variance() const;
    T nu() const;
    T rho() const;
    
    void param( const param_type& par );
    void variance( const T val );
    void nu( const T val );
    void rho( const T val );
    

  private:
    static T _eps;
    static T _tol;
    static int _max_it;
    T _norm_c;
    T _sqrt_2nu_rho;
    void _compute_normalizing_constant();
  };

  
  
}
// namespace abseil



#include <abseil/covariance/covariance_functor_def.inl>
#include <abseil/covariance/radial_basis_def.inl>
#include <abseil/covariance/matern_def.inl>




#endif  // _ABSEIL_COVARIANCE_FUNCTORS_
