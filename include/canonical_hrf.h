
#include <cmath>


#ifndef _GOURD_CANONICAL_HRF_
#define _GOURD_CANONICAL_HRF_


namespace gourd {

  template< typename T = double >
  class canonical_hrf {
  public:
    typedef T result_type;

    canonical_hrf() : canonical_hrf(6) { ; }
    explicit canonical_hrf(
      const T a1 = 6,   /*!< HRF peak time */
      const T a2 = 16,  /*!< HRF undershoot time */
      const T b1 = 1,   /*!< Peak scale parameter */
      const T b2 = 1,   /*!< Undershoot scale parameter */
      const T c  = 6    /*!< Ratio between peak/undershoot */
    );

    T operator()( const T t ) const;
    T gradient( const T t ) const;
    T gradient_scale( const T t ) const;

    T peak_scale() const;
    T peak_time()  const;
    T undershoot_scale() const;
    T undershoot_time()  const;
    T ratio() const;
    
    void peak_scale( const T b1 );
    void peak_time( const T a1 );
    void undershoot_scale( const T b2 );
    void undershoot_time( const T a2 );
    void ratio( const T c );
    
  private:
    T _a1;  /* HRF peak time */
    T _a2;  /* HRF undershoot time */
    T _b1;  /* Peak scale parameter */
    T _b2;  /* Undershoot scale parameter */
    T _c;   /* Ratio between peak/undershoot */
    
    T _Norm1;  /* Peak normalizing constant */
    T _Norm2;  /* Undershoot normalizing constant */
    void _set_norm1();
    void _set_norm2();
  };

};


template< typename T >
gourd::canonical_hrf<T>::canonical_hrf(
  const T a1,
  const T a2,
  const T b1,
  const T b2,
  const T c
) {
  _a1 = std::abs( a1 );
  _a2 = std::abs( a2 );
  _b1 = std::abs( b1 );
  _b2 = std::abs( b2 );
  _c  = std::abs( c );
  _set_norm1();
  _set_norm2();
};




template< typename T >
T gourd::canonical_hrf<T>::operator()( const T t ) const {
  return std::pow( t, _a1 - 1 ) * std::exp( -_b1 * t ) * _Norm1 -
    std::pow( t, _a2 - 1 ) * std::exp( -_b2 * t ) * _Norm2 / _c;
};


template< typename T >
T gourd::canonical_hrf<T>::gradient( const T t ) const {
  const T d_peak_dt = std::exp( -_b1 * t ) * _Norm1 *
    ( (_a1 - 1) * std::pow(t, _a1 - 2) - _b1 * std::pow(t, _a1 - 1) );
  const T d_undershoot_dt = std::exp( -_b2 * t ) * _Norm2 / _c *
    ( (_a2 - 1) * std::pow(t, _a2 - 2) - _b2 * std::pow(t, _a2 - 1) );
  return d_peak_dt - d_undershoot_dt;
};


template< typename T >
T gourd::canonical_hrf<T>::gradient_scale( const T t ) const {
  return std::pow( t, _a1 - 1 ) * std::exp( -_b1 * t ) *
    ( _a1 * std::pow( _b1, _a1 - 1 ) - t ) * _Norm1 /
    std::pow( _b1, _a1 );
};



// --- Getters -------------------------------------------------------

template< typename T >
T gourd::canonical_hrf<T>::peak_scale() const {
  return _b1;
};


template< typename T >
T gourd::canonical_hrf<T>::peak_time() const {
  return _a1;
};


template< typename T >
T gourd::canonical_hrf<T>::undershoot_scale() const {
  return _b2;
};


template< typename T >
T gourd::canonical_hrf<T>::undershoot_time() const {
  return _a2;
};


template< typename T >
T gourd::canonical_hrf<T>::ratio() const {
  return _c;
};


// --- Setters -------------------------------------------------------

template< typename T >
void gourd::canonical_hrf<T>::peak_scale( const T b1 ) const {
  _b1 = std::abs( b1 );
  _set_norm1();
};


template< typename T >
void gourd::canonical_hrf<T>::peak_time( const T a1 ) const {
  _a1 = std::abs( a1 );
  _set_norm1();
};


template< typename T >
void gourd::canonical_hrf<T>::undershoot_scale( const T b2 ) const {
  _b2 = std::abs( b2 );
  _set_norm2();
};


template< typename T >
void gourd::canonical_hrf<T>::undershoot_time( const T a2 ) const {
  _a2 = std::abs( a2 );
  _set_norm2();
};


template< typename T >
void gourd::canonical_hrf<T>::ratio( const T c ) const {
  _c = std::abs( c );
};



template< typename T >
void gourd::canonical_hrf<T>::_set_norm1() {
  _Norm1 = std::pow( _b1, _a1 ) / std::tgamma( _a1 );
};

template< typename T >
void gourd::canonical_hrf<T>::_set_norm2() {
  _Norm2 = std::pow( _b2, _a2 ) / std::tgamma( _a2 );
};





#endif  // _GOURD_CANONICAL_HRF_

