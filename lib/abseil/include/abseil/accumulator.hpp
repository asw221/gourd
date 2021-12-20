

#ifndef _ABSEIL_ACCUMULATOR_
#define _ABSEIL_ACCUMULATOR_

namespace abseil {

  /*! Basic accumulator class */
  template< typename T >
  class accumulator {
  public:
    typedef T value_type;
    
    explicit accumulator() : val_(0) { ; }
    explicit accumulator( const T x ) : val_(x) { ; }

    value_type value() const { return val_; };
    operator T() const { return val_; };
    
    bool operator==( const T x ) const { return val_ == x; };
    bool operator!=( const T x ) const { return val_ != x; };
    bool operator< ( const T x ) const { return val_ <  x; };
    bool operator> ( const T x ) const { return val_ >  x; };
    bool operator<=( const T x ) const { return val_ <= x; };
    bool operator>=( const T x ) const { return val_ >= x; };
    
    accumulator<T>& operator++() { ++val_; return *this; };
    accumulator<T>& operator--() { --val_; return *this; };
    
    //
    accumulator<T>& operator=( const T x ) {
      val_ = x; return *this;
    };
    accumulator<T>& operator+=( const T x ) {
      val_ += x; return *this;
    };
    accumulator<T>& operator*=( const T x ) {
      val_ *= x; return *this;
    };
    accumulator<T>& operator%=( const T x ) {
      val_ %= x; return *this;
    };
    accumulator<T>& operator-=( const T x ) {
      return operator+=(-x);
    };
    accumulator<T>& operator/=( const T x ) {
      return operator*=(1/x);
    };

    accumulator<T> operator+( const T x ) const {
      abseil::accumulator<T> y(x); y += x; return y;
    };
    accumulator<T> operator-( const T x ) const {
      abseil::accumulator<T> y(x); y -= x; return y;
    };
    accumulator<T> operator*( const T x ) const {
      abseil::accumulator<T> y(x); y *= x; return y;
    };
    accumulator<T> operator/( const T x ) const {
      abseil::accumulator<T> y(x); y /= x; return y;
    };
    accumulator<T> operator%( const T x ) const {
      abseil::accumulator<T> y(x); y %= x; return y;
    };
    //
    
  protected:
    T val_;
  };
  /* ****************************************************************/

  

  /*! Accumulator class using the Kahan summation algorithm
   */
  template< typename T >
  class kahan_accumulator :
    public abseil::accumulator<T> {
  public:
    typedef T value_type;

    explicit kahan_accumulator() : c_(0) { ; }
    explicit kahan_accumulator( const T x ) :
      abseil::accumulator<T>(x), c_(0) { ; }

    kahan_accumulator<T>& operator=( const T x ) {
      this->val_ = x; c_ = 0; return *this;
    };
    kahan_accumulator<T>& operator+=( const T x ) {
      T y = x - c_;
      volatile T t = this->val_ + y;
      volatile T z = t - this->val_;
      c_ = z - y;
      this->val_ = t;
      return *this;
    };
    kahan_accumulator<T>& operator-=( const T x ) {
      return operator+=(-x);
    };
    kahan_accumulator<T>& operator*=( const T x ) {
      this->val_ *= x; c_ *= x; return *this;
    };
    kahan_accumulator<T>& operator/=( const T x ) {
      return operator*=(1/x);
    };
    kahan_accumulator<T>& operator%=( const T x ) {
      this->val_ %= x; c_ /= x; return *this;
    };
    
    kahan_accumulator<T> operator+( const T x ) const {
      abseil::kahan_accumulator<T> y(this->val_); y += x; return y;
    };
    kahan_accumulator<T> operator-( const T x ) const {
      abseil::kahan_accumulator<T> y(this->val_); y -= x; return y;
    };
    kahan_accumulator<T> operator*( const T x ) const {
      abseil::kahan_accumulator<T> y(this->val_); y *= x; return y;
    };
    kahan_accumulator<T> operator/( const T x ) const {
      abseil::kahan_accumulator<T> y(this->val_); y /= x; return y;
    };
    kahan_accumulator<T> operator%( const T x ) const {
      abseil::kahan_accumulator<T> y(this->val_); y %= x; return y;
    };    
    
  protected:
    T c_;
  };
  /* ****************************************************************/
  
}  // namespace abseil

#endif  // _ABSEIL_ACCUMULATOR_
