
#include <iostream>


#ifndef _ABSEIL_PROGRESS_BAR_
#define _ABSEIL_PROGRESS_BAR_


namespace abseil {

namespace utilities {


    /*! Text progress bar
     * 
     * Std i/o
     */
    class progress_bar {
    public:
      progress_bar( unsigned int max_val );
      
      void finish();
      void operator++();
      void operator++( int );
      void value( unsigned int value );

      template< typename OStream >
      friend OStream& operator<<(
        OStream& os,
	const progress_bar& pb
      );
      
    private:
      bool _active;
      char __;
      unsigned int _max_val;
      unsigned int _print_width;
      unsigned int _bar_print_width;
      unsigned int _value;
    };


};
  // namespace utilities
  
};
// namespace utilities





abseil::utilities::progress_bar::progress_bar( unsigned int max_val ) {
  _active = true;
  __ = '=';
  
  _max_val = max_val;
  _print_width = 60;
  _bar_print_width = _print_width - 8;  // 8 additional characters: || xy.z%
  _value = 0;
};
      

void abseil::utilities::progress_bar::finish() {
  _active = false;
  std::cout << std::setprecision(4) << std::endl;
};

void abseil::utilities::progress_bar::operator++() {
  _value++;
  _value = (_value > _max_val) ? _max_val : _value;
};

void abseil::utilities::progress_bar::operator++(int) {
  ++(*this);
};

void abseil::utilities::progress_bar::value( unsigned int value ) {
  _value = value;
  _value = (_value > _max_val) ? _max_val : _value;
};
      



template< typename OStream >
OStream& abseil::utilities::operator<<(
    OStream& os,
    const abseil::utilities::progress_bar& pb
) {
  const double prop = (double)pb._value / pb._max_val;
  const unsigned int bars = (unsigned int)(prop * pb._bar_print_width);
  if (pb._active) {
    if (pb._value > 0) {
      for (unsigned int i = 0; i < pb._print_width; i++)  os << "\b";
    }
    os << "|";
    for (unsigned int i = 1; i <= pb._bar_print_width; i++) {
      if (i <= bars)
	os << pb.__;
      else
	os << " ";
    }
    if ( prop < 0.095 )
      os << "|  ";
    else if ( prop < 0.995 )
      os << "| ";
    else
      os << "|";
    os << std::setprecision(1) << std::fixed << (prop * 100) << "%"
       << std::flush;
  }
  return os;
};



#endif
