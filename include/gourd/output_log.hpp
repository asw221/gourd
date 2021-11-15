
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>


#ifndef _GOURD_OUTPUT_LOG_
#define _GOURD_OUTPUT_LOG_

namespace gourd {

  class output_log {
  public:
    explicit output_log( const std::string& opath = "" );
    ~output_log();
    
    void add_log( const std::string& id );
    std::string make_filename( const std::string& suffix );

    std::ofstream& operator[]( const std::string& id );
    /* Throws error if desired element has not yet been created
     * with \c add_log()
     */

    template< typename IterT, typename StrideT = int >
    void write(
      const std::string& id,
      const IterT first,
      const IterT last,
      const StrideT stride = 1
    );

    static std::string fext;
    
  private:
    static char delim_;
    static char eol_;
    std::string basename_;
    std::map<std::string, std::ofstream> logs_;
  };
  // class output_log

}  // namespace gourd


std::string gourd::output_log::fext = ".dat";
char gourd::output_log::delim_ = '\t';
char gourd::output_log::eol_ = '\n';


gourd::output_log::output_log( const std::string& opath ) {
  basename_ = opath;
};

std::ofstream& gourd::output_log::operator[](
  const std::string& id
) {
  return logs_.at( id );
};


template< typename IterT, typename StrideT >
void gourd::output_log::write(
  const std::string& id,
  const IterT first,
  const IterT last,
  const StrideT stride
) {
  std::ofstream& os = logs_.at( id );
  for ( IterT it = first; it != last; it += stride ) {
    os << (*it);
    if ( it + stride < last )  os << delim_;
  }
  os << eol_;
};


gourd::output_log::~output_log() {
  for ( auto& elem : logs_ )  elem.second.close();
};


void gourd::output_log::add_log( const std::string& id ) {
  if ( !logs_[ id ].is_open() ) {
    logs_[id].open( make_filename(id), std::ofstream::out );
    if ( !logs_[id] ) {
      std::ostringstream msg;
      msg << "output_log: could not create output buffer for "
	  << id;
      throw std::runtime_error( msg.str() );
    }
  };
};


std::string gourd::output_log::make_filename(
  const std::string& suffix
) {
  return basename_ + suffix + fext;
};


#endif  // _GOURD_OUTPUT_LOG_
