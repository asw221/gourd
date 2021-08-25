
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>



#ifndef _GOURD_BASIC_COMMAND_PARSER_
#define _GOURD_BASIC_COMMAND_PARSER_


namespace gourd {

  /* ****************************************************************/
  /*! Parse basic command line input
   */
  class basic_command_parser {
  public:
    enum class call_status { success, error, help };

    basic_command_parser( const int argc, const char** argv );
    
    virtual void show_help() const { ; }
    virtual void show_usage() const { ; }

    bool error() const;
    bool help_invoked() const;
    operator bool() const;
    bool operator!() const;

    call_status status() const;
    std::string caller() const;

    bool is_file(
      const std::string fname,
      const bool warn = true
    ) const;
    
    void process_file_argument(
      const int argc,
      const char** argv,
      int& i,
      std::string& fpath
    );
    
    template< typename T >
    void process_numeric_argument(
      const int argc,
      const char** argv,
      int& i,
      T& param
    );

    
    void process_string_argument(
      const int argc,
      const char** argv,
      int& i,
      std::string& param
    );

    template< typename T >
    void process_vector_argument(
      const int argc,
      const char** argv,
      int& i,
      std::vector<T>& param
    );
    
  protected:    
    call_status status_;
    std::string caller_;
  };
  // class basic_command_parser
  /* ****************************************************************/

};
// namespace gourd







gourd::basic_command_parser::basic_command_parser(
  const int argc,
  const char *argv[]
) {
  status_ = call_status::success;
  caller_ = std::filesystem::path( argv[0] ).filename();  // generic_string()
};
// basic_command_parser( const int argc, const char *argv[] )




gourd::basic_command_parser::call_status
gourd::basic_command_parser::status() const {
  return status_;
};

std::string gourd::basic_command_parser::caller() const {
  return caller_;
};




bool gourd::basic_command_parser::is_file(
  const std::string fname,
  const bool warn
) const {
  std::ifstream ifs( fname, std::ifstream::in );
  const bool good = ifs.is_open();
  ifs.close();
  if ( !good && warn ) {
    std::cerr << "\nCould not read " << fname << "\n";
  }
  return good;
};



void gourd::basic_command_parser::process_file_argument(
  const int argc,
  const char* argv[],
  int& i,
  std::string& fpath
) {
  const std::string argi = argv[i];
  if ( i + 1 < argc ) {
    i++;
    fpath = argv[i];
    if ( !is_file(fpath) ) {
      status_ = call_status::error;
    }
  }
  else {
    std::cerr << argi << " option requires one argument\n";
    status_ = call_status::error;
  }
};




template< typename T >
void gourd::basic_command_parser::process_numeric_argument(
  const int argc,
  const char** argv,
  int& i,
  T& param
) {
  const std::string argi = argv[i];
  if ( i + 1 < argc ) {
    i++;
    try {
      param = T( std::stod(argv[i]) );
    }
    catch (...) {
      std::cerr << argi << " option requires one numeric argument\n";
      status_ = call_status::error;
    }
  }
  else {
    std::cerr << argi << " option requires one numeric argument\n";
    status_ = call_status::error;
  }
};




void gourd::basic_command_parser::process_string_argument(
  const int argc,
  const char** argv,
  int& i,
  std::string& param
) {
  const std::string argi = argv[i];
  if ( i + 1 < argc ) {
    i++;
    param = argv[i];
  }
  else {
    std::cerr << argi << " option requires string argument\n";
    status_ = call_status::error;
  }
};




template<>
void gourd::basic_command_parser::process_vector_argument<double>(
  const int argc,
  const char** argv,
  int& i,
  std::vector<double>& param
) {
  const std::string argi = argv[i];
  bool seek = true;
  while ( (i + 1) < argc && seek ) {
    i++;
    try {
      param.push_back( std::stod(argv[i]) );
    }
    catch (...) {
      seek = false;
      i--;
    }
  }
};


template<>
void gourd::basic_command_parser::process_vector_argument<float>(
  const int argc,
  const char** argv,
  int& i,
  std::vector<float>& param
) {
  const std::string argi = argv[i];
  bool seek = true;
  while ( (i + 1) < argc && seek ) {
    i++;
    try {
      param.push_back( std::stof(argv[i]) );
    }
    catch (...) {
      seek = false;
      i--;
    }
  }
};




template< typename T >
void gourd::basic_command_parser::process_vector_argument(
  const int argc,
  const char** argv,
  int& i,
  std::vector<T>& param
) {
  std::cerr << "Vector argument type not available\n";
  status_ = call_status::error;
};






bool gourd::basic_command_parser::error() const {
  return status_ == call_status::error;
};


bool gourd::basic_command_parser::help_invoked() const {
  return status_ == call_status::help;
};

gourd::basic_command_parser::operator bool() const {
  return !error();
};

bool gourd::basic_command_parser::operator!() const {
  return !(this->operator bool());
};



#endif  // _GOURD_BASIC_COMMAND_PARSER_
