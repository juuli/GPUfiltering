#ifndef LOGGER_H
#define LOGGER_H

#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h> 

#define LOG_COUT 4
#define LOG_TO_FILE 4

enum log_level {
    LOG_NOTHING,
    LOG_CRITICAL,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_DEBUG,
    LOG_VERBOSE
};

class Logger {
public:
  Logger(log_level level, const wchar_t* msg )
  : level_(level),
    fmt_(msg),
    logfile_("solver_log.txt",  std::fstream::out | std::fstream::app)
  {};

  ~Logger() {
    if(LOG_COUT >= level_)
      std::wcout<<level_<<L" "<<fmt_<<std::endl;

      if(LOG_TO_FILE >= level_) {
        logfile_<<level_<<L" "<<fmt_<<std::endl;
        logfile_.close();
      }
  }

  template<typename T>
  Logger& operator %(T value) {
    fmt_ % value;
    return *this;
  }

  Logger(const Logger& other)
  : level_(other.level_),
    fmt_(other.fmt_),
    logfile_("solver_log.txt",  std::fstream::out | std::fstream::app)
  {}

private:
  log_level level_;
  boost::wformat fmt_;
  std::wofstream logfile_;
};

template <log_level level>
Logger log_msg(const wchar_t* msg) {
  return Logger(level, msg);
}

void loggerInit(); 


// The C interface
//typedef struct c_Logger Logger;


#endif
