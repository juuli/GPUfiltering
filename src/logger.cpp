#include "logger.h"

void loggerInit() {
	std::wofstream logfile("solver_log.txt",  std::fstream::out | std::fstream::trunc);
}

