#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>

class Logger {
public:
    Logger(const std::string& filename) {
        logFile.open(filename, std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void log(const std::string& message) {
        if (logFile.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto now_c = std::chrono::system_clock::to_time_t(now);
            logFile << std::put_time(std::localtime(&now_c), "%Y-%m-%d %X") << " - " << message << std::endl;
        }
    }

private:
    std::ofstream logFile;
};

#endif // LOGGING_H