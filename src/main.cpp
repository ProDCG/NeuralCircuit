#include "neuro/session.hpp"
#include "neuro/ws_server.hpp"
#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

static std::atomic<bool> g_shutdown{false};

void signal_handler(int) {
    g_shutdown.store(true);
}

int main(int argc, char* argv[]) {
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        }
    }
}