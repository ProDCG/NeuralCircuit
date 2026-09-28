// lightweight websocket server
#pragma once

#include "neuro/session.hpp"
#include <string>
#include <memory>
#include <atomic>
#include <thread>

namespace neuro {
    class SimWebSocketServer {
        public:
            SimWebSocketServer(Session& session, int port = 8080);
            ~SimWebSocketServer();

            bool start();
            void stop();

            bool is_running() const noexcept { return running_.load(); }
            int port() const noexcept { return port_; }

        private:
            void run_broadcast_loop();
            void handle_client_message(const std::string& msg);

            Session& session_;
            int port_{8080};
            std::atomic<bool> running_{false};
            std::thread broadcast_thread_;

            struct Impl;
            std::unique_ptr<Impl> pimpl_;
    };
} // namespace neuro