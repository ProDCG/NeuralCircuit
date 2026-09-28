// clean JSON import/export
#pragma once

#include "neuro/session.hpp"
#include <string>

namespace neuro {

    class CircuitSerializer {
        public:
            static std::string export_json(const Session& session, const std::string& circuit_name = "Untitled Circuit");
            static bool import_json(Session& session, const std::string& json_str, std::string& out_error);
    }
}