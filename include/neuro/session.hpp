// thread-safe command queue and circuit state
#pragma once

#include "neuro/engine.hpp"
#include <mutex>
#include <variant>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <string>

namespace neuro {

    struct NodeDescriptor {
        uint32_t id{0};
        NeuronType type{NeuronType::Excitatory};
        IzhikevichParams izh;
        LocalChemParams chem;
        Vec3f position;
    };

    struct EdgeDescriptor {
        uint32_t id{0};
        uint32_t source_id{0};
        uint32_t target_id{0};
        float weight{1.0f};
        float delay_ms{1.0f};
    };

    // interactive command variants
    struct CmdPlaceNeuron {
        NodeDescriptor node;
    };

    struct CmdRemoveNeuron {
        uint32_t id;
    };

    struct CmdUpdateNeuron {
        uint32_t id;
        IzhikevichParams izh;
        LocalChemParams chem;
        NeuronType type;
    };

    struct CmdConnectEdge {
        EdgeDescriptor edge;
    };

    struct CmdDisconnectEdge {
        uint32_t id;
    };

    struct CmdUpdateEdge {
        uint32_t id;
        float weight;
        float delay_ms;
    };

    struct CmdSetStimulus {
        uint32_t target_neuron_id;
        StimulusWaveform waveform;
        float amplitude_pa;
        float frequency_hz;
        float duty_cycle;
        float phase_deg;
        bool active;
    };

    struct CmdSetOutputRule {
        uint32_t target_neuron_id;
        OutputDecodeMode mode;
        float window_ms;
        float threshold;
    };

    struct CmdSimControl {
        enum class Action { Play, Pause, Step, Reset, ClearAll } action;
        uint32_t step_count{1};
        float speed_multiplayer{1.0f};
    };

    using SessionCommand = std::variant<
        CmdPlaceNeuron,
        CmdRemoveNeuron,
        CmdUpdateNeuron,
        CmdConnectEdge,
        CmdDisconnectEdge,
        CmdUpdateEdge,
        CmdSetStimulus,
        CmdSetOutputRule,
        CmdSimControl
    >;

    class Session {
        public:
            explicit Session(float dt_ms = 0.05f, float dt_chem_ms = 1.0f);

            void push_command(SessionCommand cmd);

            // main frame execution: flush commands, step simulation if playing, return telemetry
            EngineTelemetry step_frame(size_t default_steps_per_frame);
            
            bool is_playing() const noexcept { return is_playing_.load(); }
            float speed_multiplier() const noexcept { return speed_multiplier_.load(); }

            // direct read of current circuit topology
            std::vector<NodeDescriptor> get_nodes() const;
            std::vector<EdgeDescriptor> get_edges() const;
        
        private:
            void flush_commands();
            void rebuild_circuit();

            SimulationEngine engine_;

            mutable std::mutex mutex_;
            std::vector<SessionCommand> command_queue_;

            std::atomic<bool> is_playing_{false};
            std::atomic<float> speed_multiplier_{1.0f};

            // user id to dense index mapping
            std::vector<NodeDescriptor> nodes_;
            std::vector<EdgeDescriptor> edges_;
            std::unordered_map<uint32_t, size_t> user_to_dense_node_;
            std::unordered_map<size_t, uint32_t> dense_to_user_node_;
    };
} // namespace neuro