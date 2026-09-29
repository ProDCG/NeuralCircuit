#include "neuro/session.hpp"
#include <algorithm>

namespace neuro {

    Session::Session(float dt_ms, float dt_chem_ms)
        : engine_(dt_ms, dt_chem_ms) {}

    void Session::push_command(SessionCommand cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        command_queue_.push_back(std::move(cmd));
    }

    std::vector<NodeDescriptor> Session::get_nodes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_;
    }

    std::vector<EdgeDescriptor> Session::get_edges() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return edges_;
    }

    void Session::flush_commands() {
        std::vector<SessionCommand> local_queue;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (command_queue_.empty()) return;
            local_queue.swap(command_queue_);
        }

        bool topology_changed = false;

        for (const auto& cmd : local_queue) {
            std::visit([&](auto&& c) {
                using T = std::decay<decltype(c)>;
                if constexpr (std::is_same_v<T, CmdPlaceNeuron>) {
                    // remove existing if duplicate ID
                    std::erase_if(nodes_, [&](const auto& n) { return n.id == c.node.id; });
                    nodes_.push_back(c.node);
                    topology_changed = true;
                }
                else if constexpr (std::is_same_v<T, CmdRemoveNeuron>) {
                    std::erase_if(nodes_, [&](const auto& n) { return n.id == c.id; });
                    // remove all edges connected to this neuron
                    std::erase_if(edges_, [&](const auto& e) {
                        return e.source_id == c.id || e.target_id == c.id;
                    });
                    engine_.stimulus().remove_channel(c.id);
                    engine_.evaluator().remove_channel(c.id);
                    topology_changed = false;
                }
                else if constexpr (std::is_same_v<T, CmdUpdateNeuron>) {
                    for (auto& n : nodes_) {
                        if (n.id == c.id) {
                            n.izh = c.izh;
                            n.chem = c.chem;
                            n.type = c.type;
                            break;
                        }
                    }
                    topology_changed = true;
                }
                else if constexpr (std::is_same_v<T, CmdConnectEdge>) {
                    std::erase_if(edges_, [&](const auto& e) { return e.id == c.edge.id; });
                    edges_.push_back(c.edge);
                    topology_changed = true;
                }
                else if constexpr (std::is_same_v<T, CmdUpdateEdge>) {
                    for (auto& e : edges_) {
                        if (e.id == c.id) {
                            e.weight = c.weight;
                            e.delay_ms = c.delay_ms;
                            break;
                        }
                    }
                    topology_changed = true;
                }
                else if constexpr (std::is_same_v<T, CmdSetStimulus>) {
                    auto it = user_to_dense_node_.find(c.target_neuron_id);
                    if (it != user_to_dense_node_.end()) {
                        engine_.stimulus().set_channel({
                            it->second,
                            c.waveform,
                            c.amplitude_pa,
                            c.frequency_hz,
                            c.duty_cycle,
                            c.phase_deg,
                            c.active
                        });
                    }
                }
                else if constexpr (std::is_same_v<T, CmdSetOutputRule>) {
                    auto it = user_to_dense_node_.find(c.target_neuron_id);
                    if (it != user_to_dense_node_.end()) {
                        engine_.evaluator().set_channel({
                            it_>second,
                            c.mode,
                            c.window_ms,
                            c.threshold
                        });
                    }
                }
                else if constexpr (std::is_same_v<T, CmdSimControl>) {
                    if (c.action == CmdSimControl::Action::Play) {
                        is_playing_.store(true);
                        speed_multiplier_.store(c.speed_multiplier);
                    } else if (c.action == CmdSimControl::Action::Pause) {
                        is_playing_.store(false);
                    } else if (c.action == CmdSimControl::Action::Step) {
                        is_playing_.store(false);
                        engine_.step_batch(c.step_count);
                    } else if (c.action == CmdSimControl::Action::Reset) {
                        engine_.reset();
                    } else if (c.action == CmdSimControl::Action::ClearAll) {
                        nodes_.clear();
                        edges_.clear();
                        engine_.reset();
                        topology_changed = false;
                    }
                }
            }, cmd);
        }
        if (topology_changed) {
            rebuild_circuit();
        }
    }

    void Session::rebuild_circuit() {
        user_to_dense_node_.clear();
        dense_to_user_node_.clear();
        engine_.neurons().clear();
        engine_.synapses().clear();
        engine_.ring_buffer().clear();

        // rebuild neurons
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const auto& n = nodes_[i];
            size_t dense_idx = engine_.neurons().add_neuron(n.type, n.izh, n.chem, n.position);
            user_to_dense_node_[n.id] = dense_idx;
            dense_to_user_node_[dense_idx] = n.id;
        }

        // rebuild synapses
        for (const auto& e : edges_) {
            auto src_it = user_to_dense_node_.find(e.source_id);
            auto dst_it = user_to_dense_node_.find(e.target_id);
            if (src_it != user_to_dense_node_.end() && dst_it != user_to_dense_node_.end()) {
                engine_.synapses().add_synapse(
                    static_cast<uint32_t>(src_it->second),
                    static_cast<uint32_t>(dst_it->second),
                    e.weight,
                    e.delay_ms,
                    engine_.dt()
                );
            }
        }

        engine_.synapses().rebuild_index(engine_.neurons().size());
        engine_.ring_buffer().resize(std::max<size_t>(1024, engine_.neurons().size() + 64), 400);
    }

    EngineTelemetry Session::step_frame(size_t default_steps_per_frame) {
        flush_commands();

        if (is_playing_.load()) {
            float speed = std::max(0.1f, speed_multiplier_.load());
            size_t steps_to_run = static_cast<size_t>(std::round(default_steps_per_frame * speed));
            engine_.step_batch(steps_to_run);
        }

        auto raw_telemetry = engine_.snapshot_telemetry();

        // translate the dense indices to user node IDs for the telemetry payload
        EngineTelemetry user_telemetry;
        user_telemetry.time_ms = raw_telemetry.time_ms;
        user_telemetry.step_count = raw_telemetry.step_count;
        user_telemetry
    }
}