#include "neuro/engine.hpp"
#include <cmath>
#include <algorithm>

namespace neuro {

    SimulationEngine::SimulationEngine(float dt_ms, float dt_chem_ms)
    : dt_(dt_ms), dt_chem_(dt_chem_ms), neurons_(1024), ring_buffer_(1024, 400) {
        set_dt(dt_ms);
    }

    void SimulationEngine::set_dt(float dt) noexcept {
        dt_ = std::max(0.001f, dt);
        chem_ratio_ = static_cast<size_t>(std::max(1.0f, std::round(dt_chem_ / dt_)));
    }

    void SimulationEngine::set_dt_chem(float dt_chem) noexcept {
        dt_chem_ = std::max(0.01f, dt_chem);
        chem_ratio_ = static_cast<size_t>(std::max(1.0f, std::round(dt_chem_ / dt_)));
    }

    void SimulationEngine::reset() noexcept {
        current_time_ms_ = 0.0f;
        total_steps_ = 0;
        chem_step_counter_ = 0;
        ring_buffer_.clear();
        evaluator_.clear();
    }

    void SimulationEngine::step() {
        size_t num_neurons = neurons_.size();
        if (num_neurons == 0) {
            current_time_ms_ += dt_;
            total_steps_++;
            return;
        }

        // ingest external waveforms into I_ext
        stimulus_.apply(current_time_ms_, dt_, neurons_.I_ext());

        // read arriving delayed synaptic currents from ring buffer into I_syn
        ring_buffer_.read_and_clear(neurons_.I_syn());

        // fast electrical integration (euler numerical step)
        neurons_.integrate_electrical(dt_);

        // outgoing spike propagation
        auto spiked = neurons_.spiked();
        auto receptor_gain = neurons_.receptor_gain();
        auto metabolic_gate = neurons_.metabolic_gate();

        for (size_t i = 0; i < num_neurons; ++i) {
            if (spiked[i]) {
                // trigger local chemical release and glucose energy cost
                neurons_.on_neuron_spiked(i);

                // fetch outgoing synapses in O(1) time
                auto outgoing = synapses_.get_outgoing(i);
                if (outgoing.count > 0) {
                    // modulate base weight by source neuron's local chemical state
                    float gain = receptor_gain[i] * metabolic_gate[i];

                    for (uint32_t k = 0; k < outgoing.count; ++k) {
                        uint32_t dst = outgoing.targets[k];
                        float w_eff = outgoing.weights[k] * gain;
                        uint16_t delay = outgoing.delay_steps[k];

                        ring_buffer_.push(dst, delay, w_eff);
                    }
                }
            }
        }

        // dual-rate chemical micro-domain step
        if (++chem_step_counter_ >= chem_ratio_) {
            chem_step_counter_ = 0;
            neurons_.integrate_chemical(dt_chem_);
        }

        // record step in oiutput evaluator
        evaluator_.record_step(current_time_ms_, spiked);

        current_time_ms_ += dt_;
        total_steps_++;
    }

    void SimulationEngine::step_batch(size_t n_steps) {
        for (size_t s = 0; s < n_steps; ++s) {
            step();
        }
    }

    EngineTelemetry SimulationEngine::snapshot_telemetry() const {
        EngineTelemetry t;
        t.time_ms = current_time_ms_;
        t.step_count = total_steps_;

        size_t n = neurons_.size();
        auto spiked = neurons_.spiked();
        auto v = neurons_.v();
        auto c_trans = neurons_.C_trans();
        auto c_glc = neurons_.C_glc();

        t.spiked_indices.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (spiked[i]) {
                t.spiked_indices.push_back(static_cast<uint32_t>(i));
            }
        }
        t.active_spikes_count = static_cast<uint32_t>(t.spiked_indices.size());

        // sample voltages and chemicals
        t.sampled_voltages.assign(v.begin(), v.end());
        t.sampled_transmitters.assign(c_trans.begin(), c_trans.end());
        t.sampled_glucose.assign(c_glc.begin(), c_glc.end());

        t.output_decisions = evaluator_.evaluate(current_time_ms_);

        return t;
    }
} // namespace neuro