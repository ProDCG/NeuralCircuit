// dual-rate simulation coordinator
#pragma once

#include "neuro/neurons.hpp"
#include "neuro/synapses.hpp"
#include "neuro/stimulus.hpp"
#include "neuro/evaluator.hpp"
#include <cstddef>
#include <vector>

namespace neuro {
    struct EngineTelemetry {
        float time_ms{0.0f};
        uint64_t step_count{0};
        uint32_t active_spikes_count{0};
        std::vector<uint32_t> spiked_indices;
        std::vector<float> sampled_voltages;
        std::vector<float> sampled_transmitters;
        std::vector<float> sampled_glucose;
        std::vector<OutputChannelState> output_decisions;
    };

    class SimulationEngine {
        public:
        SimulationEngine(float dt_ms = 0.05f, float dt_chem_ms = 1.0f);

        void reset() noexcept;
        void step();
        void step_batch(size_t n_steps);

        // subsystem accessors
        NeuronPopulation& neurons() noexcept { return neurons_; }
        const NeuronPopulation& neurons() const noexcept { return neurons_; }

        SynapseTable& synapses() noexcept { return synapses_; }
        const SynapseTable& synapses() const noexcept { return synapses_; }

        DelayRingBuffer& ring_buffer() noexcept { return ring_buffer_; }
        const DelayRingBuffer& ring_buffer() const noexcept { return ring_buffer_; }

        StimulusEngine& stimulus() noexcept { return stimulus_; }
        const StimulusEngine& stimulus() const noexcept { return stimulus_; }

        OutputEvaluator& evaluator() noexcept { return evaluator_; }
        const OutputEvaluator& evaluator() const noexcept { return evaluator_; }

        // timing and state readouts
        float current_time_ms() const noexcept { return current_time_ms_; }
        uint64_t total_steps() const noexcept { return total_steps_; }
        float dt() const noexcept { return dt_; }
        float dt_chem() const noexcept { return dt_chem_; }

        void set_dt(float dt) noexcept;
        void set_dt_chem(float dt_chem) noexcept;

        // generate telemetry snapshot for the frontend
        EngineTelemetry snapshot_telemetry() const;
    private:
        float dt_{0.05f};
        float dt_chem_{1.0f};
        size_t chem_ratio_{20};
        size_t chem_step_counter_{0};

        float current_time_ms_{0.0f};
        uint64_t total_steps_{0};

        NeuronPopulation neurons_;
        SynapseTable synapses_;
        DelayRingBuffer ring_buffer_;
        StimulusEngine stimulus_;
        OutputEvaluator evaluator_;
    };
} // namespace neuro