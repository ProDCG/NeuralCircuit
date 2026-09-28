// output rate/count decoders and truth tables
#pragma once

#include "neuro/types.hpp"
#include <vector>
#include <deque>
#include <span>

namespace neuro {
    struct OutputChannelConfig {
        size_t target_neuron_idx{0};
        OutputDecodeMode mode{OutputDecodeMode::SpikeCount};
        float window_ms{50.0f};
        float threshold{3.0f}; // >= threshold triggers digital boolean '1'
    };

    struct OutputChannelState {
        size_t target_neuron_idx{0};
        float raw_value{0.0f};
        float firing_rate_hz{0.0f};
        bool boolean_decision{false};
        uint32_t total_spikes{0};
    };

    struct TruthTableEntry{
        std::vector<bool> inputs;
        bool expected_output;
        bool actual_output{false};
        bool passed{false};
    };

    class OutputEvaluator {
        public:
            OutputEvaluator();

            void set_channel(const OutputChannelConfig& config);
            void remove_channel(size_t target_neuron_idx);
            void clear() noexcept;

            // called on every simulation step to track timestamp of spiking events
            void record_step(float current_time_ms, std::span<const uint8_t> spiked);

            // evaluate all configured output channels
            std::vector<OutputChannelState> evaluate(float current_time_ms) const;

            const std::vector<OutputChannelConfig>& channels() const noexcept { return channels_; }
        private:
            struct SpikeRecord {
                size_t target_neuron_idx;
                std::deque<float> timestamps;
                uint32_t lifetime_spikes{0};
            };

            std::vector<OutputChannelConfig> channels_;
            std::vector<SpikeRecord> records_;
    };
} // namespace neuro