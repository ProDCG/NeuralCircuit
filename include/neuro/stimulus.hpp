// real-time waveform generators
#pragma once

#include "neuro/types.hpp"
#include <vector>
#include <random>
#include <span>

namespace neuro {
    struct StimulusChannel {
        size_t target_neuron_idx{0};
        StimulusWaveform waveform{StimulusWaveform::None};
        float amplitude_pa{15.0f};
        float frequency_hz{50.0f};
        float duty_cycle{0.5f};
        float phase_deg{0.0f};
        bool active{true};
    };

    class StimulusEngine {
        public:
            StimulusEngine();

            void set_channel(const StimulusChannel& channel);
            void remove_channel(size_t target_neuron_idx);
            void clear() noexcept;

            // apply active stimuli to the external current array for the current time step
            void apply(float current_time_ms, float dt, std::span<float> I_ext);

            const std::vector<StimulusChannel>& channels() const noexcept { return channels_; }
        private:
            std::vector<StimulusChannel> channels_;
            std::mt19937 rng_;
            std::uniform_real_distribution<float> uniform_dist_{0.0f, 1.0f};
    };
} // namespace neuro