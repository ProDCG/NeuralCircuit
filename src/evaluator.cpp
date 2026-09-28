#include "neuro/evaluator.hpp"
#include <algorithm>

namespace neuro {
    OutputEvaluator::OutputEvaluator() = default;

    void OutputEvaluator::set_channel(const OutputChannelConfig& config) {
        for (size_t i = 0; i < channels_.size(); ++i) {
            if (channels_[i].target_neuron_idx == config.target_neuron_idx) {
                channels_[i] = config;
                return;
            }
        }
        channels_.push_back(config);
        records_.push_back({config.target_neuron_idx, {}, 0});
    }

    void OutputEvaluator::remove_channel(size_t target_neuron_idx) {
        for (size_t i = 0; i < channels_.size(); ++i) {
            if (channels_[i].target_neuron_idx == target_neuron_idx) {
                channels_.erase(channels_.begin() + i);
                records_.erase(records_.begin() + i);
                return;
            }
        }
    }

    void OutputEvaluator::clear() noexcept {
        channels_.clear();
        records_.clear();
    }

    void OutputEvaluator::record_step(float current_time_ms, std::span<const uint8_t> spiked) {
        for (size_t i = 0; i < channels_.size(); ++i) {
            size_t target = channels_[i].target_neuron_idx;
            if (target < spiked.size() && spiked[target]) {
                records_[i].timestamps.push_back(current_time_ms);
                records_[i].lifetime_spikes++;
            }

            // evict timestamps outside the rolling window
            float cutoff = current_time_ms - channels_[i].window_ms;
            while (!records_[i].timestamps.empty() && records_[i].timestamps.front() < cutoff) {
                records_[i].timestamps.pop_front();
            }
        }
    }

    std::vector<OutputChannelState> OutputEvaluator::evaluate(float current_time_ms) const {
        std::vector<OutputChannelState> states;
        states.reserve(channels_.size());

        for (size_t i = 0; i < channels_.size(); ++i) {
            const auto& cfg = channels_[i];
            const auto& rec = records_[i];

            float count = static_cast<float>(rec.timestamps.size());
            float window_sec = cfg.window_ms / 1000.0f;
            float rate_hz = (window_sec > 0.0f) ? (count / window_sec) : 0.0f;

            float raw_value = 0.0f;
            bool decision = false;

            switch (cfg.mode) {
                case OutputDecodeMode::SpikeCount:
                    raw_value = count;
                    decision = (count >= cfg.threshold);
                    break;
                case OutputDecodeMode::FiringRateHz:
                    raw_value = rate_hz;
                    decision = (rate_hz >= cfg.threshold);
                    break;
                case OutputDecodeMode::FirstSpikeLatency:
                    if (!rec.timestamps.empty()) {
                        float latency = rec.timestamps.front() - (current_time_ms - cfg.window_ms);
                        raw_value = latency;
                        decision = (latency <= cfg.threshold);
                    } else {
                        raw_value = 9999.0f;
                        decision = false;
                    }
                    break;
            }

            states.push_back({
                cfg.target_neuron_idx,
                raw_value,
                rate_hz,
                decision,
                rec.lifetime_spikes
            });
        }

        return states;
    }
} // namespace neuro