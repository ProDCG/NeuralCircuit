#include "neuro/stimulus.hpp"
#include <cmath>
#include <numbers>

namespace neuro {
    StimulusEngine::StimulusEngine() : rng_(42) {}

    void StimulusEngine::set_channel(const StimulusChannel& channel) {
        for (auto& ch : channels_) {
            if (ch.target_neuron_idx == channel.target_neuron_idx) {
                ch = channel;
                return;
            }
        }
        channels_.push_back(channel);
    }

    void StimulusEngine::remove_channel(size_t target_neuron_idx) {
        std::erase_if(channels_, [target_neuron_idx](const auto& ch) {
            return ch.target_neuron_idx == target_neuron_idx;
        });
    }

    void StimulusEngine::clear() noexcept {
        channels_.clear();
    }

    void StimulusEngine::apply(float current_time_ms, float dt, std::span<float> I_ext) {
        constexpr float pi = 3.14159265358979323846f;

        for (const auto& ch : channels_) {
            if (!ch.active || ch.target_neuron_idx >= I_ext.size()) continue;

            float injected = 0.0f;
            switch (ch.waveform) {
                case StimulusWaveform::DCCurrent: {
                    injected = ch.amplitude_pa;
                    break;
                }
                case StimulusWaveform::PulseTrain: {
                    if (ch.frequency_hz > 0.0f) {
                        float period_ms = 1000.0f / ch.frequency_hz;
                        float phase_offset_ms = (ch.phase_deg / 360.0f) * period_ms;
                        float t = std::fmod(current_time_ms + phase_offset_ms, period_ms);
                        if (t < 0.0f) t += period_ms;
                        if (t < period_ms * ch.duty_cycle) {
                            injected = ch.amplitude_pa;
                        }
                    }
                    break;
                }
                case StimulusWaveform::PoissonSpikes: {
                    // probability of poisson event in time dt: P = lambda * dt_seconds
                    float prob = ch.frequency_hz * (dt / 1000.0f);
                    if (uniform_dist_(rng_) < prob) {
                        injected = ch.amplitude_pa;
                    }
                    break;
                }
                case StimulusWaveform::SineWave: {
                    if (ch.frequency_hz > 0.0f) {
                        float omega = 2.0f * pi * (ch.frequency_hz / 1000.0f);
                        float phase_rad = ch.phase_deg * (pi / 180.0f);
                        injected = ch.amplitude_pa * std::sin(omega * current_time_ms + phase_rad);
                    }
                    break;
                }
                case StimulusWaveform::None:
                default:
                    break;
            }

            I_ext[ch.target_neuron_idx] += injected;
        }
    }
} // namespace neuro