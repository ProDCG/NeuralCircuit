// forward-star edge table and ring buffer delays
#pragma once

#include "neuro/types.hpp"
#include <vector>
#include <cstdint>
#include <span>

namespace neuro {
    class DelayRingBuffer {
        public:
            DelayRingBuffer(size_t max_neurons = 1024, size_t max_delay_steps = 200);

            void resize(size_t max_neurons, size_t max_delay_steps);
            void clear() noexcept;

            // schedule delayed synaptic current arrival at (head + delay_steps) % max_delay
            inline void push(size_t target_idx, size_t delay_steps, float current_amount) noexcept {
                if (target_idx >= max_neurons_ || max_delay_steps_ == 0) return;
                size_t delay = (delay_steps == 0) ? 1 : delay_steps;
                size_t slot = (head_idx_ + delay) % max_delay_steps_;
                size_t flat_idx = target_idx * max_delay_steps_ + slot;
                buffer_[flat_idx] += current_amount;
            }

            // read and clear the currents arriving at the current simulation step
            inline void read_and_clear(std::span<float> out_I_syn) noexcept {
                size_t count = std::min(out_I_syn.size(), max_neurons_);
                for (size_t i = 0; i < count; i++) {
                    size_t flat_idx = i * max_delay_steps_ + head_idx_;
                    out_I_syn[i] += buffer_[flat_idx];
                    buffer_[flat_idx] = 0.0f; // clear slot for future curciular reuse
                }
                head_idx_ = (head_idx_ + 1) % max_delay_steps_;
            }

        private:
            size_t max_neurons_{0};
            size_t max_delay_steps_{0};
            size_t head_idx_{0};
            AlignedVector<float> buffer_;
    };

    struct OutgoingSynapses {
        const uint32_t* targets{nullptr};
        const float* weights{nullptr};
        const uint16_t* delay_steps{nullptr};
        uint32_t count{0};
    };

    class SynapseTable {
    public:
        void clear() noexcept;
        void add_synapse(uint32_t src, uint32_t dst, float weight, float delay_ms, float dt);
        void rebuild_index(size_t num_neurons);

        OutgoingSynapses get_outgoing(size_t src_idx) const noexcept {
            if (src_idx >= edge_offsets_.size() || edge_counts_[src_idx] == 0) {
                return {nullptr, nullptr, nullptr, 0};
            }
            uint32_t start = edge_offsets_[src_idx];
            uint32_t count = edge_counts_[src_idx];
            return {
                targets_.data() + start,
                weights_.data() + start,
                delays_.data() + start,
                count
            };
        }

        size_t size() const noexcept { return raw_src_.size(); }

        // direct access to raw edge descriptors for serialization
        const std::vector<uint32_t>& raw_src() const noexcept { return raw_src_; }
        const std::vector<uint32_t>& raw_dst() const noexcept { return raw_dst_; }
        const std::vector<float>& raw_weights() const noexcept { return raw_weights_; }
        const std::vector<float>& raw_delays_ms() const noexcept { return raw_delays_ms_; }
    private:
        // Raw definitions
        std::vector<uint32_t> raw_src_;
        std::vector<uint32_t> raw_dst_;
        std::vector<float> raw_weights_;
        std::vector<float> raw_delays_ms_;
        std::vector<uint16_t> raw_delay_steps_;

        // fast forward-star packed CSR index
        std::vector<uint32_t> targets_;
        std::vector<float> weights_;
        std::vector<uint16_t> delays_;
        std::vector<uint32_t> edge_offsets_;
        std::vector<uint32_t> edge_counts_;
    };

} // namespace mono