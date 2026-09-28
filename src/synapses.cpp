#include "neuro/synapses.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neuro {

    DelayRingBuffer::DelayRingBuffer(size_t max_neurons, size_t max_delay_steps) {
        resize(max_neurons, max_delay_steps);
    }

    void DelayRingBuffer::resize(size_t max_neurons, size_t max_delay_steps) {
        max_neurons_ = max_neurons;
        max_delay_steps_ = max_delay_steps;
        head_idx_ = 0;
        buffer_.assign(max_neurons_ * max_delay_steps_, 0.0f);
    }

    void DelayRingBuffer::clear() noexcept {
        head_idx_ = 0;
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    }

    void SynapseTable::clear() noexcept {
        raw_src_.clear();
        raw_dst_.clear();
        raw_weights_.clear();
        raw_delays_ms_.clear();
        targets_.clear();
        weights_.clear();
        delays_.clear();
        edge_offsets_.clear();
        edge_counts_.clear();
    }

    void SynapseTable::add_synapse(uint32_t src, uint32_t dst, float weight, float delay_ms, float dt) {
        raw_src_.push_back(src);
        raw_dst_.push_back(dst);
        raw_weights_.push_back(weight);
        raw_delays_ms_.push_back(delay_ms);

        uint16_t steps = static_cast<uint16_t>(std::max(1.0f, std::round(delay_ms / dt)));
        raw_delay_steps_.push_back(steps);
    }

    void SynapseTable::rebuild_index(size_t num_neurons) {
        size_t num_edges = raw_src_.size();
        edge_offsets_.assign(num_neurons, 0);
        edge_counts_.assign(num_neurons, 0);

        if (num_edges == 0) return;

        // sort indices by source neuron index
        std::vector<size_t> p(num_edges);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), [&](size_t i, size_t j) {
            if (raw_src_[i] != raw_src_[j]) return raw_src_[i] < raw_src_[j];
            return raw_dst_[i] < raw_dst_[j];
        });

        targets_.resize(num_edges);
        weights_.resize(num_edges);
        delays_.resize(num_edges);

        // pack contiguous arrays
        for (size_t k = 0; k < num_edges; ++k) {
            size_t orig = p[k];
            targets_[k] = raw_dst_[orig];
            weights_[k] = raw_weights_[orig];
            delays_[k] = raw_delay_steps_[orig];
        }

        // populate csr offset and count tables
        for (size_t k = 0; k < num_edges; ++k) {
            uint32_t src = raw_src_[p[k]];
            if (src < num_neurons) {
                if (edge_counts_[src] == 0) {
                    edge_offsets_[src] = static_cast<uint32_t>(k);
                }
                edge_counts_[src]++;
            }
        }
    }
} // namespace neuro