// SoA electrical and local chemical states

#pragma once

#include "neuro/types.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace neuro {
    struct IzhikevichParams {
        float a{0.02f};     // Time scale of recovery variable u
        float b{0.20f};     // Sensitivity of u to subtreshold fluctuations of v
        float c{-0.65f};    // After-spike reset potential of v (mV)
        float d{8.0f};      // After-spike reset increment of u
        float v_init{-65.0f};
        float u_init{-13.0f};
    };

    struct LocalChemParams {
        float tau_clearance_ms{3.0f};       // local neurotransmitter reuptake time constant
        float q_release{1.0f};              // quantal transmitter released per spike
        float kd_saturation{2.0f};          // half-saturation constant (Hill equation)
        float hill_coeff{2.0f};             // Hill exponent for receptor saturation
        float glc_max{1.0f};                // Baseline maximum glucose level (normalized)
        float glc_replenish_rate{0.05f};    // Replenishment rate back to glc_max (1/ms)
        float glc_cost_per_spike{0.06f};    // Metabolic glucose consumed per action potential
        float glc_crit{0.15f};              // Critical threshold below which neuron starves
    }; 

    class NeuronPopulation {
    public:
        explicit NeuronPopulation(size_t reserve_capacity = 1024);

        size_t size() const noexcept { return count_; }
        void clear() noexcept;

        size_t add_neuron(
            NeuronType type,
            const IzhikevichParams& izh,
            const LocalChemParams& chem,
            Vec3f pos
        );

        void remove_neuron(size_t dense_idx);

        // Fast electrical step (e.g., 0.05 ms) - SIMD vectorized euler integration
        void integrate_electrical(float dt);

        // Slow chemical step (e.g., 0.05 ms) - Transmitter clearance and glucose recovery
        void integrate_chemical(float dt_chem);

        // Spike-triggered chemical release and energy consumption
        void on_neuron_spiked(size_t dense_idx);

        // accessors (span for safe zero-copy access)
        std::span<float> v() noexcept { return {v_.data(), count_}; }
    private:
        size_t count_{0};

        // hot electrical data (data cache residency)
        AlignedVector<float> v_;
        AlignedVector<float> u_;
        AlignedVector<float> I_syn_;
        AlignedVector<float> I_ext_;
        AlignedVector<float> I_base_;
        AlignedVector<uint8_t> spiked_;

        // hot/warm chemical and metabolic data
        AlignedVector<float> C_trans_; // local transmitter concentration?
        AlignedVector<float> C_glc_; // local glucose energy level [0..1]
        AlignedVector<float> metabolic_gate_; // [0..1] gating multiplier
        AlignedVector<float> receptor_gain_; // [0.<.1] saturation gain multiplier

        // parameter and configuration
        AlignedVector<IzhikevichParams> izh_params_;
        AlignedVector<LocalChemParams> chem_params_;
        AlignedVector<NeuronType> types_;

        // cold spatial data
        std::vector<Vec3f> positions_;
    };
} // namespace neuro