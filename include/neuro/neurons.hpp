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
        std::span<const float> v() const noexcept { return {v_.data(), count_}; }

        std::span<float> u() noexcept { return { u_.data(), count_}; }
        std::span<const float> u() const noexcept { return { u_.data(), count_}; }

        std::span<float> I_syn() noexcept { return {I_syn_.data(), count_}; }
        std::span<float> I_ext() noexcept { return {I_ext_.data(), count_}; }
        std::span<float> I_base() noexcept { return {I_base_.data(), count_}; }

        std::span<const uint8_t> spiked() const noexcept { return {spiked_.data(), count_}; }
        
        // chemical and metabolic accessors
        std::span<const float> C_trans() const noexcept { return {C_trans_.data(), count_}; }
        std::span<const float> C_glc() const noexcept { return {C_glc_.data(), count_}; }
        std::span<const float> metabolic_gate() const noexcept { return {metabolic_gate_.data(), count_}; }
        std::span<const float> receptor_gain() const noexcept { return {receptor_gain_.data(), count_}; }

        // parameters and metadata accessors
        std::span<IzhikevichParams> izh_params() noexcept { return {izh_params_.data(), count_}; }
        std::span<const IzhikevichParams> izh_params() const noexcept { return {izh_params_.data(), count_}; }

        std::span<LocalChemParams> chem_params() noexcept { return {chem_params_.data(), count_}; }
        std::span<const LocalChemParams> chem_params() const noexcept { return {chem_params_.data(), count_}; }

        std::span<NeuronType> types() noexcept { return {types_.data(), count_}; }
        std::span<const NeuronType> types() const noexcept { return {types_.data(), count_}; }

        std::span<Vec3f> positions() noexcept { return {positions_.data(), count_}; }
        std::span<const Vec3f> positions() const noexcept { return {positions_.data(), count_}; }

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