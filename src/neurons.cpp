#include "neuro/neurons.hpp"
#include <cmath>
#include <algorithm>

namespace neuro {
    NeuronPopulation::NeuronPopulation(size_t reserve_capacity) {
        v_.reserve(reserve_capacity);
        u_.reserve(reserve_capacity);
        I_syn_.reserve(reserve_capacity);
        I_ext_.reserve(reserve_capacity);
        I_base_.reserve(reserve_capacity);
        spiked_.reserve(reserve_capacity);

        C_trans_.reserve(reserve_capacity);
        C_glc_.reserve(reserve_capacity);
        metabolic_gate_.reserve(reserve_capacity);
        receptor_gain_.reserve(reserve_capacity);

        izh_params_.reserve(reserve_capacity);
        chem_params_.reserve(reserve_capacity);
        types_.reserve(reserve_capacity);
        positions_.reserve(reserve_capacity);
    }

    void NeuronPopulation::clear() noexcept {
        count_ = 0;
        v_.clear();
        u_.clear();
        I_syn_.clear();
        I_ext_.clear();
        I_base_.clear();
        spiked_.clear();

        C_trans_.clear();
        C_glc_.clear();
        metabolic_gate_.clear();
        receptor_gain_.clear();

        izh_params_.clear();
        chem_params_.clear();
        types_.clear();
        positions_.clear();

        count_--;
    }

    void NeuronPopulation::on_neuron_spiked(size_t dense_idx) {
        if (dense_idx >= count_) return;
        const auto& chem = chem_params_[dense_idx];

        // release quantal transmitter to local micro-domain
        C_trans_[dense_idx] += chem.q_release;

        // consume glucose for action potential repolarization
        float glc = C_glc_[dense_idx] - chem.glc_cost_per_spike;
        C_glc_[dense_idx] = std::max(0.0f, glc);
    }

    void NeuronPopulation::integrate_electrical(float dt) {
        // vectorized simd hot loop
        #pragma omp simd
        for (size_t i = 0; i < count_; ++i) {
            // gated inputs by locla metabolic health
            float gate = metabolic_gate_[i];
            float I_tot = (I_syn_[i] + I_ext_[i] + I_base_[i]) * gate;

            float v_curr = v_[i];
            float u_curr = u_[i];
            const auto& p = izh_params_[i];

            // izhikevich numerical integration: dv/dt = 0.04v^2 + 5v + 140 - u + I
            float v_next = v_curr + dt * (0.04f * v_curr * v_curr + 5.0f * v_curr + 140.0f - u_curr + I_tot);
            float u_next = u_curr + dt * (p.a * (p.b * v_curr - u_curr));

            bool spike = (v_next >= 30.0f);
            spiked_[i] = spike ? 1 : 0;

            // reset
            v_[i] = spike ? p.c : v_next;
            u_[i] = spike ? (u_next + p.d) : u_next;

            // clear synaptic current for next step
            I_syn_[i] = 0.0f;
        }
    }

    void NeuronPopulation::integrate_chemical(float dt_chem) {
        for (size_t i = 0; i < count_; ++i) {
            const auto& chem = chem_params_[i];

            // local transmitter reuptake / clearnace: dC/dt = -C / tau
            float decay = std::exp(-dt_chem / std::max<float>(0.1f, chem.tau_clearance_ms));
            C_trans_[i] *= decay;

            // receptor saturation / hill kinetics gain:
            // gain = 1 + (C^n / (Kd^n + C^n))
            float c = C_trans_[i];
            if (c > 0.001f) {
                float c_pow = std::pow(c, chem.hill_coeff);
                float kd_pow = std::pow(chem.kd_saturation, chem.hill_coeff);
                receptor_gain_[i] = 1.0f + (c_pow / (kd_pow + c_pow));
            } else {
                receptor_gain_[i] = 1.0f;
            }

            // glucose micro-channel replenishment
            float glc = C_glc_[i];
            glc += dt_chem * chem.glc_replenish_rate * (chem.glc_max - glc);
            glc = std::clamp<float>(glc, 0.0f, chem.glc_max);
            C_glc_[i] = glc;

            // metabolic exhaustions
            constexpr float k_steepness = 30.0f;
            metabolic_gate_[i] = 1.0f / (1.0f + std::exp(-k_steepness * (glc - chem.glc_crit)));
        }
    }
}