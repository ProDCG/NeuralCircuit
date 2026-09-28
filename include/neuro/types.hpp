// aligned allocators, strong IDs, enums
#pragma once

#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>
#include <string>

namespace neuro {
    struct NeuronId {
        uint32_t value{0};
        constexpr NeuronId() = default;
        // this means we can define this at compile time and save allocation/compute time during runtime
        constexpr explicit NeuronId(uint32_t v): value(v) {}
        constexpr bool operator==(const NeuronId& o) const = default;
        // compound relational operator, allowing us to define <, <=, >, >= simultaneously.
        // const=default means automatically define the boilerplate code underneath for the impl.
        constexpr auto operator<=>(const NeuronId& o) const = default;
    };

    struct EdgeId {
        uint32_t value{0};
        constexpr EdgeId() = default;
        constexpr explicit EdgeId(uint32_t v) : value(v) {}
        constexpr bool operator==(const EdgeId& o) const = default;
        constexpr auto operator<=>(const EdgeId& o) const = default;
    };

    enum class NeuronType : uint8_t {
        Excitatory = 0, // Glutamatergic (e.g., NGN2)
        Inhibitory = 1 // GABAergic (e.g., ASCL1)
    };

    enum class StimulusWaveform : uint8_t {
        None = 0,
        DCCurrent = 1,
        PulseTrain = 2,
        PoissonSpikes = 3,
        SineWave = 4
    };

    enum class OutputDecodeMode : uint8_t {
        SpikeCount = 0,
        FiringRateHz = 1,
        FirstSpikeLatency = 2
    };

    struct Vec3f {
        float x{0.0f};
        float y{0.0f};
        float z{0.0f}; // micrometer scale?

        constexpr Vec3f() = default;
        constexpr Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    };

    // 64-byte cache-line aligned allocator for AVX2 / AVX-512 vectorization
    template <typename T, size_t Alignment = 64>
    struct AlignedAllocator {
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        template <typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        AlignedAllocator() noexcept = default;
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        T* allocate(size_t n) {
            if (n == 0) return nullptr;
            void* ptr = nullptr;
#if defined (_MSC_VER)
            ptr = _aligned_malloc(n * sizeof(T), Alignment);
            if (!ptr) throw std::bad_alloc()
#else
            if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
                throw std::bad_alloc();
            }
#endif
            return static_cast<T*>(ptr);
        }

        void deallocate(T* p, size_t) noexcept {
#if defined(_MSC_VER)
            _aligned_free(p);
#else
            free(p);
#endif
        }

        template <typename U>
        bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
        template <typename U>
        bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
    };

    template <typename T>
    using AlignedVector = std::vector<T, AlignedVector<T, 64>>;
} // namespace neuro