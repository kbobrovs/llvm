// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::INTEL::gpu;
using namespace cl::sycl;

void kernel(
    float *Ptr,
    accessor<float, 1, access::mode::read_write, access::target::global_buffer>
        Acc) SYCL_ESIMD_FUNCTION {

  constexpr int enable_all = 1;

  simd<uint32_t, 32> offsets32(0, 1);
  simd<uint32_t, 16> offsets16(0, 1);

  // (test0) 'gather': USM pointer version, full template and function parameter
  // set
  simd<float, 32> v0 = gather< // 32 elements = 16 blocks * 2 elements
      float,                   // element type
      16,                      // number of blocks
      2                        // number of elements per block
      >(Ptr,                   // base memory pointer
        offsets16,  // offsets to the base pointer in bytes per each block
        enable_all, // per-lane predicates, implicitly converted to
                    // simd<uint16_t,16>
        l1_cache_hint::none{}, l3_cache_hint::write_back{} // cache hints
  );

  // (test1) 'gather': same as above, the order of cache hints is reversed
  auto v1 = gather<float, 16, 2>(Ptr, offsets16, 1, l3_cache_hint::write_back{},
                                 l1_cache_hint::none{});

  // (test2) 'gather': accessor-based version, full template and function
  // parameter set. Does not support predication or varying block size.
  simd<float, 16> v2 = gather<float, 16>(
      Acc,       // accessor to a buffer
      offsets16, // offsets to the beginning of the buffer in bytes per each
                 //   block
      1024,      // global offset
      l1_cache_hint::none{}, l3_cache_hint::write_back{} // cache hints
  );

  // (test3) scalar load with cache hints.
  auto v4 = scalar_load<float>(Acc, 0, l1_cache_hint::none{},
                               l3_cache_hint::write_back{});

  // (test4) atomic operations with cache hints.
  {
    unsigned int *UPtr = reinterpret_cast<unsigned int *>(Ptr);
    simd<unsigned int, 16> src1 = 10, src2 = 10;
    // 0 src
    flat_atomic<EsimdAtomicOpType::ATOMIC_INC, unsigned int, 16>(
        UPtr, offsets16, enable_all, l1_cache_hint::none{},
        l3_cache_hint::write_back{});
    // 1 src
    flat_atomic<EsimdAtomicOpType::ATOMIC_ADD, unsigned int, 16>(
        UPtr, offsets16, src1, enable_all, l1_cache_hint::none{},
        l3_cache_hint::write_back{});
    // 2 src
    flat_atomic<EsimdAtomicOpType::ATOMIC_CMPXCHG, unsigned int, 16>(
        UPtr, offsets16, src1, src2, enable_all, l1_cache_hint::none{},
        l3_cache_hint::write_back{});
  }
}
