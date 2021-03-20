// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-device-only -fsyntax-only -pedantic -Xclang -verify -DSRC1 %s
// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-device-only -fsyntax-only -pedantic -Xclang -verify -DSRC2 %s

// This test checks few main error diagnostics of the ESIMD memory API
// NOTE: The test is split into two so that all error messages are output, not
// only first few. (-ferror-limit=... does not help).

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

using namespace sycl::INTEL::gpu;
using namespace cl::sycl;

#if SRC1
void kernel1(
    int *Ptr,
    accessor<int, 1, access::mode::read_write, access::target::global_buffer>
        Acc) SYCL_ESIMD_FUNCTION {

  simd<uint32_t, 32> offsets32(0, 1);
  simd<uint32_t, 16> offsets16(0, 1);

  constexpr int enable_all = 1;

  // 1) Check that only cache hints are allowed as vararg parameters
  // expected-error@CL/sycl/INTEL/esimd/esimd_memory.hpp:100 {{Only cache hint arguments are allowed, and only 0 or 1 occurrences each}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:156 {{}}
  // expected-note@+1 {{}}
  { auto v = gather<int, 16>(Ptr, offsets16, enable_all, 1); }

  // 2) Check that a cache hint can't be duplicated
  // expected-error@CL/sycl/INTEL/esimd/esimd_memory.hpp:100 {{Only cache hint arguments are allowed, and only 0 or 1 occurrences each}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:156 {{}}
  // expected-note@+2 {{}}
  {
    auto v = gather<int, 16>(Ptr, offsets16, enable_all, l1_cache_hint::none{},
                             l1_cache_hint::none{});
  }

  // 3) Check that flat_atomic with missing argument gives an error
  // expected-error@+6 {{no matching function for call to 'flat_atomic'}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:599 {{}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:619 {{}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:640 {{}}
  {
    unsigned int *UPtr = reinterpret_cast<unsigned int *>(Ptr);
    flat_atomic<EsimdAtomicOpType::ATOMIC_ADD, unsigned int, 16>(UPtr,
                                                                 offsets16, 1);
  }
}
#endif

#if SRC2
void kernel2(
    int *Ptr,
    accessor<int, 1, access::mode::read_write, access::target::global_buffer>
        Acc) SYCL_ESIMD_FUNCTION {

  simd<uint32_t, 16> offsets16(0, 1);

  // 4) Check that flat_atomic with redundant argument gives an error
  // expected-error@CL/sycl/INTEL/esimd/esimd_memory.hpp:100 {{Only cache hint arguments are allowed, and only 0 or 1 occurrences each}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:623 {{}}
  // expected-note@+4 {{}}
  {
    unsigned int *UPtr = reinterpret_cast<unsigned int *>(Ptr);
    simd<unsigned int, 16> src1 = 10, src2 = 10;
    flat_atomic<EsimdAtomicOpType::ATOMIC_ADD, unsigned int, 16>(
        UPtr, offsets16, src1, src2, 1);
  }
}
#endif
