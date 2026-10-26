// Production flat 2-hop allreduce (fp16, strict relay ordering, multi-CTA
// slices). Sibling of csrc/sm70_hier_ar.cu; loaded through ctypes by
// vllm/distributed/device_communicators/sm70_flat_ar.py and selected with
// VLLM_SM70_AR_IMPL=flat (the hier route stays the default).
//
// Protocol (all traffic on NVLink; SYS peers go through a common NVLink relay):
//   1. push my slice into the NVLink neighbours' inbound slots (rank-indexed)
//   2. __threadfence, publish readiness in MY OWN region (local store)
//   3. poll each neighbour's flag in the NEIGHBOUR's region
//   4. strict ordering: a rank that hosts a 2-hop peer re-publishes that
//      peer's readiness in its own region, and readers wait for THAT flag
//      (without this the relayed read races the relay's own ready flag)
//   5. reduce every source in fixed global-rank order (bitwise-stable)
//
// World size, neighbour count and relay count are RUNTIME parameters; the host
// derives the adjacency from `nvidia-smi topo -m` for the devices the group
// actually uses. TP8 therefore runs with 4 neighbours + 3 relays per rank, and
// a fully NVLink-connected TP4 (devices 0-3 or 4-7) runs with 3 neighbours, no
// relays, i.e. the protocol degenerates to a single hop. The W == 8 case keeps
// a compile-time specialisation so its codegen and reduction order are
// unchanged.
//
// Why the layout below: every rank needs room for the peers' pushes
// (W payloads per slot), so the region is W x the hier in_buf. That is why this
// route declares a takeover cap (bytes), and everything larger stays on the
// next backend.
//
// Epoch: device-side per-slice counters (atomicAdd), so a CUDA graph replay
// advances them without any host participation.
//
// Build: CMake target `sm70_flat_ar` (see CMakeLists.txt / build_sm70.sh).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define F_MAXW 16
#define F_S_MAX 16
#define F_THREADS 512
#define F_DATA_OFF 2048  // 64 + F_S_MAX*F_MAXW*4 flags (1088 B) fits below this
#define F_SPIN_LIMIT 20000000ULL

struct FlatArgs {
  unsigned long long base[F_MAXW];
  int neigh[F_MAXW];
  int sysp[F_MAXW];
  int relay[F_MAXW];
  int host[F_MAXW];
  int n_host;
  int n_sys;
  int n_neigh;
  int n_world;
  int rank;
  int n;
  int n_max;
};

__device__ __forceinline__ void st_flag_local(uint32_t* addr, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_flag_acquire(const uint32_t* addr) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(addr) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t pack2(float x, float y) {
  __half2 h = __floats2half2_rn(x, y);
  return (uint32_t)__half_as_ushort(__low2half(h)) |
         ((uint32_t)__half_as_ushort(__high2half(h)) << 16);
}

// W > 0 pins the world size at compile time (W == 8 keeps the production
// codegen); W == 0 reads it from the argument struct. NNEIGH / NSYS do the
// same for the neighbour and relay counts, so the production shape (8, 4, 3)
// compiles back to the original fully unrolled loops.
template <int W, int NNEIGH, int NSYS>
__global__ void flat_kernel(FlatArgs a, const half* __restrict__ in,
                            half* __restrict__ out) {
  const int tid = threadIdx.x;
  const int b = blockIdx.x;
  const int S = gridDim.x;
  const int ns = a.n / S;
  const int off = ns * b;
  const int nw = (W > 0) ? W : a.n_world;
  const int nneigh = (NNEIGH > 0) ? NNEIGH : a.n_neigh;
  const int nsys = (NSYS > 0) ? NSYS : a.n_sys;

  // Device-side per-slice epoch: survives CUDA graph replay untouched.
  __shared__ uint32_t s_e;
  if (tid == 0) s_e = atomicAdd(reinterpret_cast<uint32_t*>(a.base[a.rank]) + b, 1u) + 1u;
  __syncthreads();
  const uint32_t e = s_e;
  const int s = (int)(e & 1u);

  const size_t slot_b = (size_t)nw * (size_t)a.n_max * 2;
  auto pay = [&](unsigned long long base, int r) {
    return reinterpret_cast<half*>(base + F_DATA_OFF + (size_t)s * slot_b) +
           (size_t)r * a.n_max + off;
  };
  auto flagp = [&](unsigned long long base, int r) {
    return reinterpret_cast<uint32_t*>(base + 64) + (size_t)b * nw + r;
  };

  // Push my slice into every neighbour's slot for my rank. The warps are split
  // evenly over the neighbours; with 16 warps and 4 neighbours this is exactly
  // the original (warp >> 2, warp & 3) mapping.
  {
    const int warp = tid >> 5, lane = tid & 31;
    constexpr int NW = F_THREADS / 32;
    const int nv4 = ns / 8;
    const uint4* s4 = reinterpret_cast<const uint4*>(in + off);
    if (NNEIGH > 0) {
      // Compile-time shape: exactly the original mapping (warps split evenly
      // over the neighbours, 4 warps x stride 128 for the production tables).
      constexpr int NSUB = NW / (NNEIGH > 0 ? NNEIGH : 1);
      const int peer_j = warp / NSUB, sub = warp % NSUB;
      if (peer_j < NNEIGH) {
        uint4* d4 = reinterpret_cast<uint4*>(pay(a.base[a.neigh[peer_j]],
                                                a.rank));
        for (int i = sub * 32 + lane; i < nv4; i += NSUB * 32) d4[i] = s4[i];
      }
    } else {
      int n_sub = NW / nneigh;
      if (n_sub < 1) n_sub = 1;
      const int step = (NW + n_sub - 1) / n_sub;
      const int sub = warp % n_sub;
      for (int j = warp / n_sub; j < nneigh; j += step) {
        uint4* d4 = reinterpret_cast<uint4*>(pay(a.base[a.neigh[j]], a.rank));
        for (int i = sub * 32 + lane; i < nv4; i += n_sub * 32) d4[i] = s4[i];
      }
    }
  }
  __syncthreads();
  __threadfence();
  __syncthreads();

  if (tid == 0) st_flag_local(flagp(a.base[a.rank], a.rank), e);
  __syncthreads();

  if (tid < nneigh) {
    const int nb = a.neigh[tid];
    const uint32_t* f = flagp(a.base[nb], nb);
    unsigned long long spins = 0;
    while (ld_flag_acquire(f) != e) {
      if (++spins > F_SPIN_LIMIT) break;  // never wedge the GPU
    }
  }
  __syncthreads();

  // strict relay ordering: re-flag the 2-hop peers whose data landed here
  if (tid < a.n_host) st_flag_local(flagp(a.base[a.rank], a.host[tid]), e);
  __syncthreads();
  if (tid < nsys) {
    const int rl = a.relay[tid], p = a.sysp[tid];
    const uint32_t* f = flagp(a.base[rl], p);
    unsigned long long spins = 0;
    while (ld_flag_acquire(f) != e) {
      if (++spins > F_SPIN_LIMIT) break;
    }
  }
  __syncthreads();

  // Sized by the compile-time world when there is one, so the W == 8
  // instantiation keeps the original 8-entry array (and its register count).
  const half* src[W > 0 ? W : F_MAXW];
#pragma unroll
  for (int r = 0; r < nw; ++r) src[r] = pay(a.base[a.rank], r);
  src[a.rank] = in + off;
  for (int j = 0; j < nsys; ++j)
    src[a.sysp[j]] = pay(a.base[a.relay[j]], a.sysp[j]);

  const int nv4 = ns / 8;
  for (int v = tid; v < nv4; v += F_THREADS) {
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int r = 0; r < nw; ++r) {
      uint4 u = *reinterpret_cast<const uint4*>(src[r] + v * 8);
      uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        __half2 h = __halves2half2(__ushort_as_half((unsigned short)(q[k] & 0xffffu)),
                                   __ushort_as_half((unsigned short)(q[k] >> 16)));
        float2 f = __half22float2(h);
        acc[2 * k] += f.x;
        acc[2 * k + 1] += f.y;
      }
    }
    uint4 o;
    o.x = pack2(acc[0], acc[1]);
    o.y = pack2(acc[2], acc[3]);
    o.z = pack2(acc[4], acc[5]);
    o.w = pack2(acc[6], acc[7]);
    *reinterpret_cast<uint4*>(out + off + v * 8) = o;
  }
}

extern "C" int flat_n_max(int max_bytes) {
  int n_max = max_bytes / 2;
  n_max = (n_max + 127) & ~127;
  return n_max < 128 ? 128 : n_max;
}

extern "C" int flat_region_bytes(int max_bytes, int n_world) {
  if (n_world < 1) n_world = 1;
  if (n_world > F_MAXW) n_world = F_MAXW;
  const int n_max = flat_n_max(max_bytes);
  return F_DATA_OFF + (int)(2 * (size_t)n_world * (size_t)n_max * 2);
}

extern "C" int flat_launch(const unsigned long long* bases, const int* neigh,
                           const int* sysp, const int* relay, const int* host,
                           int n_host, int n_sys, int n_neigh, int n_world,
                           int rank, int n, int n_max, int s_blocks,
                           const void* in, void* out, void* stream) {
  if (s_blocks < 1 || s_blocks > F_S_MAX) return -1;
  if (n <= 0 || n > n_max) return -2;
  if (n % (128 * s_blocks) != 0) return -3;
  if (n / s_blocks < 512) return -4;
  if (n_world < 1 || n_world > F_MAXW) return -5;
  if (rank < 0 || rank >= n_world) return -6;
  if (n_neigh < 0 || n_neigh > n_world || n_sys < 0 || n_sys > n_world) {
    return -6;
  }
  if (n_host < 0 || n_host > F_MAXW) return -6;
  FlatArgs a{};
  for (int i = 0; i < n_world; ++i) a.base[i] = bases[i];
  for (int i = 0; i < n_neigh; ++i) {
    if (neigh[i] < 0 || neigh[i] >= n_world) return -6;
    a.neigh[i] = neigh[i];
  }
  for (int i = 0; i < n_sys; ++i) {
    if (sysp[i] < 0 || sysp[i] >= n_world) return -6;
    if (relay[i] < 0 || relay[i] >= n_world) return -6;
    a.sysp[i] = sysp[i];
    a.relay[i] = relay[i];
  }
  for (int i = 0; i < n_host; ++i) {
    if (host[i] < 0 || host[i] >= n_world) return -6;
    a.host[i] = host[i];
  }
  a.n_host = n_host;
  a.n_sys = n_sys;
  a.n_neigh = n_neigh;
  a.n_world = n_world;
  a.rank = rank;
  a.n = n;
  a.n_max = n_max;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const half* ip = reinterpret_cast<const half*>(in);
  half* op = reinterpret_cast<half*>(out);
  if (n_world == 8 && n_neigh == 4 && n_sys == 3) {
    // Production topology: compile-time tables, identical to the original
    // 8-rank kernel (same unrolling, same register budget).
    flat_kernel<8, 4, 3><<<s_blocks, F_THREADS, 0, st>>>(a, ip, op);
  } else {
    flat_kernel<0, 0, 0><<<s_blocks, F_THREADS, 0, st>>>(a, ip, op);
  }
  return (int)cudaGetLastError();
}
