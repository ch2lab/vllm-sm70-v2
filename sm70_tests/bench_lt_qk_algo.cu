// Is the service's QK^T (985 us in the prefill trace) leaving a better cublasLt
// algorithm on the table?  A first probe said 845.6 us for the same GEMM, but
// across processes the same candidate measured 995.7 / 847.3 / 845.6 us, and
// cross-process absolutes are not comparable on this box (clock drift).
//
// So: time every candidate for the QK^T shape, interleaved inside ONE process,
// over several rounds, and report the full matrix.  A candidate only counts as
// a real win if it is consistently below the default across rounds.
//
// build: nvcc -O3 -std=c++14 -arch=sm_70 ltqk.cu -lcublasLt -o ltqk
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x)                                                          \
  do {                                                                 \
    cublasStatus_t s_ = (x);                                           \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                 \
      printf("cublas err %d at line %d\n", (int)s_, __LINE__);         \
      exit(1);                                                         \
    }                                                                  \
  } while (0)
#define CU(x)                                                                \
  do {                                                                       \
    cudaError_t e_ = (x);                                                    \
    if (e_ != cudaSuccess) {                                                 \
      printf("cuda err %s at line %d\n", cudaGetErrorString(e_), __LINE__);  \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

int main(int argc, char** argv) {
  const int M = argc > 1 ? atoi(argv[1]) : 32768;
  const int D = 256, BN = 4096;
  const double flops = 2.0 * BN * M * D;
  __half *Qb, *Kb, *S;
  CU(cudaMalloc(&Qb, (size_t)M * D * 2));
  CU(cudaMalloc(&Kb, (size_t)BN * D * 2));
  CU(cudaMalloc(&S, (size_t)M * BN * 2));
  CU(cudaMemset(Qb, 0x35, (size_t)M * D * 2));
  CU(cudaMemset(Kb, 0x35, (size_t)BN * D * 2));
  CU(cudaMemset(S, 0x35, (size_t)M * BN * 2));

  cublasLtHandle_t lt;
  CK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                    sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                    sizeof(tb)));
  // service layout: S_cm[BN,M] = Kb_cm[D,BN]^T @ Qb_cm[D,M]
  cublasLtMatrixLayout_t La, Lb, Lc;
  CK(cublasLtMatrixLayoutCreate(&La, CUDA_R_16F, D, BN, D));
  CK(cublasLtMatrixLayoutCreate(&Lb, CUDA_R_16F, D, M, D));
  CK(cublasLtMatrixLayoutCreate(&Lc, CUDA_R_16F, BN, M, BN));
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = 64ull << 20;
  void* wsp = nullptr;
  CU(cudaMalloc(&wsp, ws));
  CK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  std::vector<cublasLtMatmulHeuristicResult_t> res(32);
  int found = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, op, La, Lb, Lc, Lc, pref, 32,
                                    res.data(), &found));
  const float alpha = 1.f, beta = 0.f;
  const float alpha1 = 1.f, beta0 = 0.f;
  (void)alpha; (void)beta;
  printf("QK^T m=%d n=%d k=%d : %d candidates, interleaved in one process\n",
         BN, M, D, found);
  std::vector<std::vector<float>> best(found);
  const int ROUNDS = 6;
  for (int r = 0; r < ROUNDS; ++r) {
    for (int i = 0; i < found; ++i) {
      // one warm call, then min over 8 timed bursts of 5
      cublasLtMatmul(lt, op, &alpha1, Kb, La, Qb, Lb, &beta0, S, Lc, S, Lc,
                     &res[i].algo, wsp, ws, 0);
      CU(cudaDeviceSynchronize());
      float mn = 1e30f;
      for (int rep = 0; rep < 8; ++rep) {
        cudaEvent_t e0, e1;
        CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
        CU(cudaEventRecord(e0));
        for (int t = 0; t < 5; ++t)
          cublasLtMatmul(lt, op, &alpha1, Kb, La, Qb, Lb, &beta0, S, Lc, S, Lc,
                         &res[i].algo, wsp, ws, 0);
        CU(cudaEventRecord(e1));
        CU(cudaEventSynchronize(e1));
        float ms;
        CU(cudaEventElapsedTime(&ms, e0, e1));
        if (ms / 5 < mn) mn = ms / 5;
        CU(cudaEventDestroy(e0)); CU(cudaEventDestroy(e1));
      }
      if (r == 0) best[i].resize(ROUNDS, 1e30f);
      if (mn < best[i][r]) best[i][r] = mn;
    }
  }
  printf("  cand  algo/tile/stages/splitK        min over rounds        TF    "
         "%%peak   vs default\n");
  float def = 1e30f; { for (int r=0;r<ROUNDS;++r) def = std::min(def, best[0][r]); }
  float global = 1e30f;
  int gbi = -1;
  for (int i = 0; i < found; ++i) {
    float mn = 1e30f;
    for (int r = 0; r < ROUNDS; ++r) mn = std::min(mn, best[i][r]);
    int aid = -1, tile = -1, st = -1, sk = -1;
    size_t wr = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID,
                                         &aid, sizeof(aid), &wr);
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo,
                                         CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                         sizeof(tile), &wr);
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo,
                                         CUBLASLT_ALGO_CONFIG_STAGES_ID, &st,
                                         sizeof(st), &wr);
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo,
                                         CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk,
                                         sizeof(sk), &wr);
    double tf = flops / (mn * 1e-3) / 1e12;
    printf("   [%d]  %4d/%3d/%3d/%3d   %8.1f us   %6.1f TF  %5.1f%%   %+6.1f%%%s\n",
           i, aid, tile, st, sk, mn*1e3, tf, tf / 123.0 * 100.0,
           (mn / def - 1.0) * 100.0, i == 0 ? "  <-- default" : "");
    if (mn < global) { global = mn; gbi = i; }
  }
  printf("  service QK^T is 985.0 us; best candidate %.1f us (%+.1f%%) [%d]\n",
         global*1e3, (global / 0.985 - 1.0) * 100.0, gbi);
  return 0;
}
