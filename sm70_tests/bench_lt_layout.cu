// Probe: can ANY cublasLt algorithm beat cuBLAS's default on the two
// 250K-prefill attention GEMMs?  This is the gate before writing any kernel.
//
// Shapes from sm70_nvfp4_prefill_twopass.py run(): per (request, kv head, q
// block) and per BN chunk it does  S[M,BN] = Qb[M,D] @ Kb^T,  PV[M,D] = S @ Vb.
// The service trace times them at 985.0 us and 935.9 us = 69.8 / 73.4 TFLOPS on
// a 123-TFLOPS peak (57% / 60%), both as cuBLAS 128x128 tiles, 128 threads,
// est. occupancy 13%.
//
// Column-major mapping (it reproduces the observed kernel names, which is the
// check that it is right):
//   QK^T  S_cm[BN,M] = Kb_cm[D,BN]^T @ Qb_cm[D,M]   m=BN n=M k=D  A=T lda=D,
//                                                            B=N ldb=D   -> "tn"
//   PV    PV_cm[D,M] = Vb_cm[D,BN]   @ S_cm[BN,M]   m=D  n=M k=BN A=N lda=D,
//                                                            B=N ldb=BN  -> "nn"
// If nothing in the heuristic list beats ~65%, the GEMM path is closed and only
// a hand-written kernel could beat cuBLAS; if something does, swapping these two
// at::mm_out calls for an explicit Lt call is a cheap lossless prefill win.
//
// build: nvcc -O3 -arch=sm_70 ltprobe.cu -lcublasLt -o ltprobe
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                        \
  do {                                                               \
    cublasStatus_t s_ = (x);                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                               \
      printf("cublas err %d at line %d\n", (int)s_, __LINE__);       \
      exit(1);                                                       \
    }                                                                \
  } while (0)
#define CU(x)                                                              \
  do {                                                                     \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
      printf("cuda err %s at line %d\n", cudaGetErrorString(e_), __LINE__);\
      exit(1);                                                             \
    }                                                                      \
  } while (0)

static void probe(const char* tag, int m, int n, int k, cublasOperation_t ta,
                  cublasOperation_t tb, int lda, int ldb, int ldc,
                  const void* A, const void* B, void* C) {
  cublasLtHandle_t lt;
  CK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                    sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                    sizeof(tb)));
  cublasLtMatrixLayout_t La, Lb, Lc, Ld;
  CK(cublasLtMatrixLayoutCreate(&La, CUDA_R_16F, ta == CUBLAS_OP_T ? k : m,
                                ta == CUBLAS_OP_T ? m : k, lda));
  CK(cublasLtMatrixLayoutCreate(&Lb, CUDA_R_16F, tb == CUBLAS_OP_T ? n : k,
                                tb == CUBLAS_OP_T ? k : n, ldb));
  CK(cublasLtMatrixLayoutCreate(&Lc, CUDA_R_16F, m, n, ldc));
  CK(cublasLtMatrixLayoutCreate(&Ld, CUDA_R_16F, m, n, ldc));

  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = 64ull << 20;
  void* wsp = nullptr;
  CU(cudaMalloc(&wsp, ws));
  CK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

  const int REQ = 32;
  std::vector<cublasLtMatmulHeuristicResult_t> res(REQ);
  int found = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, op, La, Lb, Lc, Ld, pref, REQ,
                                    res.data(), &found));
  printf("\n%s: m=%d n=%d k=%d A=%c B=%c ldc=%d -> %d candidates\n", tag, m, n,
         k, ta == CUBLAS_OP_T ? 'T' : 'N', tb == CUBLAS_OP_T ? 'T' : 'N', ldc,
         found);
  if (!found) { printf("  none\n"); return; }

  const float alpha = 1.f, beta = 0.f;
  const double flops = 2.0 * m * n * k;
  float best_ms = 1e30f;
  int best_i = -1;
  for (int i = 0; i < found; ++i) {
    cublasStatus_t st = cublasLtMatmul(lt, op, &alpha, A, La, B, Lb, &beta, C,
                                       Lc, C, Lc, &res[i].algo, wsp, ws, 0);
    if (st != CUBLAS_STATUS_SUCCESS) continue;
    CU(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
    float mn = 1e30f;
    for (int r = 0; r < 5; ++r) {
      CU(cudaEventRecord(e0));
      for (int t = 0; t < 5; ++t)
        cublasLtMatmul(lt, op, &alpha, A, La, B, Lb, &beta, C, Lc, C, Lc,
                       &res[i].algo, wsp, ws, 0);
      CU(cudaEventRecord(e1));
      CU(cudaEventSynchronize(e1));
      float ms;
      CU(cudaEventElapsedTime(&ms, e0, e1));
      if (ms / 5 < mn) mn = ms / 5;
    }
    CU(cudaEventDestroy(e0)); CU(cudaEventDestroy(e1));
    double tf = flops / (mn * 1e-3) / 1e12;
    int aid = -1, tile = -1, stages = -1, splitk = -1;
    size_t wr = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID,
                                         &aid, sizeof(aid), &wr);
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo,
                                         CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                         sizeof(tile), &wr);
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo,
                                         CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &stages, sizeof(stages), &wr);
    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo,
                                         CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &splitk, sizeof(splitk), &wr);
    if (i < 6 || mn < best_ms)
      printf("   [%2d] %8.1f us  %6.1f TF  %5.1f%% peak  algo=%d tile=%d "
             "stages=%d splitK=%d%s\n",
             i, mn * 1e3, tf, tf / 123.0 * 100.0, aid, tile, stages, splitk,
             i == 0 ? "   <-- heuristic default" : "");
    if (mn < best_ms) { best_ms = mn; best_i = i; }
  }
  double btf = flops / (best_ms * 1e-3) / 1e12;
  printf("  BEST=[%d] %.1f us  %.1f TF  %.1f%% peak\n", best_i, best_ms * 1e3,
         btf, btf / 123.0 * 100.0);
  CU(cudaFree(wsp));
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(La);
  cublasLtMatrixLayoutDestroy(Lb);
  cublasLtMatrixLayoutDestroy(Lc);
  cublasLtMatrixLayoutDestroy(Ld);
  cublasLtMatmulDescDestroy(op);
  cublasLtDestroy(lt);
}

// Same GEMM, four ways of expressing it.  Which operand ends up as the wide
// one decides how many column tiles cuBLAS sees, and PV's logical N=256 is only
// two 128-wide tiles, which is the likeliest reason it sits at 60%.  A layout
// win here is free to wire (it is only a different view of the same buffers).
int main() {
  const int M = 32768, D = 256, BN = 4096;
  __half *Qb, *Kb, *Vb, *S, *PV, *ST, *PVT;
  CU(cudaMalloc(&Qb, (size_t)M * D * 2));
  CU(cudaMalloc(&Kb, (size_t)BN * D * 2));
  CU(cudaMalloc(&Vb, (size_t)BN * D * 2));
  CU(cudaMalloc(&S, (size_t)M * BN * 2));
  CU(cudaMalloc(&PV, (size_t)M * D * 2));
  CU(cudaMalloc(&ST, (size_t)M * BN * 2));
  CU(cudaMalloc(&PVT, (size_t)M * D * 2));
  CU(cudaMemset(Qb, 0x35, (size_t)M * D * 2));
  CU(cudaMemset(Kb, 0x35, (size_t)BN * D * 2));
  CU(cudaMemset(Vb, 0x35, (size_t)BN * D * 2));
  CU(cudaMemset(S, 0x35, (size_t)M * BN * 2));
  CU(cudaMemset(ST, 0x35, (size_t)M * BN * 2));
  CU(cudaMemset(PV, 0x35, (size_t)M * D * 2));
  CU(cudaMemset(PVT, 0x35, (size_t)M * D * 2));
  // QK^T as the service runs it: C_cm[BN,M] = Kb_cm[D,BN]^T @ Qb_cm[D,M]  ("tn")
  probe("QK^T  as-is   ", BN, M, D, CUBLAS_OP_T, CUBLAS_OP_N, D, D, BN, Kb, Qb,
        S);
  // QK^T with the score matrix stored transposed (softmax would address S^T):
  // C_cm[M,BN] = Qb_cm[D,M]^T @ Kb_cm[D,BN]
  probe("QK^T  ->S^T   ", M, BN, D, CUBLAS_OP_T, CUBLAS_OP_N, D, D, M, Qb, Kb,
        ST);
  // PV as the service runs it: C_cm[D,M] = Vb_cm[D,BN] @ S_cm[BN,M]          ("nn")
  probe("PV    as-is   ", D, M, BN, CUBLAS_OP_N, CUBLAS_OP_N, D, BN, D, Vb, S,
        PV);
  // PV as a TN kernel with the output transposed: C_cm[M,D] = S_cm[BN,M]^T @
  // Vb_cm[D,BN]^T.  PV's as-is form has only m=256 (two 128-wide tiles), which
  // is the likeliest reason it sits at 60%; TN is also the friendliest HMMA
  // layout on Volta.
  probe("PV    ->PV^T  ", M, D, BN, CUBLAS_OP_T, CUBLAS_OP_T, BN, D, M, S, Vb,
        PVT);
  return 0;
}
