
#include <stdlib.h>
#include <omp.h>
struct npu_vision_config {
  int dim;
  int n_heads;
  int head_dim;
  int inter;
  int max_seq;
  int max_layers;
  float eps;
};
static struct npu_vision_config g_cfg = {0};
static int vis_init_done = 0;
static float *g_x, *g_h, *g_qkv, *g_attn, *g_proj, *g_fc, *g_gelu, *g_down;
static uint16_t *g_h16, *g_gelu16;
static float *g_q, *g_k, *g_v;
static float **vis_ln1w, **vis_ln1b, **vis_ln2w, **vis_ln2b;
static float **vis_qkv_b, **vis_proj_b, **vis_fc_b, **vis_down_b;
static double vis_t_mm = 0, vis_t_ln = 0, vis_t_attn = 0, vis_t_gelu = 0, vis_t_res = 0;
extern int npu_cna_cache_setup(int n_w, int M, int max_K, int max_N);
extern int npu_cna_cache_load(int w_idx, const void* Wh, int K, int N);
extern int npu_cna_cache_run_m(int w_idx, int M, const void* Xh, void* Zh32);
extern int npu_cna_ready(void);
extern int npu_cna_wt_n(int w_idx);
extern void npu_cna_close(void);
extern int npu_cna_layernorm(const float* x, const float* scale_w,
                             const float* bias, int M, int H, float eps,
                             float* out);
extern void npu_erf_approx(const float* x, int n, float* out);
static uint16_t f32_to_fp16(float f);
static void softmax_row(float* x, int n);
#define ATTN_K_SLOT 126
#define ATTN_V_SLOT 127
static void alloc_bufs(void) {
  int S = g_cfg.max_seq, D = g_cfg.dim, I = g_cfg.inter;
  int NH = g_cfg.n_heads, HD = g_cfg.head_dim, NL = g_cfg.max_layers;
  g_x = malloc(S*D*4); g_h = malloc(S*D*4);
  g_qkv = malloc(S*D*3*4); g_attn = malloc(S*D*4);
  g_proj = malloc(S*D*4); g_fc = malloc(S*I*4);
  g_gelu = malloc(S*I*4); g_down = malloc(S*D*4);
  g_h16 = malloc(S*D*2); g_gelu16 = malloc(S*I*2);
  g_q = malloc(NH*S*HD*4); g_k = malloc(NH*S*HD*4); g_v = malloc(NH*S*HD*4);
  vis_ln1w = calloc(NL, sizeof(float*)); vis_ln1b = calloc(NL, sizeof(float*));
  vis_ln2w = calloc(NL, sizeof(float*)); vis_ln2b = calloc(NL, sizeof(float*));
  vis_qkv_b = calloc(NL, sizeof(float*)); vis_proj_b = calloc(NL, sizeof(float*));
  vis_fc_b = calloc(NL, sizeof(float*)); vis_down_b = calloc(NL, sizeof(float*));
}
static void free_bufs(void) {
  free(g_x); free(g_h); free(g_qkv); free(g_attn); free(g_proj);
  free(g_fc); free(g_gelu); free(g_down); free(g_h16); free(g_gelu16);
  free(vis_ln1w); free(vis_ln1b); free(vis_ln2w); free(vis_ln2b);
  free(vis_qkv_b); free(vis_proj_b); free(vis_fc_b); free(vis_down_b);
  g_x = g_h = g_qkv = g_attn = g_proj = g_fc = g_gelu = g_down = NULL;
  g_h16 = g_gelu16 = g_q = g_k = g_v = NULL;
}
__attribute__((visibility("default")))
int npu_vision_init(int dim, int n_heads, int inter, int max_seq, int max_layers, float eps) {
  if (vis_init_done) return 0;
  g_cfg.dim = dim;
  g_cfg.n_heads = n_heads;
  g_cfg.head_dim = dim / n_heads;
  g_cfg.inter = inter;
  g_cfg.max_seq = max_seq;
  g_cfg.max_layers = max_layers;
  g_cfg.eps = eps;
  alloc_bufs();
  
  int rc = npu_cna_cache_setup(max_layers * 4 + 2, max_seq, dim, inter);
  if (rc) { free_bufs(); return rc; }
  vis_init_done = 1;
  return 0;
}
__attribute__((visibility("default")))
int npu_vision_load_weights(int layer, const void* w_qkv, const void* w_proj,
                            const void* w_fc, const void* w_down,
                            const float* ln1w, const float* ln1b,
                            const float* ln2w, const float* ln2b,
                            const float* qkv_b, const float* proj_b,
                            const float* fc_b, const float* down_b) {
  if (!vis_init_done) return -1;
  int D = g_cfg.dim, I = g_cfg.inter;
  int base = layer * 4;
  if (npu_cna_cache_load(base + 0, w_qkv, D, D * 3) < 0) return -1;
  if (npu_cna_cache_load(base + 1, w_proj, D, D) < 0) return -1;
  if (npu_cna_cache_load(base + 2, w_fc, D, I) < 0) return -1;
  if (npu_cna_cache_load(base + 3, w_down, I, D) < 0) return -1;
  vis_ln1w[layer] = (float*)ln1w; vis_ln1b[layer] = (float*)ln1b;
  vis_ln2w[layer] = (float*)ln2w; vis_ln2b[layer] = (float*)ln2b;
  vis_qkv_b[layer] = (float*)qkv_b; vis_proj_b[layer] = (float*)proj_b;
  vis_fc_b[layer] = (float*)fc_b; vis_down_b[layer] = (float*)down_b;
  return 0;
}
__attribute__((visibility("default")))
int npu_vision_forward(const float* input, int seq_len, int n_layers, float* output) {
  if (!vis_init_done) return -1;
  int D = g_cfg.dim, I = g_cfg.inter, NH = g_cfg.n_heads, HD = g_cfg.head_dim;
  memcpy(g_x, input, seq_len * D * sizeof(float));
  double t0;
  for (int l = 0; l < n_layers; l++) {
    int wb = l * 4;
    t0 = omp_get_wtime();
    npu_cna_layernorm(g_x, vis_ln1w[l], vis_ln1b[l], seq_len, D, g_cfg.eps, g_h);
    vis_t_ln += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    for (int i = 0; i < seq_len * D; i++) g_h16[i] = f32_to_fp16(g_h[i] * 10.0f);
    npu_cna_cache_run_m(wb + 0, seq_len, g_h16, g_qkv);
    for (int i = 0; i < seq_len * D * 3; i++) g_qkv[i] *= 0.1f;
    if (vis_qkv_b[l])
      for (int s = 0; s < seq_len; s++)
        for (int d = 0; d < D * 3; d++)
          g_qkv[s * D * 3 + d] += vis_qkv_b[l][d];
    vis_t_mm += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    for (int h = 0; h < NH; h++) {
      for (int s = 0; s < seq_len; s++) {
        int off = s * D * 3;
        for (int d = 0; d < HD; d++) {
          g_q[h * seq_len * HD + s * HD + d] = g_qkv[off + h * HD + d];
          g_k[h * seq_len * HD + s * HD + d] = g_qkv[off + D + h * HD + d];
          g_v[h * seq_len * HD + s * HD + d] = g_qkv[off + 2 * D + h * HD + d];
        }
      }
    }
    float scale = 1.0f / sqrtf((float)HD);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; h++) {
      float* qh = g_q + h * seq_len * HD;
      float* kh = g_k + h * seq_len * HD;
      float* vh = g_v + h * seq_len * HD;
      for (int m = 0; m < seq_len; m++) {
        float sc[512];
        float* qm = qh + m * HD;
        for (int s = 0; s < seq_len; s++) {
          float dot = 0;
          float* ks = kh + s * HD;
          for (int d = 0; d < HD; d++) dot += qm[d] * ks[d];
          sc[s] = dot * scale;
        }
        softmax_row(sc, seq_len);
        float* om = g_attn + m * D + h * HD;
        for (int d = 0; d < HD; d++) om[d] = 0;
        for (int s = 0; s < seq_len; s++) {
          float a = sc[s];
          float* vs = vh + s * HD;
          for (int d = 0; d < HD; d++) om[d] += a * vs[d];
        }
      }
    }
    vis_t_attn += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    for (int i = 0; i < seq_len * D; i++) g_h16[i] = f32_to_fp16(g_attn[i] * 10.0f);
    npu_cna_cache_run_m(wb + 1, seq_len, g_h16, g_proj);
    for (int i = 0; i < seq_len * D; i++) g_proj[i] *= 0.1f;
    if (vis_proj_b[l])
      for (int s = 0; s < seq_len; s++)
        for (int d = 0; d < D; d++)
          g_proj[s * D + d] += vis_proj_b[l][d];
    vis_t_mm += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < seq_len * D; i++) g_x[i] += g_proj[i];
    vis_t_res += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    npu_cna_layernorm(g_x, vis_ln2w[l], vis_ln2b[l], seq_len, D, g_cfg.eps, g_h);
    vis_t_ln += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    for (int i = 0; i < seq_len * D; i++) g_h16[i] = f32_to_fp16(g_h[i] * 10.0f);
    npu_cna_cache_run_m(wb + 2, seq_len, g_h16, g_fc);
    for (int i = 0; i < seq_len * I; i++) g_fc[i] *= 0.1f;
    if (vis_fc_b[l])
      for (int s = 0; s < seq_len; s++)
        for (int d = 0; d < I; d++)
          g_fc[s * I + d] += vis_fc_b[l][d];
    vis_t_mm += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < seq_len * I; i++) {
      float x = g_fc[i];
      g_gelu[i] = 0.5f * x * (1.0f + tanhf(0.7978845f * (x + 0.044715f * x * x * x)));
    }
    vis_t_gelu += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    for (int i = 0; i < seq_len * I; i++) g_gelu16[i] = f32_to_fp16(g_gelu[i] * 10.0f);
    npu_cna_cache_run_m(wb + 3, seq_len, g_gelu16, g_down);
    for (int i = 0; i < seq_len * D; i++) g_down[i] *= 0.1f;
    if (vis_down_b[l])
      for (int s = 0; s < seq_len; s++)
        for (int d = 0; d < D; d++)
          g_down[s * D + d] += vis_down_b[l][d];
    vis_t_mm += omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < seq_len * D; i++) g_x[i] += g_down[i];
    vis_t_res += omp_get_wtime() - t0;
  }
  memcpy(output, g_x, seq_len * D * sizeof(float));
  return 0;
}
__attribute__((visibility("default")))
void npu_vision_profile(double* t_mm, double* t_ln, double* t_attn,
                        double* t_gelu, double* t_res, double* t_total) {
  *t_mm = vis_t_mm * 1000.0;
  *t_ln = vis_t_ln * 1000.0;
  *t_attn = vis_t_attn * 1000.0;
  *t_gelu = vis_t_gelu * 1000.0;
  *t_res = vis_t_res * 1000.0;
  *t_total = *t_mm + *t_ln + *t_attn + *t_gelu + *t_res;
}
__attribute__((visibility("default")))
void npu_vision_reset_profile(void) {
  vis_t_mm = vis_t_ln = vis_t_attn = vis_t_gelu = vis_t_res = 0;
}
__attribute__((visibility("default")))
void npu_vision_close(void) {
  free_bufs();
  npu_cna_close();
  vis_init_done = 0;
}