#define _GNU_SOURCE
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <omp.h>

struct lm_config {
  int dim;
  int n_heads;
  int head_dim;
  int inter;
  int vocab;
  int n_layers;
  int max_seq;
  float eps;
  int proj_blocks;
};

static struct lm_config lm_cfg = {0};
static int lm_init_done = 0;

static float *lm_wte, *lm_wpe;
static float **lm_ln1w, **lm_ln1b, **lm_ln2w, **lm_ln2b;
static float *lm_lnfw, *lm_lnfb;
static float **lm_qkvb, **lm_ob, **lm_fcb, **lm_pjb;

static float *lm_kcache;
static float *lm_vcache;
static int lm_pos = 0;

static float *lm_x, *lm_h, *lm_qkv, *lm_q, *lm_av, *lm_fc, *lm_gelu_buf, *lm_proj, *lm_blk, *lm_pjout;
static uint16_t *lm_x16;

static double lm_t_mm=0, lm_t_ln=0, lm_t_attn=0, lm_t_lh=0, lm_t_step=0;
static long lm_n_steps=0;

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static uint16_t f32_to_fp16(float f){ __fp16 h = (__fp16)f; uint16_t r; memcpy(&r,&h,2); return r; }
static float* rdvec(FILE* f, int n){ float* p=(float*)malloc((size_t)n*4); if(p){ if(fread(p,4,n,f)!=(size_t)n){ free(p); p=NULL; } } return p; }

static float lm_gelu(float x){
  float a = 0.7978845f*(x + 0.044715f*x*x*x); float aa = a<0?-a:a; float th;
  if(aa >= 3.5f){ th = a<0?-1.0f:1.0f; }
  else { float a2=a*a; th = a*(135135.0f+a2*(17325.0f+a2*(378.0f+a2))) / (135135.0f+a2*(62370.0f+a2*(3150.0f+a2*28.0f))); }
  return 0.5f*x*(1.0f+th);
}

extern int npu_cna_cache_setup(int n_w, int M, int max_K, int max_N);
extern int npu_cna_cache_load(int w_idx, const void* Wh, int K, int N);
extern int npu_cna_cache_run_m(int w_idx, int M, const void* Xh, void* Zh32);
extern int npu_cna_ready(void);
extern void npu_cna_close(void);

static void lm_alloc(void){
  int D=lm_cfg.dim, I=lm_cfg.inter, V=lm_cfg.vocab;
  int NL=lm_cfg.n_layers, S=lm_cfg.max_seq;
  lm_kcache = malloc((size_t)NL*S*D*4);
  lm_vcache = malloc((size_t)NL*S*D*4);
  lm_x = malloc(D*4); lm_h = malloc(D*4);
  lm_qkv = malloc(D*3*4); lm_q = malloc(D*4);
  lm_av = malloc(D*4); lm_fc = malloc(I*4);
  lm_gelu_buf = malloc(I*4); lm_proj = malloc(D*4);
  lm_blk = malloc(D*4); lm_pjout = malloc(D*4);
  lm_x16 = malloc(D*2);
  lm_ln1w = calloc(NL,sizeof(float*)); lm_ln1b = calloc(NL,sizeof(float*));
  lm_ln2w = calloc(NL,sizeof(float*)); lm_ln2b = calloc(NL,sizeof(float*));
  lm_qkvb = calloc(NL,sizeof(float*)); lm_ob = calloc(NL,sizeof(float*));
  lm_fcb = calloc(NL,sizeof(float*)); lm_pjb = calloc(NL,sizeof(float*));
}

static void lm_free(void){
  free(lm_kcache); free(lm_vcache);
  free(lm_x); free(lm_h); free(lm_qkv); free(lm_q); free(lm_av);
  free(lm_fc); free(lm_gelu_buf); free(lm_proj); free(lm_blk); free(lm_pjout);
  free(lm_x16);
  free(lm_ln1w); free(lm_ln1b); free(lm_ln2w); free(lm_ln2b);
  free(lm_qkvb); free(lm_ob); free(lm_fcb); free(lm_pjb);
  lm_kcache = lm_vcache = lm_x = lm_h = lm_qkv = lm_q = lm_av = NULL;
  lm_fc = lm_gelu_buf = lm_proj = lm_blk = lm_pjout = NULL;
  lm_x16 = NULL;
}

__attribute__((visibility("default")))
int npu_lm_init(int dim, int n_heads, int inter, int vocab, int n_layers, int max_seq, float eps){
  if(lm_init_done) return 0;
  lm_cfg.dim = dim; lm_cfg.n_heads = n_heads;
  lm_cfg.head_dim = dim / n_heads;
  lm_cfg.inter = inter; lm_cfg.vocab = vocab;
  lm_cfg.n_layers = n_layers; lm_cfg.max_seq = max_seq;
  lm_cfg.eps = eps;
  lm_cfg.proj_blocks = (inter % dim == 0) ? inter / dim : 0;
  lm_alloc();
  int n_w = n_layers * (3 + lm_cfg.proj_blocks) + 2;
  if(!npu_cna_ready()){
    int rc = npu_cna_cache_setup(n_w, 64, dim > inter ? dim : inter, dim > inter ? dim : inter);
    if(rc){ lm_free(); return rc; }
  }
  lm_init_done = 1;
  return 0;
}

__attribute__((visibility("default")))
int npu_lm_load_params(const char* path){
  FILE* f=fopen(path,"rb"); if(!f) return -1;
  int D=lm_cfg.dim, I=lm_cfg.inter, V=lm_cfg.vocab, NL=lm_cfg.n_layers, S=lm_cfg.max_seq;
  lm_wte = rdvec(f, V*D); lm_wpe = rdvec(f, S*D);
  for(int i=0;i<NL;i++){
    lm_ln1w[i]=rdvec(f,D); lm_ln1b[i]=rdvec(f,D);
    lm_ln2w[i]=rdvec(f,D); lm_ln2b[i]=rdvec(f,D);
    lm_qkvb[i]=rdvec(f,D*3); lm_ob[i]=rdvec(f,D);
    lm_fcb[i]=rdvec(f,I); lm_pjb[i]=rdvec(f,D);
  }
  lm_lnfw=rdvec(f,D); lm_lnfb=rdvec(f,D); fclose(f);
  int ok = lm_wte&&lm_wpe&&lm_lnfw&&lm_lnfb;
  for(int i=0;i<NL&&ok;i++) ok = ok && lm_ln1w[i]&&lm_ln1b[i]&&lm_ln2w[i]&&lm_ln2b[i]&&lm_qkvb[i]&&lm_ob[i]&&lm_fcb[i]&&lm_pjb[i];
  return ok?0:-2;
}

__attribute__((visibility("default")))
void npu_lm_reset(void){ lm_pos = 0; }

static void lm_layernorm(const float* x, const float* w, const float* b, float* out){
  int D=lm_cfg.dim; float mean=0;
  for(int j=0;j<D;j++) mean+=x[j]; mean/=D;
  float var=0; for(int j=0;j<D;j++){float d=x[j]-mean; var+=d*d;} var/=D;
  float inv=1.0f/sqrtf(var+lm_cfg.eps);
  for(int j=0;j<D;j++) out[j]=(x[j]-mean)*inv*w[j]+b[j];
}

static int lm_mm1(const float* x, int w_idx, int K, float* z){
  for(int i=0;i<K;i++) lm_x16[i]=f32_to_fp16(x[i]);
  return npu_cna_cache_run_m(w_idx, 1, lm_x16, z);
}

static void lm_attention(const float* q, const float* kc, const float* vc, int n_kv, float* av){
  int NH=lm_cfg.n_heads, HD=lm_cfg.head_dim, D=lm_cfg.dim, S=lm_cfg.max_seq;
  float scale=1.0f/sqrtf((float)HD);
  #pragma omp parallel for schedule(static)
  for(int nh=0;nh<NH;nh++){
    float row[4096]; int hoff=nh*HD;
    float mx=-1e30f;
    for(int s=0;s<n_kv;s++){
      float dot=0; const float* qp=q+hoff; const float* kp=kc+(size_t)s*D+hoff;
      for(int d=0;d<HD;d++) dot+=qp[d]*kp[d];
      float sc=dot*scale; row[s]=sc; if(sc>mx) mx=sc;
    }
    float sum=0; for(int s=0;s<n_kv;s++){ row[s]=expf(row[s]-mx); sum+=row[s]; }
    float inv=1.0f/sum;
    for(int d=0;d<HD;d++){ float a=0; for(int s=0;s<n_kv;s++) a+=row[s]*inv*vc[(size_t)s*D+hoff+d]; av[hoff+d]=a; }
  }
}

static int lm_idx(int i, int role){ return i*(3+lm_cfg.proj_blocks)+role; }

__attribute__((visibility("default")))
int npu_lm_step(int token_id, float* logits){
  if(!lm_wte) return -1;
  if(!lm_init_done) return -1;
  int D=lm_cfg.dim, I=lm_cfg.inter, V=lm_cfg.vocab, NL=lm_cfg.n_layers;
  int PB=lm_cfg.proj_blocks;
  if(!lm_t_step){ omp_set_num_threads(3); }
  double t0=now_s();
  int pos = lm_pos;
  if(pos >= lm_cfg.max_seq) return -3;

  for(int j=0;j<D;j++) lm_x[j]=lm_wte[(size_t)token_id*D+j]+lm_wpe[pos*D+j];

  for(int i=0;i<NL;i++){
    { double a=now_s(); lm_layernorm(lm_x, lm_ln1w[i], lm_ln1b[i], lm_h); lm_t_ln+=now_s()-a; }
    { double a=now_s();
      if(lm_mm1(lm_h, lm_idx(i,0), D, lm_qkv)) return -2;
      float* k = lm_kcache + (size_t)i*lm_cfg.max_seq*D + pos*D;
      float* v = lm_vcache + (size_t)i*lm_cfg.max_seq*D + pos*D;
      for(int j=0;j<D;j++){
        lm_q[j]=lm_qkv[j]+lm_qkvb[i][j];
        k[j]=lm_qkv[D+j]+lm_qkvb[i][D+j];
        v[j]=lm_qkv[2*D+j]+lm_qkvb[i][2*D+j];
      }
      lm_attention(lm_q, lm_kcache+(size_t)i*lm_cfg.max_seq*D, lm_vcache+(size_t)i*lm_cfg.max_seq*D, pos+1, lm_av);
      if(lm_mm1(lm_av, lm_idx(i,1), D, lm_h)) return -2;
      for(int j=0;j<D;j++) lm_x[j] += lm_h[j] + lm_ob[i][j];
      lm_t_mm+=now_s()-a;
    }
    { double a=now_s(); lm_layernorm(lm_x, lm_ln2w[i], lm_ln2b[i], lm_h); lm_t_ln+=now_s()-a; }
    { double a=now_s();
      if(lm_mm1(lm_h, lm_idx(i,2), D, lm_fc)) return -2;
      for(int j=0;j<I;j++) lm_gelu_buf[j]=lm_gelu(lm_fc[j]+lm_fcb[i][j]);
      memset(lm_proj,0,D*4);
      if(PB > 0){
        for(int b=0;b<PB;b++){
          for(int j=0;j<D;j++) lm_blk[j]=lm_gelu_buf[b*D+j];
          if(lm_mm1(lm_blk, lm_idx(i,3+b), D, lm_pjout)) return -2;
          for(int j=0;j<D;j++) lm_proj[j]+=lm_pjout[j];
        }
      } else {
        if(lm_mm1(lm_gelu_buf, lm_idx(i,3), I, lm_proj)) return -2;
      }
      for(int j=0;j<D;j++) lm_x[j] += lm_proj[j] + lm_pjb[i][j];
      lm_t_mm+=now_s()-a;
    }
  }
  lm_pos++;
  { double a=now_s(); lm_layernorm(lm_x, lm_lnfw, lm_lnfb, lm_h); lm_t_ln+=now_s()-a; }
  { double a=now_s();
    #pragma omp parallel for schedule(static)
    for(int v=0;v<V;v++){ const float* wr=lm_wte+(size_t)v*D; float s=0; for(int j=0;j<D;j++) s+=lm_h[j]*wr[j]; logits[v]=s; }
    lm_t_lh+=now_s()-a;
  }
  lm_t_step+=now_s()-t0; lm_n_steps++;
  return 0;
}

__attribute__((visibility("default")))
int npu_lm_prefill(const int* ids, int n_ids, float* logits){
  for(int t=0;t<n_ids;t++){
    int rc = npu_lm_step(ids[t], logits);
    if(rc) return rc;
  }
  return 0;
}

__attribute__((visibility("default")))
void npu_lm_profile(double* out){
  if(lm_n_steps>0){
    out[0]=lm_t_mm*1000/lm_n_steps; out[1]=lm_t_ln*1000/lm_n_steps;
    out[2]=lm_t_attn*1000/lm_n_steps; out[3]=0;
    out[4]=lm_t_lh*1000/lm_n_steps; out[5]=lm_t_step*1000/lm_n_steps;
  } else { for(int i=0;i<6;i++) out[i]=0; }
}

__attribute__((visibility("default")))
void npu_lm_close(void){
  lm_free();
  npu_cna_close();
  lm_init_done = 0;
}
