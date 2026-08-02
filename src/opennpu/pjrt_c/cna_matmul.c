#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#define CNA_DEV_DEFAULT "/dev/dri/card1"
static const char *cna_dev_path = NULL;
#define CNA_CREATE 0xC0306442
#define CNA_MAP 0xC0106443
#define CNA_SYNC 0xC0206445
#define CNA_SUBMIT 0xC0686441
#define CNA_BATCH_SUBMIT 0xC0106446
#define CNA_ACTION 0xC0086440

#define CNA_FLAGS_BUF 0x403
#define CNA_FLAGS_TASK 0x40b
#define CNA_FLAGS_WT  0x403

static void user_dcache_flush(void* addr, uint64_t size) {
  uintptr_t start = (uintptr_t)addr & ~63ULL;
  uintptr_t end = (uintptr_t)addr + size;
  for (uintptr_t p = start; p < end; p += 64)
    asm volatile("dc civac, %0" :: "r"(p));
  asm volatile("dsb sy" ::: "memory");
}

__attribute__((optimize("no-tree-vectorize")))
static void nc_zero16(void* p, uint64_t bytes) {
  volatile uint16_t* w = (volatile uint16_t*)p;
  uint64_t n = bytes / 2;
  for (uint64_t i = 0; i < n; i++) w[i] = 0;
}
#define MAX_NW 144

#define NPUOP(op,val,reg) (((uint64_t)((op)&0xffff))<<48)|(((uint64_t)((val)&0xffffffff))<<16)|(uint64_t)((reg)&0xffff)

static void cna_sync(int fd, uint64_t obj, uint64_t sz, uint32_t dir) {
  asm volatile("dsb sy" ::: "memory");
  uint8_t sb[32]; memset(sb,0,32); *(uint32_t*)(sb+0)=dir; *(uint64_t*)(sb+8)=obj; *(uint64_t*)(sb+24)=sz;
  ioctl(fd, CNA_SYNC, sb);
  asm volatile("dsb sy" ::: "memory");
}
static void cna_action(int fd, uint32_t flags, uint32_t value) {
  uint32_t act[2]={flags,value}; ioctl(fd, CNA_ACTION, act);
}
static int cna_create_bo(int fd, uint64_t size, uint32_t flags, uint32_t *handle, uint64_t *obj, uint64_t *dma, void **mm) {
  uint8_t buf[48]; memset(buf,0,48); *(uint32_t*)(buf+4)=flags; *(uint64_t*)(buf+8)=size; *(uint32_t*)(buf+44)=1;
  if (ioctl(fd, CNA_CREATE, buf)<0) return -1;
  *handle=*(uint32_t*)(buf+0); *obj=*(uint64_t*)(buf+16); *dma=*(uint64_t*)(buf+24);
  uint8_t mb[16]; memset(mb,0,16); *(uint32_t*)(mb+0)=*handle; ioctl(fd, CNA_MAP, mb);
  *mm=mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, *(uint64_t*)(mb+8));
  if(*mm==MAP_FAILED) return -1;
  cna_sync(fd, *obj, size, 3);
  return 0;
}

static int weight_fp16_off(int C, int k, int c) {
  int kpg=((k-1)/16), cpg=((c-1)/32);
  return ((cpg*32)*16)+(kpg*16*C)+((c-1)%32)+(((k-1)%16)*32);
}
static int feature_data_off(int C, int H, int W, int C2, int c, int h, int w) {
  int plane=(c-1)/C2; int src=plane*H*W*C2; int off=(c-1)%C2;
  return src+C2*((h-1)*W+(w-1))+off;
}

static void cna_gen_task(uint64_t *ops, int M, int K, int N, uint32_t in_dma, uint32_t wt_dma, uint32_t out_dma) {
  uint32_t v;
  ops[0]=NPUOP(0x1001,0xE,0x4004);
  v=((2&7)<<7)|((2&7)<<4)|0; ops[1]=NPUOP(0x0201,v,0x100C);
  v=(0<<16)|((M+1)&0x3FF)<<4; ops[2]=NPUOP(0x0201,v,0x1010);
  v=(1<<3)|1; ops[3]=NPUOP(0x0201,v,0x1014);
  v=((1&0x7FF)<<16)|(M&0x7FF); ops[4]=NPUOP(0x0201,v,0x1020);
  v=(((K-1)&0xFFFF)<<16)|(K&0xFFFF); ops[5]=NPUOP(0x0201,v,0x1024);
  ops[6]=NPUOP(0x0201,1&0x7FF,0x1028);
  ops[7]=NPUOP(0x0201,(1*M)&0x3FFFF,0x102C);
  ops[8]=NPUOP(0x0201,K*2*N,0x1030);
  ops[9]=NPUOP(0x0201,(K*2)&0x7FFFF,0x1034);
  v=((1&0x1F)<<24)|((1&0x1F)<<16)|(N&0x3FFF); ops[10]=NPUOP(0x0201,v,0x1038);
  uint32_t fb=(M*K*2+32767)/32768; if(fb<1)fb=1; uint32_t wb=12-fb;
  ops[11]=NPUOP(0x0201,((wb&0xF)<<4)|(fb&0xF),0x1040);
  uint32_t de=(1*K)/32; if(K%32)de++; ops[12]=NPUOP(0x0201,de&0x1FFF,0x1044);
  ops[13]=NPUOP(0x0201,0xB,0x104C);
  ops[14]=NPUOP(0x0201,0x10000,0x1050); ops[15]=NPUOP(0x0201,0x10000,0x1054);
  ops[16]=NPUOP(0x0201,0x10000,0x1058); ops[17]=NPUOP(0x0201,0x10000,0x105C);
  ops[18]=NPUOP(0x0201,0,0x1060); ops[19]=NPUOP(0x0201,0,0x1064); ops[20]=NPUOP(0x0201,0,0x1068);
  ops[21]=NPUOP(0x0201,in_dma,0x1070); ops[22]=NPUOP(0x0201,0,0x1074);
  ops[23]=NPUOP(0x0201,(0xF<<16)|0xF,0x1078); ops[24]=NPUOP(0x0201,4,0x107C);
  int ss= M>=4 ? 4*((M/4)-1) : 0; ops[25]=NPUOP(0x0201,ss,0x1080);
  ops[26]=NPUOP(0x0201,(1<<16)|M,0x1084); ops[27]=NPUOP(0x0201,K,0x1088);
  ops[28]=NPUOP(0x0201,0,0x1100); ops[29]=NPUOP(0x0201,0,0x1104);
  ops[30]=NPUOP(0x0201,wt_dma,0x1110);
  int i;
  for(i=31;i<=46;i++) ops[i]=NPUOP(0x0201,0,0x1140+(i-31)*4);
  ops[47]=NPUOP(0x0201,0,0x1180); ops[48]=NPUOP(0x0201,0,0x1184);
  v=((2<<8)|1); ops[49]=NPUOP(0x0801,v,0x3010);
  v=(((M-1)<<16)|0); ops[50]=NPUOP(0x0801,v,0x3014);
  ops[51]=NPUOP(0x0801,N-1,0x3018); ops[52]=NPUOP(0x0801,0,0x301C); ops[53]=NPUOP(0x0801,0,0x3030);
  ops[54]=NPUOP(0x1001,(0xF<<5)|(2<<1)|0,0x400C);
  ops[55]=NPUOP(0x1001,((5<<29)|(2<<26)|2),0x4010); ops[56]=NPUOP(0x1001,0,0x4014);
  ops[57]=NPUOP(0x1001,out_dma,0x4020); ops[58]=NPUOP(0x1001,(M<<4),0x4024);
  ops[59]=NPUOP(0x1001,0,0x4030); ops[60]=NPUOP(0x1001,M-1,0x4034); ops[61]=NPUOP(0x1001,0,0x4038);
  ops[62]=NPUOP(0x1001,((N-1)<<16)|(N-1),0x403C);
  ops[63]=NPUOP(0x1001,0x53,0x4040); ops[64]=NPUOP(0x1001,0,0x4044);
  ops[65]=NPUOP(0x1001,0,0x4048); ops[66]=NPUOP(0x1001,0,0x404C);
  ops[67]=NPUOP(0x1001,((3<<8)|(3<<5)|(3<<2)|2),0x4050); ops[68]=NPUOP(0x1001,0,0x4054);
  ops[69]=NPUOP(0x1001,N-1,0x4058); ops[70]=NPUOP(0x1001,((M-1)<<16)|0,0x405C);
  ops[71]=NPUOP(0x1001,0x53,0x4060); ops[72]=NPUOP(0x1001,0,0x4064);
  ops[73]=NPUOP(0x1001,0,0x4068); ops[74]=NPUOP(0x1001,0,0x406C);
  ops[75]=NPUOP(0x1001,0x383,0x4070); ops[76]=NPUOP(0x1001,0,0x4074);
  ops[77]=NPUOP(0x1001,1,0x4078); ops[78]=NPUOP(0x1001,0,0x407C); ops[79]=NPUOP(0x1001,0,0x4080);
  ops[80]=NPUOP(0x1001,1,0x4084); ops[81]=NPUOP(0x1001,0,0x4088);
  for(i=82;i<=89;i++) ops[i]=NPUOP(0x1001,0,0x4090+(i-82)*4);
  ops[90]=NPUOP(0x1001,(M*4<<4),0x40C0); ops[91]=NPUOP(0x1001,0,0x40C4);
  for(i=92;i<=103;i++) ops[i]=NPUOP(0x1001,0,0x4100+(i-92)*4);
  ops[104]=NPUOP(0,0,0); ops[105]=NPUOP(0x0101,0,0x0014); ops[106]=NPUOP(0x0041,0,0);
  ops[107]=NPUOP(0x0081,0x0d,0x0008);
}

typedef struct {
  int fd;
  uint32_t rc_h, tk_h, in_h, wt_h[MAX_NW], out_h;
  uint64_t rc_obj, tk_obj, in_obj, wt_obj[MAX_NW], out_obj;
  uint64_t rc_dma, tk_dma, in_dma, wt_dma[MAX_NW], out_dma;
  void *rc_mm, *tk_mm, *in_mm, *wt_mm[MAX_NW], *out_mm;
  int M, max_K, max_N, n_w;
  uint32_t core_mask;
int wt_K[MAX_NW], wt_N[MAX_NW], wt_Np[MAX_NW];
  int wt_Kt[MAX_NW], wt_ntiles[MAX_NW], wt_tile_bytes[MAX_NW];
} CNAContext;

static CNAContext g_cna = {.fd=-1};

static int cna_init(int M, int max_K, int max_N, int n_w) {
  if (g_cna.fd >= 0) { close(g_cna.fd); g_cna.fd=-1; }
  CNAContext *c = &g_cna;
  memset(c, 0, sizeof(*c));
  c->fd = open(cna_dev_path ? cna_dev_path : getenv("NPU_DEV") ? getenv("NPU_DEV") : CNA_DEV_DEFAULT, O_RDWR);
  if (c->fd < 0) return -1;
  cna_action(c->fd, 6, 0); cna_action(c->fd, 1, 0); cna_action(c->fd, 19, 0xFFFFFFEDu); cna_action(c->fd, 1, 0); cna_action(c->fd, 18, 0);
  c->M=M; c->max_K=max_K; c->max_N=max_N; c->n_w=n_w;
  int M4 = ((M + 3) / 4) * 4; if (M4 < 4) M4 = 4;
  int in_Kt = max_K > 768 ? 768 : ((max_K+31)/32)*32;
  if (cna_create_bo(c->fd, 1024, CNA_FLAGS_BUF, &c->rc_h, &c->rc_obj, &c->rc_dma, &c->rc_mm)<0) return -2;
  if (cna_create_bo(c->fd, 4096, CNA_FLAGS_TASK, &c->tk_h, &c->tk_obj, &c->tk_dma, &c->tk_mm)<0) return -3;
  if (cna_create_bo(c->fd, M4*in_Kt*2+4096, CNA_FLAGS_BUF, &c->in_h, &c->in_obj, &c->in_dma, &c->in_mm)<0) return -4;
  if (cna_create_bo(c->fd, M4*max_N*4+4096, CNA_FLAGS_BUF, &c->out_h, &c->out_obj, &c->out_dma, &c->out_mm)<0) return -6;
  return 0;
}

static void cna_submit_wt(CNAContext *c, int M, int K, int N, uint64_t wt_dma) {
  uint64_t ops[112]; memset(ops,0,sizeof(ops));
  cna_gen_task(ops, M, K, N, (uint32_t)c->in_dma, (uint32_t)wt_dma, (uint32_t)c->out_dma);
  memcpy(c->rc_mm, ops, sizeof(ops)); user_dcache_flush(c->rc_mm, 1024);
  uint8_t *tk=c->tk_mm; memset(tk,0,40);
  *(uint32_t*)(tk+8)=0xd; *(uint32_t*)(tk+12)=0x300; *(uint32_t*)(tk+16)=0x1ffff;
  *(uint32_t*)(tk+24)=112-(4+4); *(uint64_t*)(tk+32)=c->rc_dma;
  user_dcache_flush(c->tk_mm, 4096);
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=1|4; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+12)=1;
  *(uint64_t*)(sub+24)=c->tk_obj; *(uint32_t*)(sub+56)=c->core_mask; *(int32_t*)(sub+60)=-1;
  int ci = (c->core_mask==2) ? 1 : (c->core_mask==4) ? 2 : 0;
  *(uint32_t*)(sub+64+ci*8)=0; *(uint32_t*)(sub+68+ci*8)=1;
  ioctl(c->fd, CNA_SUBMIT, sub);
  user_dcache_flush(c->out_mm, (uint64_t)M*N*4+4096);
}

__attribute__((visibility("default")))
int npu_cna_cache_setup(int n_w, int M, int max_K, int max_N) {
  return cna_init(M, max_K, max_N, n_w);
}

__attribute__((visibility("default")))
int npu_cna_cache_load(int w_idx, const void* Wh, int K, int N) {
  if (g_cna.fd < 0 || w_idx < 0 || w_idx >= g_cna.n_w) return -1;
 CNAContext *c = &g_cna;
  int Kt = K > 768 ? 768 : ((K+31)/32)*32;
  int Kta = ((Kt+31)/32)*32;
  int n_tiles = (K + Kt - 1) / Kt;
  int Np = N < 16 ? 16 : ((N + 15) / 16) * 16;
  int tile_elems = Np * Kta;
  int tile_bytes = tile_elems * 2;
  int total_bytes = n_tiles * tile_bytes + 4096;

  if (cna_create_bo(c->fd, total_bytes, CNA_FLAGS_WT, &c->wt_h[w_idx], &c->wt_obj[w_idx], &c->wt_dma[w_idx], &c->wt_mm[w_idx])<0) return -5;

  c->wt_K[w_idx] = K; c->wt_N[w_idx] = N;
  c->wt_Np[w_idx] = Np;
  c->wt_Kt[w_idx] = Kt; c->wt_ntiles[w_idx] = n_tiles;
  c->wt_tile_bytes[w_idx] = tile_bytes;

  const uint16_t *w=(const uint16_t*)Wh;
  uint16_t *wm=(uint16_t*)c->wt_mm[w_idx];
  memset(wm, 0, total_bytes);

  for (int t=0; t<n_tiles; t++) {
    int ks = t*Kt, ke = ks+Kt; if(ke>K) ke=K; int Kt2=ke-ks;
    int Kta2 = ((Kt2+31)/32)*32;
    uint16_t *tile = wm + t * tile_elems;
    for(int n=1;n<=Np;n++) for(int k=1;k<=Kt2;k++){
      int sk = ks+k-1;
      if (sk < K && n <= N) tile[weight_fp16_off(Kta2,n,k)] = w[sk*N+(n-1)];
    }
  }
  user_dcache_flush(c->wt_mm[w_idx], total_bytes);
  return 0;
}

__attribute__((visibility("default")))
int npu_cna_cache_run_m(int w_idx, int M, const void* Xh, void* Zh32) {
  if (g_cna.fd < 0 || w_idx < 0 || w_idx >= g_cna.n_w) return -1;
  CNAContext *c = &g_cna;
  int K=c->wt_K[w_idx], N=c->wt_N[w_idx], Np=c->wt_Np[w_idx], Kt=c->wt_Kt[w_idx];
  int n_tiles = c->wt_ntiles[w_idx];
  int tile_bytes = c->wt_tile_bytes[w_idx];
  float *out32 = (float*)Zh32;
  memset(out32, 0, M*N*4);
  int M4 = M; if (M4 < 4) M4 = 4; if (M4 % 4) M4 = ((M4 + 3) / 4) * 4;
  const uint16_t *x=(const uint16_t*)Xh;
  for (int t=0;t<n_tiles;t++){
    int ks=t*Kt, ke=ks+Kt; if(ke>K) ke=K; int Kt2=ke-ks; int Kta=((Kt2+31)/32)*32;
    uint16_t *im=(uint16_t*)c->in_mm;
    memset(im, 0, M4*Kt*2+4096);
    for(int m=1;m<=M;m++) for(int k=1;k<=Kt2;k++){
      int sk=ks+k-1; if(sk<K) im[feature_data_off(Kta,M4,1,8,k,m,1)]=x[(m-1)*K+sk];
    }
    user_dcache_flush(c->in_mm, (uint64_t)M4*Kt*2+4096);
    uint64_t wt_dma = c->wt_dma[w_idx] + (uint64_t)t * tile_bytes;
    cna_submit_wt(c, M4, Kta, Np, wt_dma);
    float *om=(float*)c->out_mm;
    for(int m=1;m<=M;m++) for(int n=1;n<=N;n++)
      out32[(m-1)*N+(n-1)] += om[feature_data_off(Np,M4,1,4,n,m,1)];
  }
  return 0;
}

__attribute__((visibility("default")))
int npu_cna_cache_run(int w_idx, const void* Xh, void* Zh32) {
  if (g_cna.fd < 0 || w_idx < 0 || w_idx >= g_cna.n_w) return -1;
  CNAContext *c = &g_cna;
  int M=c->M, K=c->wt_K[w_idx], N=c->wt_N[w_idx], Np=c->wt_Np[w_idx], Kt=c->wt_Kt[w_idx];
  int n_tiles = c->wt_ntiles[w_idx];
  int tile_bytes = c->wt_tile_bytes[w_idx];
  float *out32 = (float*)Zh32;
  memset(out32, 0, M*N*4);
  const uint16_t *x=(const uint16_t*)Xh;
  for (int t=0;t<n_tiles;t++){
    int ks=t*Kt, ke=ks+Kt; if(ke>K) ke=K; int Kt2=ke-ks; int Kta=((Kt2+31)/32)*32;
    uint16_t *im=(uint16_t*)c->in_mm;
    memset(im, 0, M*Kt*2+4096);
    for(int m=1;m<=M;m++) for(int k=1;k<=Kt2;k++){
      int sk=ks+k-1; if(sk<K) im[feature_data_off(Kta,M,1,8,k,m,1)]=x[(m-1)*K+sk];
    }
    user_dcache_flush(c->in_mm, (uint64_t)M*Kt*2+4096);
    uint64_t wt_dma = c->wt_dma[w_idx] + (uint64_t)t * tile_bytes;
    cna_submit_wt(c, M, Kta, Np, wt_dma);
    float *om=(float*)c->out_mm;
    for(int m=1;m<=M;m++) for(int n=1;n<=N;n++)
      out32[(m-1)*N+(n-1)] += om[feature_data_off(Np,M,1,4,n,m,1)];
  }
  return 0;
}

__attribute__((visibility("default")))
int npu_cna_mm(int M, int K, int N, const void* Xh, const void* Wh, void* Zh32) {
  if (g_cna.fd < 0) {
    if (cna_init(M, K, N, 1) < 0) return -1;
  }
  if (g_cna.wt_K[0] != K || g_cna.wt_N[0] != N) {
    if (g_cna.wt_h[0]) { close(g_cna.wt_h[0]); g_cna.wt_h[0]=0; }
    npu_cna_cache_load(0, Wh, K, N);
  }
  return npu_cna_cache_run(0, Xh, Zh32);
}

__attribute__((visibility("default")))
void npu_cna_close(void) {
  if (g_cna.fd >= 0) { close(g_cna.fd); g_cna.fd=-1; }
}
__attribute__((visibility("default")))
int npu_cna_ready(void) { return g_cna.fd >= 0 ? 1 : 0; }

__attribute__((visibility("default")))
int npu_cna_wt_k(int w_idx) {
  if (w_idx < 0 || w_idx >= g_cna.n_w || g_cna.fd < 0) return 0;
  return g_cna.wt_K[w_idx];
}

__attribute__((visibility("default")))
int npu_cna_wt_n(int w_idx) {
  if (w_idx < 0 || w_idx >= g_cna.n_w || g_cna.fd < 0) return 0;
  return g_cna.wt_N[w_idx];
}

__attribute__((visibility("default")))
void npu_cna_set_core_mask(int mask) {
  if (g_cna.fd >= 0) g_cna.core_mask = (uint32_t)mask;
}

__attribute__((visibility("default")))
int npu_cna_get_core_mask(void) {
  return (int)g_cna.core_mask;
}

/* ---- NPU Attention: QK^T + softmax + AV via CNA matmuls ----
 *
 * Lowering:
 *   QK^T  = Q @ K^T   -> CNA matmul (M=dh->K, N=n_kv, K=dh... no: M=batch, K=dh, N=n_kv)
 *   scale = scores * (1/sqrt(dh))  -> CPU elementwise (trivial)
 *   softmax(scores)                -> CPU (NPU template only for seq=64)
 *   AV    = attn @ V   -> CNA matmul (M=batch, K=n_kv, N=dh)
 *
 * K and V are reloaded into CNA weight slots each call (KV cache grows).
 * For M=1 decode: CPU is faster (NPU per-submit overhead > compute).
 * For M>4 prefill: NPU matmuls beneficial.
 * The `use_npu` flag selects NPU vs CPU path.
 */
#define ATTN_K_SLOT 126
#define ATTN_V_SLOT 127

static uint16_t f32_to_fp16(float f);

static void softmax_row(float* row, int n) {
  float mx = row[0];
  for (int i = 1; i < n; i++) if (row[i] > mx) mx = row[i];
  float s = 0;
  for (int i = 0; i < n; i++) { row[i] = expf(row[i] - mx); s += row[i]; }
  float inv = 1.0f / s;
  for (int i = 0; i < n; i++) row[i] *= inv;
}

__attribute__((visibility("default")))
int npu_cna_attention(const float* q, const float* k_cache,
                      const float* v_cache, int n_heads, int M,
                      int dh, int n_kv, int use_npu, float* out) {
  float scale = 1.0f / sqrtf((float)dh);
  if (!use_npu || M < 4 || g_cna.fd < 0) {
    /* CPU attention (OpenMP for multi-head) */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < n_heads; h++) {
      const float* qh = q + h * M * dh;
      const float* kh = k_cache + h * n_kv * dh;
      const float* vh = v_cache + h * n_kv * dh;
      float* outh = out + h * M * dh;
      for (int m = 0; m < M; m++) {
        float scores[1024];
        const float* qm = qh + m * dh;
        for (int s = 0; s < n_kv; s++) {
          float dot = 0;
          const float* ks = kh + s * dh;
          for (int d = 0; d < dh; d++) dot += qm[d] * ks[d];
          scores[s] = dot * scale;
        }
        softmax_row(scores, n_kv);
        float* om = outh + m * dh;
        for (int d = 0; d < dh; d++) om[d] = 0;
        for (int s = 0; s < n_kv; s++) {
          float a = scores[s];
          const float* vs = vh + s * dh;
          for (int d = 0; d < dh; d++) om[d] += a * vs[d];
        }
      }
    }
    return 0;
  }
  /* NPU attention: CNA matmuls for QK^T and AV, CPU softmax */
  static uint16_t kt16[768 * 1024];
  static uint16_t v16[768 * 1024];
  static uint16_t q16[768 * 1024];
  static uint16_t s16[768 * 1024];
  static float scores[1024 * 1024];
  for (int h = 0; h < n_heads; h++) {
    const float* qh = q + h * M * dh;
    const float* kh = k_cache + h * n_kv * dh;
    const float* vh = v_cache + h * n_kv * dh;
    float* outh = out + h * M * dh;
    /* QK^T: Q[M,dh] @ K^T[dh,n_kv] -> scores[M,n_kv]
     * CNA weight = K^T [dh, n_kv] (K[h] is [n_kv, dh], transpose to [dh, n_kv])
     * CNA W is [K, N] row-major: W[d* n_kv + s] = K[h][s * dh + d] */
    for (int d = 0; d < dh; d++)
      for (int s = 0; s < n_kv; s++)
        kt16[d * n_kv + s] = f32_to_fp16(kh[s * dh + d]);
    if (g_cna.wt_h[ATTN_K_SLOT]) { close(g_cna.wt_h[ATTN_K_SLOT]); g_cna.wt_h[ATTN_K_SLOT] = 0; }
    npu_cna_cache_load(ATTN_K_SLOT, kt16, dh, n_kv);
    for (int m = 0; m < M; m++)
      for (int d = 0; d < dh; d++)
        q16[m * dh + d] = f32_to_fp16(qh[m * dh + d]);
    npu_cna_cache_run_m(ATTN_K_SLOT, M, q16, scores);
    /* Scale + softmax on CPU */
    for (int m = 0; m < M; m++) {
      float* row = scores + m * n_kv;
      for (int s = 0; s < n_kv; s++) row[s] *= scale;
      softmax_row(row, n_kv);
    }
    /* AV: attn[M,n_kv] @ V[n_kv,dh] -> out[M,dh]
     * CNA weight = V[h] [n_kv, dh] (no transpose needed) */
    for (int s = 0; s < n_kv; s++)
      for (int d = 0; d < dh; d++)
        v16[s * dh + d] = f32_to_fp16(vh[s * dh + d]);
    if (g_cna.wt_h[ATTN_V_SLOT]) { close(g_cna.wt_h[ATTN_V_SLOT]); g_cna.wt_h[ATTN_V_SLOT] = 0; }
    npu_cna_cache_load(ATTN_V_SLOT, v16, n_kv, dh);
    for (int m = 0; m < M; m++)
      for (int s = 0; s < n_kv; s++)
        s16[m * n_kv + s] = f32_to_fp16(scores[m * n_kv + s]);
    npu_cna_cache_run_m(ATTN_V_SLOT, M, s16, outh);
  }
  return 0;
}

/* ---- NPU LayerNorm: CPU reduce + NPU affine ----
 *
 * Lowering:
 *   mean = reduce_sum(X, -1) / H       -> CPU (NPU cannot reduce efficiently)
 *   var  = reduce_sum((X-mean)^2, -1)/H -> CPU
 *   inv  = 1/sqrt(var + eps)             -> CPU
 *   Y    = (X - mean) * inv * scale + B  -> CPU (NPU EW stage needs RE for
 *                                          per-channel params; NPU per-submit
 *                                          overhead > CPU cost for H=768)
 *
 * The NPU has a native LayerNorm template (captured at [1,64,768]) that does
 * the full op in one submit, but scale/B are BAKED into the scratch/regcmd
 * (not a patchable IO BO). For trained models, per-layer scale/B patching RE
 * is needed. This CPU implementation is the production path.
 */
__attribute__((visibility("default")))
int npu_cna_layernorm(const float* x, const float* scale_w,
                      const float* bias, int M, int H, float eps,
                      float* out) {
  #pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    const float* xm = x + m * H;
    float* om = out + m * H;
    float mean = 0;
    for (int h = 0; h < H; h++) mean += xm[h];
    mean /= H;
    float var = 0;
    for (int h = 0; h < H; h++) { float d = xm[h] - mean; var += d * d; }
    var /= H;
    float inv = 1.0f / sqrtf(var + eps);
    for (int h = 0; h < H; h++)
      om[h] = (xm[h] - mean) * inv * scale_w[h] + bias[h];
  }
  return 0;
}

/* ---- NPU Erf approximation via tanh ----
 *
 * Lowering: erf(x) -> tanh(1.128379167 * x * (1 + 0.044715 * x^2))
 * The NPU supports tanh natively (captured template). This approximation
 * has max error ~1e-4, sufficient for GELU.
 * Used in GELU: 0.5*x*(1+erf(x/sqrt(2)))
 *             = 0.5*x*(1+tanh(0.7978845*(x + 0.044715*x^3)))
 */
__attribute__((visibility("default")))
void npu_erf_approx(const float* x, int n, float* out) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    float v = x[i];
    float arg = 1.128379167f * v * (1.0f + 0.044715f * v * v);
    float a = arg < 0 ? -arg : arg;
    float th;
    if (a >= 3.5f) th = arg < 0 ? -1.0f : 1.0f;
    else {
      float a2 = arg * arg;
      th = arg * (135135.0f + a2 * (17325.0f + a2 * (378.0f + a2))) /
           (135135.0f + a2 * (62370.0f + a2 * (3150.0f + a2 * 28.0f)));
    }
    out[i] = th;
  }
}

#include <poll.h>

/* ---- 3-core parallel matmul (single fd, non-blocking + fence) ----
 *
 * The RK3588 NPU has 3 independent cores. This function submits 3 matmul
 * jobs to 3 cores using non-blocking submits with FENCE_OUT callbacks.
 * The kernel (6.1.141-vendor-rk35xx, rebuilt with CONFIG_ROCKCHIP_RKNPU_FENCE=y)
 * creates fence fds that we poll for completion.
 *
 * Uses the same fd as g_cna (shared IOMMU domain). Weight BOs from g_cna
 * are accessible to all 3 cores. Only input/output/regcmd BOs are per-core.
 *
 * Architecture:
 *   - 3 input BOs (one per core/image)
 *   - 3 output BOs (one per core/image)
 *   - 1 regcmd BO (4096 bytes, 3 blocks at offset i*1024)
 *   - 1 task BO (4096 bytes, 3 entries at index i, offset i*40)
 *   - 3 non-blocking submits (flags=PC|NONBLOCK|PINGPONG|FENCE_OUT)
 *   - Poll 3 fence fds for completion
 *   - subcore_task[core] set correctly for each core
 */

typedef struct {
  uint32_t in_h[3], out_h[3], rc_h, tk_h;
  uint64_t in_obj[3], out_obj[3], rc_obj, tk_obj;
  uint64_t in_dma[3], out_dma[3], rc_dma, tk_dma;
  void *in_mm[3], *out_mm[3], *rc_mm, *tk_mm;
  int M, max_K, max_N;
} CNA3Core;

static CNA3Core g_c3;

__attribute__((visibility("default")))
int npu_cna_3core_setup(int M, int max_K, int max_N) {
  if (g_cna.fd < 0) return -1;
  CNA3Core *c = &g_c3;
  memset(c, 0, sizeof(*c));
  c->M = M; c->max_K = max_K; c->max_N = max_N;
  int M4 = ((M + 3) / 4) * 4; if (M4 < 4) M4 = 4;
  int in_Kt = max_K > 768 ? 768 : ((max_K+31)/32)*32;
  int i;
  for (i = 0; i < 3; i++)
    if (cna_create_bo(g_cna.fd, M4*in_Kt*2+4096, CNA_FLAGS_BUF, &c->in_h[i], &c->in_obj[i], &c->in_dma[i], &c->in_mm[i])<0) return -2;
  for (i = 0; i < 3; i++)
    if (cna_create_bo(g_cna.fd, M4*max_N*4+4096, CNA_FLAGS_BUF, &c->out_h[i], &c->out_obj[i], &c->out_dma[i], &c->out_mm[i])<0) return -3;
  if (cna_create_bo(g_cna.fd, 4096, CNA_FLAGS_BUF, &c->rc_h, &c->rc_obj, &c->rc_dma, &c->rc_mm)<0) return -4;
  if (cna_create_bo(g_cna.fd, 4096, CNA_FLAGS_TASK, &c->tk_h, &c->tk_obj, &c->tk_dma, &c->tk_mm)<0) return -5;
  return 0;
}

__attribute__((visibility("default")))
int npu_cna_3core_run(int w_idx, int M,
                      const void* x0, const void* x1, const void* x2,
                      void* z0, void* z1, void* z2) {
  if (g_cna.fd < 0 || g_c3.rc_h == 0) return -1;
  CNA3Core *c = &g_c3;
  int K = g_cna.wt_K[w_idx], N = g_cna.wt_N[w_idx], Np = g_cna.wt_Np[w_idx];
  int Kt = g_cna.wt_Kt[w_idx], n_tiles = g_cna.wt_ntiles[w_idx];
  int tile_bytes = g_cna.wt_tile_bytes[w_idx];
  int M4 = M; if (M4 < 4) M4 = 4; if (M4 % 4) M4 = ((M4 + 3) / 4) * 4;
  const uint16_t* xs[3] = {(const uint16_t*)x0, (const uint16_t*)x1, (const uint16_t*)x2};
  float* zs[3] = {(float*)z0, (float*)z1, (float*)z2};
  int core, t;
  for (core = 0; core < 3; core++) memset(zs[core], 0, M * N * 4);

  for (t = 0; t < n_tiles; t++) {
    int ks = t * Kt, ke = ks + Kt; if (ke > K) ke = K;
    int Kt2 = ke - ks, Kta = ((Kt2 + 31) / 32) * 32;
    uint64_t wt_dma = g_cna.wt_dma[w_idx] + (uint64_t)t * tile_bytes;

    for (core = 0; core < 3; core++) {
      uint16_t *im = (uint16_t*)c->in_mm[core];
      memset(im, 0, M4 * Kt * 2 + 4096);
      const uint16_t *x = xs[core];
      int m, k;
      for (m = 1; m <= M; m++) for (k = 1; k <= Kt2; k++) {
        int sk = ks + k - 1;
        if (sk < K) im[feature_data_off(Kta, M4, 1, 8, k, m, 1)] = x[(m-1)*K + sk];
      }
      user_dcache_flush(c->in_mm[core], (uint64_t)M4 * Kt * 2 + 4096);
    }

    for (core = 0; core < 3; core++) {
      uint64_t ops[112]; memset(ops, 0, sizeof(ops));
      cna_gen_task(ops, M4, Kta, Np, (uint32_t)c->in_dma[core], (uint32_t)wt_dma, (uint32_t)c->out_dma[core]);
      memcpy((uint8_t*)c->rc_mm + core * 1024, ops, sizeof(ops));
    }
    user_dcache_flush(c->rc_mm, 4096);

    for (core = 0; core < 3; core++) {
      uint8_t *tk = (uint8_t*)c->tk_mm + core * 40;
      memset(tk, 0, 40);
      *(uint32_t*)(tk+8) = 0xd; *(uint32_t*)(tk+12) = 0x300; *(uint32_t*)(tk+16) = 0x1ffff;
      *(uint32_t*)(tk+24) = 112-(4+4); *(uint64_t*)(tk+32) = c->rc_dma + core * 1024;
    }
    user_dcache_flush(c->tk_mm, 4096);

    /* Batch submit: all 3 non-blocking jobs in one ioctl call.
     * The kernel driver's new rknpu_batch_submit processes all 3 jobs
     * in a single syscall, eliminating 2 ioctl round-trips and 2
     * power_get/put cycles. Each job uses FENCE_OUT for async completion. */
    {
      uint8_t jobs[3 * 104]; memset(jobs, 0, sizeof(jobs));
      for (core = 0; core < 3; core++) {
        uint8_t *sub = jobs + core * 104;
        *(uint32_t*)(sub+0) = 1|2|4|16;
        *(uint32_t*)(sub+4) = 6000;
        *(uint32_t*)(sub+12) = 1;
        *(uint64_t*)(sub+24) = c->tk_obj;
        *(uint32_t*)(sub+56) = 1 << core;
        *(int32_t*)(sub+60) = -1;
        *(uint32_t*)(sub+64+core*8) = core;
        *(uint32_t*)(sub+68+core*8) = 1;
      }
      uint8_t batch[16]; memset(batch, 0, 16);
      *(uint32_t*)(batch+0) = 3;
      *(uint64_t*)(batch+8) = (uint64_t)jobs;
      ioctl(g_cna.fd, CNA_BATCH_SUBMIT, batch);
      for (core = 0; core < 3; core++) {
        uint8_t *sub = jobs + core * 104;
        int fd = *(int32_t*)(sub+60);
        if (fd >= 0) {
          struct pollfd pfd = {fd, POLLIN, 0};
          poll(&pfd, 1, 6000);
          close(fd);
        }
      }
    }

    for (core = 0; core < 3; core++) {
      user_dcache_flush(c->out_mm[core], (uint64_t)M4*Np*4+4096);
      float *om = (float*)c->out_mm[core];
      int m, n;
      for (m = 1; m <= M; m++) for (n = 1; n <= N; n++)
        zs[core][(m-1)*N + (n-1)] += om[feature_data_off(Np, M4, 1, 4, n, m, 1)];
    }
  }
  return 0;
}

__attribute__((visibility("default")))
void npu_cna_3core_close(void) {
  CNA3Core *c = &g_c3;
  int i;
  for (i = 0; i < 3; i++) {
    if (c->in_mm[i]) { munmap(c->in_mm[i], 0); c->in_mm[i] = NULL; }
    if (c->out_mm[i]) { munmap(c->out_mm[i], 0); c->out_mm[i] = NULL; }
  }
  if (c->rc_mm) { munmap(c->rc_mm, 0); c->rc_mm = NULL; }
  if (c->tk_mm) { munmap(c->tk_mm, 0); c->tk_mm = NULL; }
  memset(c, 0, sizeof(*c));
}
