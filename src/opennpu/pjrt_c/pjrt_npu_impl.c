/* pjrt_npu_impl.c — real PJRT C plugin implementation for the RK3588 NPU.
 *
 * Open-stack: raw DRM ioctls to /dev/dri/card1, NO librknnrt at runtime.
 * Phase 1: client/device/memory model + error handling (jax.devices() lists NPU).
 * Phase 2: Compile (HLO -> NPU op) + Execute (raw ioctl submit) for elementwise.
 */
#define _GNU_SOURCE
#include "pjrt_c_api.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdio.h>

/* ---- NPU ioctl constants (RE'd, same as npu_dumper_v3.c / runtime.py) ---- */
#define NPU_DEV "/dev/dri/card1"
#define IOCTL_MEM_CREATE  0xC0306442
#define IOCTL_MEM_MAP     0xC0106443
#define IOCTL_MEM_SYNC    0xC0206445
#define IOCTL_MEM_DESTROY 0xC0106444
#define IOCTL_SUBMIT      0xC0686441
/* baked relu activation template (captured regcmd+task; DMAs patched at runtime) */
#include "relu_tmpl.h"
/* baked tanh activation template (3 submits, captured) */
#include "tanh_tmpl.h"
#include "matmul_tmpl.h"
#define IOCTL_ACTION      0xC0086440
#define IOCTL_BO_REGISTER 0xC008640a
#define IOCTL_PRIME_FD    0xC00C642D
#define CREATE_FLAGS_TASK 0x040b
#define CREATE_FLAGS_BUF 0x0403

static int npu_action(int fd, uint32_t flags, uint32_t value) {
  uint32_t buf[2] = {flags, value};
  ioctl(fd, IOCTL_ACTION, buf);
  return (int)buf[1];
}

struct npu_bo {
  int fd;
  uint32_t handle;
  uint64_t obj_addr;
  uint64_t dma_addr;
  uint64_t size;
  void* mm;
  int prime_fd;
};

static int npu_create_bo(int fd, uint64_t size, uint32_t flags, struct npu_bo* b) {
  uint8_t buf[48];
  memset(buf, 0, sizeof(buf));
  uint32_t* p = (uint32_t*)buf;
  p[0] = 0; p[1] = flags;
  *(uint64_t*)(buf+8) = size;
  *(uint32_t*)(buf+44) = 1; /* reserved=1 matches librknnrt */
  if (ioctl(fd, IOCTL_MEM_CREATE, buf) < 0) return -1;
  b->fd = fd;
  b->handle = *(uint32_t*)(buf+0);
  b->size = *(uint64_t*)(buf+8);
  b->obj_addr = *(uint64_t*)(buf+16);
  b->dma_addr = *(uint64_t*)(buf+24);
  /* MEM_MAP (CPU mapping). NOTE: we deliberately do NOT call BO_REGISTER
   * (0xC008640a) nor PRIME_FD here. BO_REGISTER leaks the IOVA permanently
   * (verified: a BO_REGISTER'd BO's IOVA is NOT released by MEM_DESTROY +
   * munmap + fd-close, and the leak persists, pushing the aperture top down).
   * The Python runtime never calls BO_REGISTER and every op works, so it is
   * not required for NPU access. Keeping the aperture clean is essential for
   * the aperture-top-pinned matmul. */
  /* MEM_MAP */
  uint8_t mb[16]; memset(mb,0,16);
  *(uint32_t*)(mb+0) = b->handle;
  ioctl(fd, IOCTL_MEM_MAP, mb);
  uint64_t off = *(uint64_t*)(mb+8);
  b->mm = mmap(NULL, b->size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, off);
  if (b->mm == MAP_FAILED) b->mm = NULL;
  /* BIDIR sync */
  uint8_t sb[32]; memset(sb,0,32);
  *(uint32_t*)(sb+0) = 3; *(uint64_t*)(sb+8) = b->obj_addr; *(uint64_t*)(sb+24) = b->size;
  ioctl(fd, IOCTL_MEM_SYNC, sb);
  npu_action(fd, 18, 0);
  return 0;
}

static void npu_sync_bo(int fd, uint64_t oa, uint64_t sz, uint32_t dir) {
  uint8_t sb[32]; memset(sb,0,32);
  *(uint32_t*)(sb+0) = dir; *(uint64_t*)(sb+8) = oa; *(uint64_t*)(sb+24) = sz;
  ioctl(fd, IOCTL_MEM_SYNC, sb);
}

static void npu_bo_free(struct npu_bo* b) {
  if (b->mm) { munmap(b->mm, b->size); b->mm = NULL; }
  if (b->prime_fd > 0) close(b->prime_fd);
  if (b->handle) {
    uint8_t db[16]; memset(db,0,16);
    *(uint32_t*)(db+0) = b->handle; *(uint64_t*)(db+8) = b->obj_addr;
    ioctl(b->fd, IOCTL_MEM_DESTROY, db);
  }
  b->handle = 0;
}

/* ---- PJRT internal types (opaque, we own them) ---- */
typedef struct NpuClient {
  int fd;
  int initialized;
  int n_devices;
  /* Persistent matmul kernel cache: one matmul fd (mfd) with its 6 BOs kept
   * open at the aperture top, reused for same-shape matmuls (no close+reopen,
   * no BO realloc, no DMA re-patch). Shape switch tears down the old mfd and
   * opens a fresh one at the top. c->fd is NEVER closed after the first matmul
   * (device_put inputs land below the cached mfd, so they don't push it out of
   * the DMA-reach window). This cuts ~144 close+reopens/forward to ~36. */
  int mm_mfd;                 /* -1 = none */
  int mm_shape[5];            /* (x0,x1,x2,w0,w1) of the cached BOs */
  const struct MMTemplate* mm_tpl;
  struct npu_bo mm_bo[6];
  /* multi-W-BO weight cache on the mfd: mm_wbo[0]==mm_bo[4] (the io1 BO);
   * mm_wbo[1..] are extra W BOs allocated below the kernel. Each is keyed by
   * the source device buffer's IOVA (mm_wsrc) so a W is copied ONCE and reused
   * across calls (layers). io1 DMA entries are repointed per call. */
  struct npu_bo mm_wbo[256];
  uint64_t mm_wsrc[256];   /* W content fingerprint word 0 */
  uint64_t mm_wsrc2[256];  /* W content fingerprint word 1 */
  int mm_nw;
  int mm_io1_off[32]; uint32_t mm_io1_suboff[32]; int mm_n_io1;
  int mm_nwpre;  /* number of pre-alloc'd W BOs (mm_wbo[1..mm_nwpre]) */
} NpuClient;
#define MM_NWPRE 12   /* pre-alloc'd W BOs per shape (enough for Q/K/V/O; decomposed LM sets higher) */

/* === Persistent cached-weight matmul globals (defined here, before the
 *     matmul Execute which delegates to npu_mm_cache_*). One dedicated fd
 *     (s_wc_mfd) with kernel BOs at the aperture top + weight BOs below. === */
static int s_wc_mfd = -1;
static struct npu_bo s_wc_mb[6];
static struct npu_bo* s_wc_wb = NULL;
static int s_wc_nw = 0;
static const MMTemplate* s_wc_tpl = NULL;
static int s_wc_n_io1 = 0;
static int s_wc_io1_off[32];
static uint32_t s_wc_io1_suboff[32];
static int s_wc_shape[3] = {0,0,0};  /* (M,K,N) of the current cached shape */
int npu_mm_cache_setup(int n_w, int M, int K, int N);
int npu_mm_cache_load(int w_idx, const void* Wh);
int npu_mm_cache_run(int w_idx, const void* Xh, void* Zh);
int npu_mm_cache_run_bias(int w_idx, const void* Xh, const float* bh, float* Zh32);
int npu_mm_cache_run_bias_res(int w_idx, const void* Xh, const float* bh, const float* resh, float* Zh32);
void npu_mm_cache_close(void);
/* CNA descriptor matmul (from cna_matmul.c, concatenated after this file) */
int npu_cna_cache_setup(int n_w, int M, int K, int N);
int npu_cna_cache_load(int w_idx, const void* Wh, int K, int N);
int npu_cna_cache_run(int w_idx, const void* Xh, void* Zh32);
int npu_cna_cache_run_m(int w_idx, int M, const void* Xh, void* Zh32);
void npu_cna_close(void);
int npu_cna_ready(void);
int npu_cna_wt_k(int w_idx);
int npu_cna_wt_n(int w_idx);
static uint16_t f32_to_fp16(float f);
/* KV-cached LM forward (from lm_forward_kv.c, concatenated after cna_matmul.c) */
int npu_lm_load_params(const char* path);
void npu_lm_reset(void);
int npu_lm_step(int token_id, float* logits);
int npu_lm_prefill(const int* ids, int n_ids, float* logits);
void npu_lm_profile(double* out);

typedef struct NpuDevice { int id; int local_hardware_id; } NpuDevice;
typedef struct NpuMemory { PJRT_Memory base; } NpuMemory;

typedef struct NpuBuffer {
  struct npu_bo bo;
  int64_t dims[8];
  int ndims;
  PJRT_Buffer_Type type;
  NpuClient* cli;
} NpuBuffer;

typedef struct NpuExecutable {
  NpuClient* cli;
  int op;                 /* 0=add (default), identified from HLO later */
  struct npu_bo relu_bos[5];  /* relu kernel BOs (relu sizes), pre-alloc'd at compile */
  int has_relu_bos;
  struct npu_bo tanh_bos[5];  /* tanh kernel BOs (tanh sizes), pre-alloc'd at compile */
  int has_tanh_bos;
  int64_t in_dims[2][8];
  int in_ndims[2];
  int64_t out_dims[8];
  int out_ndims;
  size_t out_dim_sizes;   /* number of dims in out_dims */
  PJRT_Buffer_Type dtype;
  /* multi-op linear elementwise chain (parsed from MLIR bytecode) */
  int is_graph;            /* parsed a supported DAG of binary elementwise ops */
  int n_ops;               /* number of result-producing ops (excl return) */
  int n_args;              /* number of function args */
  int g_ops[16];           /* op type per step: 0=add,1=sub,2=mul,3=div,4=relu,5=tanh,6=matmul */
  int g_opnd_isarg[16][2];  /* per operand: 1 = arg, 0 = prev op result */
  int g_opnd_idx[16][2];    /* arg index (if isarg) or op index (if !isarg) */
  int g_opnd_isconst[16][2];/* per operand: 1 = this operand is a baked scalar fp16 const */
  uint16_t g_const_val[16][2];/* the const fp16 value (bits) if g_opnd_isconst */
  int g_out_op;             /* op index whose result is the return value */
  int g_has_relu;          /* graph contains a relu op (needs pre-alloc'd relu BOs) */
  int g_has_tanh;          /* graph contains a tanh op (needs pre-alloc'd tanh BOs) */
  int g_has_matmul;        /* graph contains a matmul op (needs host-backed executor) */
  int parse_n_ops;         /* ops seen during parse (even if rejected) */
  int w_idx;               /* cached-weight matmul: weight BO index (op=7) */
} NpuExecutable;

/* singletons: one device, one memory */
static NpuDevice g_dev = {0, 0};
static NpuMemory g_mem;
static PJRT_Device* g_dev_arr[1] = { (PJRT_Device*)&g_dev };
static PJRT_Memory* g_mem_arr[1] = { (PJRT_Memory*)&g_mem };
static const PJRT_Memory_FunctionTable g_mem_vtable;

/* ---- Error model ---- */
typedef struct NpuError { char msg[128]; PJRT_Error_Code code; } NpuError;
static PJRT_Error* make_err(const char* m) {
  NpuError* e = (NpuError*)calloc(1, sizeof(NpuError));
  e->code = PJRT_Error_Code_UNIMPLEMENTED;
  if (m) { strncpy(e->msg, m, sizeof(e->msg)-1); }
  return (PJRT_Error*)e;
}

/* ---- Memory vtable (minimal: no user data) ---- */
static void* mem_get_user_data(PJRT_Memory* m, const void* k){ (void)m;(void)k; return NULL; }
static void  mem_set_user_data(PJRT_Memory* m, const void* k, void* d, void(*dt)(void*)){ (void)m;(void)k;(void)d;(void)dt; }
static const PJRT_Memory_FunctionTable g_mem_vtable = {
  sizeof(PJRT_Memory_FunctionTable), NULL, sizeof(PJRT_Memory),
  mem_get_user_data, mem_set_user_data
};

/* ---- stub event (ready immediately) ---- */
typedef struct NpuEvent { int dummy; } NpuEvent;
static NpuEvent g_event;

/* ====================== PJRT function impls ====================== */

static PJRT_Error* fn_Client_Create(void* a) {
  fprintf(stderr, "REAL Client_Create\n"); fflush(stderr);
  PJRT_Client_Create_Args* args = (PJRT_Client_Create_Args*)a;
  NpuClient* c = (NpuClient*)calloc(1, sizeof(NpuClient));
  c->mm_mfd = -1;  /* no cached matmul fd yet (calloc zeroes to 0, which is a valid fd!) */
  c->fd = open(NPU_DEV, O_RDWR);
  if (c->fd < 0) { free(c); return make_err("open /dev/dri/card1 failed"); }
  npu_action(c->fd, 0, 0xFFFFFFFFu);   /* GET_HW_VERSION */
  npu_action(c->fd, 1, 0);              /* GET_DRV_VERSION */
  npu_action(c->fd, 19, 0xFFFFFFEDu);  /* SET_PROC_NICE */
  npu_action(c->fd, 1, 0);
  npu_action(c->fd, 18, 0);             /* GET_IOMMU_EN */
  /* Pre-allocate the 768x768 weight cache NOW, before any device_put, so the
   * kernel BOs claim the aperture top and all device buffers land below. Then
   * the decomposed-LM matmul (all 768x768) never needs to close c->fd -- device
   * buffers stay alive (so jax.device_put-once + reuse-the-DeviceArray works)
   * and the W fingerprint is read straight from the device buffer mmap. */
  { int nwpre = MM_NWPRE; { const char*e=getenv("NPU_NWPRE"); if(e) nwpre=atoi(e); }
    if (npu_cna_cache_setup(nwpre, 64, 768, 768) >= 0) {
      s_wc_shape[0]=64; s_wc_shape[1]=768; s_wc_shape[2]=768; s_wc_nw=nwpre;
      const char* wd = getenv("NPU_WEIGHTS_DIR");
      if (wd) {
        for (int i=0; i < s_wc_nw; i++) {
          char path[512]; snprintf(path, sizeof(path), "%s/w%d.bin", wd, i);
          FILE* f = fopen(path, "rb");
          if (!f) break;
          fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
          uint8_t* buf = (uint8_t*)malloc(sz);
          if (buf) { if (fread(buf, 1, sz, f) == (size_t)sz) npu_cna_cache_load(i, buf, 768, 768); free(buf); }
          fclose(f);
        }
      }
    } }
  c->initialized = 1;
  c->n_devices = 1;
  g_mem.base.vtable = &g_mem_vtable;
  args->client = (PJRT_Client*)c;
  return NULL;
}

static PJRT_Error* fn_Client_Destroy(void* a) {
  fprintf(stderr, "REAL Client_Destroy\n"); fflush(stderr);
  PJRT_Client_Destroy_Args* args = (PJRT_Client_Destroy_Args*)a;
  NpuClient* c = (NpuClient*)args->client;
  if (c) {
    if (c->mm_mfd >= 0) { for(int i=0;i<6;i++) npu_bo_free(&c->mm_bo[i]); close(c->mm_mfd); }
    if (c->fd >= 0) close(c->fd);
    free(c);
  }
  return NULL;
}

static PJRT_Error* fn_Client_PlatformName(void* a) {
  fprintf(stderr, "REAL Client_PlatformName\n"); fflush(stderr);
  PJRT_Client_PlatformName_Args* args = (PJRT_Client_PlatformName_Args*)a;
  static const char* name = "npu";
  args->platform_name = name;
  args->platform_name_size = 3;
  return NULL;
}

static PJRT_Error* fn_Client_Devices(void* a) {
  fprintf(stderr, "REAL Client_Devices\n"); fflush(stderr);
  PJRT_Client_Devices_Args* args = (PJRT_Client_Devices_Args*)a;
  args->devices = g_dev_arr;
  args->num_devices = 1;
  return NULL;
}

static PJRT_Error* fn_Client_AddressableDevices(void* a) {
  fprintf(stderr, "REAL Client_AddressableDevices\n"); fflush(stderr);
  PJRT_Client_AddressableDevices_Args* args = (PJRT_Client_AddressableDevices_Args*)a;
  args->addressable_devices = g_dev_arr;
  args->num_addressable_devices = 1;
  return NULL;
}

static PJRT_Error* fn_Client_AddressableMemories(void* a) {
  fprintf(stderr, "REAL Client_AddressableMemories\n"); fflush(stderr);
  PJRT_Client_AddressableMemories_Args* args = (PJRT_Client_AddressableMemories_Args*)a;
  args->addressable_memories = g_mem_arr;
  args->num_addressable_memories = 1;
  return NULL;
}

static void devattr_deleter(PJRT_Device_Attributes* d){ (void)d; }
static PJRT_Error* fn_Device_GetAttributes(void* a) {
  fprintf(stderr, "REAL Device_GetAttributes\n"); fflush(stderr);
  PJRT_Device_GetAttributes_Args* args = (PJRT_Device_GetAttributes_Args*)a;
  /* Return a static Device_Attributes (opaque, we define it). jaxlib reads it
   * via known fields; we expose a minimal set through NamedValue array. */
  static PJRT_NamedValue attrs[4];
  static const char* kkind = "npu";
  static const char* kvendor = "rockchip";
  static const char* kdesc = "RK3588 NPU";
  memset(attrs, 0, sizeof(attrs));
  attrs[0].struct_size = sizeof(PJRT_NamedValue); attrs[0].name="platform_name"; attrs[0].name_size=13; attrs[0].type=PJRT_NamedValue_kString; attrs[0].string_value=kkind; attrs[0].value_size=3;
  attrs[1].struct_size = sizeof(PJRT_NamedValue); attrs[1].name="vendor"; attrs[1].name_size=6; attrs[1].type=PJRT_NamedValue_kString; attrs[1].string_value=kvendor; attrs[1].value_size=8;
  attrs[2].struct_size = sizeof(PJRT_NamedValue); attrs[2].name="description"; attrs[2].name_size=11; attrs[2].type=PJRT_NamedValue_kString; attrs[2].string_value=kdesc; attrs[2].value_size=10;
  attrs[3].struct_size = sizeof(PJRT_NamedValue); attrs[3].name="id"; attrs[3].name_size=2; attrs[3].type=PJRT_NamedValue_kInt64; attrs[3].int64_value=0; attrs[3].value_size=1;
  args->attributes = attrs;
  args->num_attributes = 4;
  args->device_attributes = NULL;       /* opaque struct; jaxlib tolerates NULL */
  args->attributes_deleter = devattr_deleter;
  return NULL;
}

static PJRT_Error* fn_Error_Destroy(void* a) {
  fprintf(stderr, "REAL Error_Destroy\n"); fflush(stderr);
  PJRT_Error_Destroy_Args* args = (PJRT_Error_Destroy_Args*)a;
  free(args->error);
  return NULL;
}

static PJRT_Error* fn_Error_Message(void* a) {
  fprintf(stderr, "REAL Error_Message\n"); fflush(stderr);
  PJRT_Error_Message_Args* args = (PJRT_Error_Message_Args*)a;
  NpuError* e = (NpuError*)args->error;
  args->message = e ? e->msg : "no error";
  args->message_size = e ? strlen(e->msg) : 8;
  return NULL;
}
static PJRT_Error* fn_Error_GetCode(void* a) {
  PJRT_Error_GetCode_Args* args = (PJRT_Error_GetCode_Args*)a;
  NpuError* e = (NpuError*)args->error;
  args->code = e ? e->code : PJRT_Error_Code_OK;
  return NULL;
}
static PJRT_Error* fn_Plugin_Initialize(void* a){ (void)a; return NULL; }

/* ---- Buffer lifecycle ---- */
static PJRT_Error* fn_Client_BufferFromHostBuffer(void* a) {
  fprintf(stderr, "REAL Client_BufferFromHostBuffer\n"); fflush(stderr);
  PJRT_Client_BufferFromHostBuffer_Args* args = (PJRT_Client_BufferFromHostBuffer_Args*)a;
  NpuClient* c = (NpuClient*)args->client;
  NpuBuffer* b = (NpuBuffer*)calloc(1, sizeof(NpuBuffer));
  b->cli = c;
  b->type = args->type;
  b->ndims = (int)args->num_dims;
  for (size_t i=0;i<args->num_dims;i++) b->dims[i]=args->dims[i];
  /* compute element count + size */
  int64_t n = 1;
  for (int i=0;i<b->ndims;i++) n *= b->dims[i];
  int esz = (b->type==PJRT_Buffer_Type_F32)?4:(b->type==PJRT_Buffer_Type_F16||b->type==PJRT_Buffer_Type_S16)?2:1;
  uint64_t sz = (uint64_t)n * esz;
  if (npu_create_bo(c->fd, sz, CREATE_FLAGS_BUF, &b->bo) < 0) { free(b); return make_err("create_bo failed"); }
  if (args->data && b->bo.mm) memcpy(b->bo.mm, args->data, sz);
  npu_sync_bo(c->fd, b->bo.obj_addr, b->bo.size, 1);
  args->buffer = (PJRT_Buffer*)b;
  args->done_with_host_buffer = NULL;
  return NULL;
}

static PJRT_Error* fn_Buffer_ToHostBuffer(void* a) {
  fprintf(stderr, "REAL Buffer_ToHostBuffer\n"); fflush(stderr);
  PJRT_Buffer_ToHostBuffer_Args* args = (PJRT_Buffer_ToHostBuffer_Args*)a;
  NpuBuffer* b = (NpuBuffer*)args->src;
  npu_sync_bo(b->bo.fd, b->bo.obj_addr, b->bo.size, 2);
  if (!args->dst) { args->dst_size = b->bo.size; args->event = (PJRT_Event*)&g_event; return NULL; }
  size_t cpy = args->dst_size < b->bo.size ? args->dst_size : b->bo.size;
  if (b->bo.mm) memcpy(args->dst, b->bo.mm, cpy);
  args->event = (PJRT_Event*)&g_event;
  return NULL;
}

static PJRT_Error* fn_Buffer_Destroy(void* a) {
  fprintf(stderr, "REAL Buffer_Destroy\n"); fflush(stderr);
  PJRT_Buffer_Destroy_Args* args = (PJRT_Buffer_Destroy_Args*)a;
  NpuBuffer* b = (NpuBuffer*)args->buffer;
  if (b) { npu_bo_free(&b->bo); free(b); }
  return NULL;
}
static PJRT_Error* fn_Buffer_Delete(void* a) {
  fprintf(stderr, "REAL Buffer_Delete\n"); fflush(stderr); return fn_Buffer_Destroy(a); }

static PJRT_Error* fn_Buffer_ElementType(void* a) {

  PJRT_Buffer_ElementType_Args* args = (PJRT_Buffer_ElementType_Args*)a;
  args->type = ((NpuBuffer*)args->buffer)->type;
  return NULL;
}
static PJRT_Error* fn_Buffer_Dimensions(void* a) {

  PJRT_Buffer_Dimensions_Args* args = (PJRT_Buffer_Dimensions_Args*)a;
  NpuBuffer* b = (NpuBuffer*)args->buffer;
  args->dims = b->dims; args->num_dims = b->ndims;
  return NULL;
}
static PJRT_Error* fn_Buffer_UnpaddedDimensions(void* a){
  PJRT_Buffer_UnpaddedDimensions_Args* args=(PJRT_Buffer_UnpaddedDimensions_Args*)a;
  NpuBuffer* b=(NpuBuffer*)args->buffer;
  args->unpadded_dims=b->dims; args->num_dims=b->ndims; return NULL; }
static PJRT_Error* fn_Buffer_DynamicDimensionIndices(void* a){
  PJRT_Buffer_DynamicDimensionIndices_Args* args=(PJRT_Buffer_DynamicDimensionIndices_Args*)a;
  args->dynamic_dim_indices=NULL; args->num_dynamic_dims=0; return NULL; }
static PJRT_Error* fn_Buffer_IsDeleted(void* a){ PJRT_Buffer_IsDeleted_Args* args=(PJRT_Buffer_IsDeleted_Args*)a; args->is_deleted=false; return NULL; }
static PJRT_Error* fn_Buffer_UnsafePointer(void* a){ PJRT_Buffer_UnsafePointer_Args* args=(PJRT_Buffer_UnsafePointer_Args*)a; NpuBuffer* b=(NpuBuffer*)args->buffer; args->buffer_pointer=(uintptr_t)b->bo.mm; return NULL; }
static PJRT_Error* fn_Buffer_IsOnCpu(void* a) {
 PJRT_Buffer_IsOnCpu_Args* args=(PJRT_Buffer_IsOnCpu_Args*)a; args->is_on_cpu=false; return NULL; }
static PJRT_Error* fn_Buffer_GetMemoryLayout(void* a) {

  PJRT_Buffer_GetMemoryLayout_Args* args=(PJRT_Buffer_GetMemoryLayout_Args*)a;
  NpuBuffer* b=(NpuBuffer*)args->buffer;
  static int64_t m2m[8];
  int n=b->ndims;
  for (int i=0;i<n;i++) m2m[i]=n-1-i;  /* minor-to-major: last dim fastest */
  args->layout.struct_size=sizeof(PJRT_Buffer_MemoryLayout);
  args->layout.extension_start=NULL;
  args->layout.type=PJRT_Buffer_MemoryLayout_Type_Tiled;
  args->layout.tiled.struct_size=sizeof(PJRT_Buffer_MemoryLayout_Tiled);
  args->layout.tiled.extension_start=NULL;
  args->layout.tiled.minor_to_major=m2m;
  return NULL; }
static PJRT_Error* fn_Buffer_Device(void* a) {
 PJRT_Buffer_Device_Args* args=(PJRT_Buffer_Device_Args*)a; args->device=(PJRT_Device*)&g_dev; return NULL; }
static PJRT_Error* fn_Buffer_OnDeviceSizeInBytes(void* a) {
  fprintf(stderr, "REAL Buffer_OnDeviceSizeInBytes\n"); fflush(stderr); PJRT_Buffer_OnDeviceSizeInBytes_Args* args=(PJRT_Buffer_OnDeviceSizeInBytes_Args*)a; args->on_device_size_in_bytes=((NpuBuffer*)args->buffer)->bo.size; return NULL; }
static PJRT_Error* fn_Buffer_Memory(void* a) {
 PJRT_Buffer_Memory_Args* args=(PJRT_Buffer_Memory_Args*)a; args->memory=(PJRT_Memory*)&g_mem; return NULL; }
static PJRT_Error* fn_Buffer_ReadyEvent(void* a) {
  fprintf(stderr, "REAL Buffer_ReadyEvent\n"); fflush(stderr); PJRT_Buffer_ReadyEvent_Args* args=(PJRT_Buffer_ReadyEvent_Args*)a; args->event=(PJRT_Event*)&g_event; return NULL; }
static PJRT_Error* fn_Event_IsReady(void* a){ PJRT_Event_IsReady_Args* args=(PJRT_Event_IsReady_Args*)a; args->is_ready=1; return NULL; }
static PJRT_Error* fn_Event_Await(void* a){ (void)a; return NULL; }
static PJRT_Error* fn_Event_Error(void* a){ (void)a; return NULL; }
static PJRT_Error* fn_Event_OnReady(void* a){ PJRT_Event_OnReady_Args* args=(PJRT_Event_OnReady_Args*)a; if(args->callback) args->callback(NULL, args->user_arg); return NULL; }
static PJRT_Error* fn_Event_Destroy(void* a) {
  fprintf(stderr, "REAL Event_Destroy\n"); fflush(stderr); (void)a; return NULL; }

/* ---- Topology + Device Description (init chain) ---- */
typedef struct NpuTopology { int dummy; } NpuTopology;
typedef struct NpuDeviceDescription { int id; int process_index; } NpuDeviceDescription;
static NpuTopology g_topo;
static NpuDeviceDescription g_devdesc = {0, 0};

static PJRT_Error* fn_Client_TopologyDescription(void* a){
  fprintf(stderr, "REAL Client_TopologyDescription\n"); fflush(stderr);
  PJRT_Client_TopologyDescription_Args* args=(PJRT_Client_TopologyDescription_Args*)a;
  args->topology=(PJRT_TopologyDescription*)&g_topo; return NULL;
}
static PJRT_Error* fn_TopologyDescription_PlatformName(void* a){
  fprintf(stderr, "REAL TopologyDescription_PlatformName\n"); fflush(stderr);
  PJRT_TopologyDescription_PlatformName_Args* args=(PJRT_TopologyDescription_PlatformName_Args*)a;
  static const char* n="npu"; args->platform_name=n; args->platform_name_size=3; return NULL;
}
static PJRT_Error* fn_TopologyDescription_PlatformVersion(void* a){
  fprintf(stderr, "REAL TopologyDescription_PlatformVersion\n"); fflush(stderr);
  PJRT_TopologyDescription_PlatformVersion_Args* args=(PJRT_TopologyDescription_PlatformVersion_Args*)a;
  static const char* v="opennpu 0.1 (RK3588 NPU)"; args->platform_version=v; args->platform_version_size=25; return NULL;
}
static PJRT_Error* fn_TopologyDescription_Attributes(void* a){
  fprintf(stderr, "REAL TopologyDescription_Attributes\n"); fflush(stderr);
  PJRT_TopologyDescription_Attributes_Args* args=(PJRT_TopologyDescription_Attributes_Args*)a;
  args->attributes=NULL; args->num_attributes=0; return NULL;
}
static PJRT_Error* fn_TopologyDescription_Destroy(void* a){
  fprintf(stderr, "REAL TopologyDescription_Destroy\n"); fflush(stderr); (void)a; return NULL; }
static PJRT_Error* fn_Client_PlatformVersion(void* a){
  fprintf(stderr, "REAL Client_PlatformVersion\n"); fflush(stderr);
  PJRT_Client_PlatformVersion_Args* args=(PJRT_Client_PlatformVersion_Args*)a;
  static const char* v="opennpu 0.1 (RK3588 NPU)"; args->platform_version=v; args->platform_version_size=25; return NULL;
}
static PJRT_Error* fn_Plugin_Attributes(void* a){
  fprintf(stderr, "REAL Plugin_Attributes\n"); fflush(stderr);
  PJRT_Plugin_Attributes_Args* args=(PJRT_Plugin_Attributes_Args*)a;
  args->attributes=NULL; args->num_attributes=0; return NULL;
}
static PJRT_Error* fn_Client_ProcessIndex(void* a){
  fprintf(stderr, "REAL Client_ProcessIndex\n"); fflush(stderr);
  PJRT_Client_ProcessIndex_Args* args=(PJRT_Client_ProcessIndex_Args*)a;
  args->process_index=0; return NULL;
}
static PJRT_Error* fn_Device_GetDescription(void* a){
  fprintf(stderr, "REAL Device_GetDescription\n"); fflush(stderr);
  PJRT_Device_GetDescription_Args* args=(PJRT_Device_GetDescription_Args*)a;
  args->device_description=(PJRT_DeviceDescription*)&g_devdesc; return NULL;
}
static PJRT_Error* fn_DeviceDescription_Id(void* a){
  fprintf(stderr, "REAL DeviceDescription_Id\n"); fflush(stderr);
  PJRT_DeviceDescription_Id_Args* args=(PJRT_DeviceDescription_Id_Args*)a;
  args->id=((NpuDeviceDescription*)args->device_description)->id; return NULL;
}
static PJRT_Error* fn_DeviceDescription_ProcessIndex(void* a){
  fprintf(stderr, "REAL DeviceDescription_ProcessIndex\n"); fflush(stderr);
  PJRT_DeviceDescription_ProcessIndex_Args* args=(PJRT_DeviceDescription_ProcessIndex_Args*)a;
  args->process_index=((NpuDeviceDescription*)args->device_description)->process_index; return NULL;
}
static PJRT_Error* fn_DeviceDescription_Attributes(void* a){
  fprintf(stderr, "REAL DeviceDescription_Attributes\n"); fflush(stderr);
  PJRT_DeviceDescription_Attributes_Args* args=(PJRT_DeviceDescription_Attributes_Args*)a;
  args->attributes=NULL; args->num_attributes=0; return NULL;
}
static PJRT_Error* fn_DeviceDescription_Kind(void* a){
  fprintf(stderr, "REAL DeviceDescription_Kind\n"); fflush(stderr);
  PJRT_DeviceDescription_Kind_Args* args=(PJRT_DeviceDescription_Kind_Args*)a;
  static const char* k="RK3588 NPU"; args->device_kind=k; args->device_kind_size=10; return NULL;
}
static PJRT_Error* fn_DeviceDescription_DebugString(void* a){
  fprintf(stderr, "REAL DeviceDescription_DebugString\n"); fflush(stderr);
  PJRT_DeviceDescription_DebugString_Args* args=(PJRT_DeviceDescription_DebugString_Args*)a;
  static const char* s="npu:0"; args->debug_string=s; args->debug_string_size=5; return NULL;
}
static PJRT_Error* fn_DeviceDescription_ToString(void* a){
  fprintf(stderr, "REAL DeviceDescription_ToString\n"); fflush(stderr);
  PJRT_DeviceDescription_ToString_Args* args=(PJRT_DeviceDescription_ToString_Args*)a;
  static const char* s="npu:0"; args->to_string=s; args->to_string_size=5; return NULL;
}
static PJRT_Error* fn_Device_AddressableMemories(void* a){
  fprintf(stderr, "REAL Device_AddressableMemories\n"); fflush(stderr);
  PJRT_Device_AddressableMemories_Args* args=(PJRT_Device_AddressableMemories_Args*)a;
  args->memories=g_mem_arr; args->num_memories=1; return NULL;
}
static PJRT_Error* fn_Memory_AddressableByDevices(void* a){
  fprintf(stderr, "REAL Memory_AddressableByDevices\n"); fflush(stderr);
  PJRT_Memory_AddressableByDevices_Args* args=(PJRT_Memory_AddressableByDevices_Args*)a;
  args->devices=g_dev_arr; args->num_devices=1; return NULL;
}
static PJRT_Error* fn_Client_DefaultDeviceAssignment(void* a){
  fprintf(stderr, "REAL Client_DefaultDeviceAssignment\n"); fflush(stderr);
  PJRT_Client_DefaultDeviceAssignment_Args* args=(PJRT_Client_DefaultDeviceAssignment_Args*)a;
  if(args->default_assignment && args->default_assignment_size>=1) args->default_assignment[0]=0; return NULL;
}
static PJRT_Error* fn_Client_LookupDevice(void* a){
  fprintf(stderr, "REAL Client_LookupDevice\n"); fflush(stderr);
  PJRT_Client_LookupDevice_Args* args=(PJRT_Client_LookupDevice_Args*)a;
  args->device=(PJRT_Device*)&g_dev; return NULL;
}
static PJRT_Error* fn_Device_IsAddressable(void* a){
  fprintf(stderr, "REAL Device_IsAddressable\n"); fflush(stderr);
  PJRT_Device_IsAddressable_Args* args=(PJRT_Device_IsAddressable_Args*)a;
  args->is_addressable=1; return NULL;
}
static PJRT_Error* fn_Device_LocalHardwareId(void* a){
  fprintf(stderr, "REAL Device_LocalHardwareId\n"); fflush(stderr);
  PJRT_Device_LocalHardwareId_Args* args=(PJRT_Device_LocalHardwareId_Args*)a;
  args->local_hardware_id=0; return NULL;
}
static PJRT_Error* fn_Device_DefaultMemory(void* a){
  fprintf(stderr, "REAL Device_DefaultMemory\n"); fflush(stderr);
  PJRT_Device_DefaultMemory_Args* args=(PJRT_Device_DefaultMemory_Args*)a;
  args->memory=(PJRT_Memory*)&g_mem; return NULL;
}
static PJRT_Error* fn_Device_MemoryStats(void* a){
  fprintf(stderr, "REAL Device_MemoryStats\n"); fflush(stderr);
  PJRT_Device_MemoryStats_Args* args=(PJRT_Device_MemoryStats_Args*)a;
  args->bytes_in_use=0; args->peak_bytes_in_use=0; args->peak_bytes_in_use_is_set=0;
  args->num_allocs=0; args->num_allocs_is_set=0;
  args->largest_alloc_size=0; args->largest_alloc_size_is_set=0;
  args->bytes_limit=0; args->bytes_limit_is_set=0;
  args->bytes_reserved=0; args->bytes_reserved_is_set=0;
  args->peak_bytes_reserved=0; args->peak_bytes_reserved_is_set=0;
  args->bytes_reservable_limit=0; args->bytes_reservable_limit_is_set=0;
  args->largest_free_block_bytes=0; args->largest_free_block_bytes_is_set=0;
  args->pool_bytes=0; args->pool_bytes_is_set=0;
  args->peak_pool_bytes=0; args->peak_pool_bytes_is_set=0;
  return NULL;
}
static PJRT_Error* fn_Memory_Id(void* a){
  fprintf(stderr, "REAL Memory_Id\n"); fflush(stderr);
  PJRT_Memory_Id_Args* args=(PJRT_Memory_Id_Args*)a;
  args->id=0; return NULL;
}
static PJRT_Error* fn_Memory_Kind(void* a){

  PJRT_Memory_Kind_Args* args=(PJRT_Memory_Kind_Args*)a;
  static const char* k="device"; args->kind=k; args->kind_size=6; return NULL;
}
static PJRT_Error* fn_Memory_Kind_Id(void* a){
  fprintf(stderr, "REAL Memory_Kind_Id\n"); fflush(stderr);
  PJRT_Memory_Kind_Id_Args* args=(PJRT_Memory_Kind_Id_Args*)a;
  args->kind_id=0; return NULL;
}
static PJRT_Error* fn_Memory_DebugString(void* a){
  fprintf(stderr, "REAL Memory_DebugString\n"); fflush(stderr);
  PJRT_Memory_DebugString_Args* args=(PJRT_Memory_DebugString_Args*)a;
  static const char* s="npu:0"; args->debug_string=s; args->debug_string_size=5; return NULL;
}
static PJRT_Error* fn_Memory_ToString(void* a){
  fprintf(stderr, "REAL Memory_ToString\n"); fflush(stderr);
  PJRT_Memory_ToString_Args* args=(PJRT_Memory_ToString_Args*)a;
  static const char* s="npu:0"; args->to_string=s; args->to_string_size=5; return NULL;
}
static PJRT_DeviceDescription* g_devdesc_arr[1] = { (PJRT_DeviceDescription*)&g_devdesc };
static PJRT_Error* fn_TopologyDescription_GetDeviceDescriptions(void* a){
  fprintf(stderr, "REAL TopologyDescription_GetDeviceDescriptions\n"); fflush(stderr);
  PJRT_TopologyDescription_GetDeviceDescriptions_Args* args=(PJRT_TopologyDescription_GetDeviceDescriptions_Args*)a;
  args->descriptions=g_devdesc_arr; args->num_descriptions=1; return NULL;
}
static PJRT_Error* fn_TopologyDescription_Fingerprint(void* a){
  fprintf(stderr, "REAL TopologyDescription_Fingerprint\n"); fflush(stderr);
  PJRT_TopologyDescription_Fingerprint_Args* args=(PJRT_TopologyDescription_Fingerprint_Args*)a;
  args->fingerprint=0x4e5055; return NULL;
}
static PJRT_Error* fn_TopologyDescription_GetMemorySpaceKindIds(void* a){
  fprintf(stderr, "REAL TopologyDescription_GetMemorySpaceKindIds\n"); fflush(stderr);
  PJRT_TopologyDescription_GetMemorySpaceKindIds_Args* args=(PJRT_TopologyDescription_GetMemorySpaceKindIds_Args*)a;
  args->memory_space_kind_ids=NULL; args->num_memory_space_kind_ids=0; return NULL;
}

/* ---- Compile / Execute (Phase 2: elementwise add via baked template) ---- */
/* ---- MLIR stablehlo bytecode parser: extract a linear elementwise chain ----
 * Parses the bytecode jax sends to Client_Compile, finds the function body
 * block, and if the ops form a strict linear elementwise chain (add/sub/mul/div),
 * fills ex->is_chain + the chain fields and returns 1. Else returns 0. */
static const uint8_t* g_bc; static size_t g_bc_cap;
static size_t g_slen[256], g_soff[256], g_nstr;
static size_t g_opname_str[64]; static size_t g_nopname;
static uint64_t bc_rvi(size_t* p){
  if(*p+1>g_bc_cap){ *p=g_bc_cap; return 0; }
  uint8_t b=g_bc[*p]; (*p)++;
  int nz=0; while(((b>>nz)&1)==0) nz++; if(nz>8) nz=8;
  uint64_t v=(uint64_t)(b>>(nz+1));
  for(int i=0;i<nz;i++){ if(*p>=g_bc_cap) break; v|=((uint64_t)g_bc[*p])<<((7-nz)+7*i); (*p)++; }
  return v;
}
static const char* bc_str(size_t i){ return i<g_nstr ? (const char*)(g_bc+g_soff[i]) : ""; }
/* collected body ops during recursive parse */
#define MAX_BODYOPS 32
static struct { size_t nameIdx; int nres; int nops; size_t ops[8]; size_t props_idx; int has_props; } g_body[MAX_BODYOPS];
static int g_nbody, g_body_nargs, g_body_found, g_collecting;
/* attr/properties tables for scalar-fp16 constant value extraction. The
 * constant_v1 op stores its 'value' attr in PROPERTIES (id=8, not op attrs); the
 * properties blob holds the attr INDEX of a DenseElementsAttr in the attr section
 * (id=2, offsets in id=3). A scalar fp16 dense attr is `1f 07 05 <2-byte LE>`. */
static size_t g_aentry_off[512]; static uint32_t g_aentry_sz[512]; static int g_numA;
static size_t g_attr_base;
static size_t g_prop_off[256]; static int g_num_props;
static void bc_parse_op(size_t* p, int depth);
static void bc_parse_region(size_t* p, int depth);
static void bc_parse_region_section(size_t* p, int depth);
static void bc_parse_block(size_t* p, int depth);
#define M_ATTR 0x01
#define M_RES  0x02
#define M_OP   0x04
#define M_SUCC 0x08
#define M_REG  0x10
#define M_USE  0x20
#define M_PROP 0x40
static void bc_parse_region_section(size_t* p, int depth){
  if(*p>=g_bc_cap) return;
  uint8_t idAlign=g_bc[*p]; (*p)++; size_t len=bc_rvi(p); int ha=(idAlign>>7)&1;
  if(ha){ size_t al=bc_rvi(p); while((*p)&(al-1)) (*p)++; }
  size_t rend=*p+len; bc_parse_region(p, depth); *p=rend;
}
static void bc_parse_region(size_t* p, int depth){
  size_t numBlocks=bc_rvi(p); size_t numValues=0; if(numBlocks) numValues=bc_rvi(p); (void)numValues;
  for(size_t b=0;b<numBlocks;b++) bc_parse_block(p, depth+1);
}
static void bc_parse_op(size_t* p, int depth){
  if(*p>=g_bc_cap||depth>16) return;
  size_t nameIdx=bc_rvi(p);
  uint8_t mask=g_bc[*p]; (*p)++;
  bc_rvi(p); /* location */
  if(mask&M_ATTR) bc_rvi(p);
  size_t props_idx=0; if(mask&M_PROP) props_idx=bc_rvi(p);
  int nres=0; if(mask&M_RES){ nres=(int)bc_rvi(p); for(int i=0;i<nres;i++) bc_rvi(p); }
  int nops=0; size_t ops[8];
  if(mask&M_OP){ nops=(int)bc_rvi(p); for(int i=0;i<nops&&i<8;i++) ops[i]=bc_rvi(p); }
  if(mask&M_SUCC){ size_t ns=bc_rvi(p); for(size_t i=0;i<ns;i++) bc_rvi(p); }
  if(mask&M_USE){ size_t nu=bc_rvi(p); for(size_t u=0;u<nu;u++){ bc_rvi(p); size_t ue=bc_rvi(p); int ni=(int)(ue>>1); for(int k=0;k<ni;k++) bc_rvi(p); } }
  if(g_collecting && g_nbody<MAX_BODYOPS){
    g_body[g_nbody].nameIdx=nameIdx; g_body[g_nbody].nres=nres; g_body[g_nbody].nops=nops;
    g_body[g_nbody].props_idx=props_idx; g_body[g_nbody].has_props=(mask&M_PROP)?1:0;
    for(int i=0;i<nops && i<8;i++) g_body[g_nbody].ops[i]=ops[i];
    g_nbody++;
  }
  if(mask&M_REG){
    size_t re=bc_rvi(p); int nr=(int)(re>>1); int isSec=(int)(re&1);
    for(int r=0;r<nr;r++){ if(isSec) bc_parse_region_section(p,depth+1); else bc_parse_region(p,depth+1); }
  }
}
static void bc_parse_block(size_t* p, int depth){
  if(*p>=g_bc_cap) return;
  size_t enc=bc_rvi(p); int numOps=(int)(enc>>1); int hasArgs=(int)(enc&1);
  if(numOps>4096) return;
  int nargs=0;
  if(hasArgs){
    size_t na=bc_rvi(p); nargs=(int)na;
    for(size_t a=0;a<na;a++){ size_t tl=bc_rvi(p); if(tl&1) bc_rvi(p); }
    uint8_t hasUL=g_bc[*p]; (*p)++; if(hasUL){ int ni=(int)na; for(int k=0;k<ni;k++){ bc_rvi(p); size_t ue=bc_rvi(p); int nj=(int)(ue>>1); for(int j=0;j<nj;j++) bc_rvi(p); } }
  }
  /* if this block has args, it's the function body -- collect its ops */
  if(hasArgs && !g_body_found){
    g_body_found=1; g_body_nargs=nargs; g_nbody=0; g_collecting=1;
    for(int i=0;i<numOps && g_nbody<MAX_BODYOPS && *p<g_bc_cap;i++) bc_parse_op(p, depth);
    g_collecting=0;
  } else {
    for(int i=0;i<numOps && *p<g_bc_cap;i++) bc_parse_op(p, depth);
  }
}
/* map an op name string to NPU op type: 0=add,1=sub,2=mul,3=div, 4=relu(maximum),
 * -1=other (tanh is handled by the single-op substring path, not as a graph op). */
static int opname_to_type(const char* s){
  if(!strcmp(s,"add_v1")) return 0;
  if(!strcmp(s,"subtract_v1")) return 1;
  if(!strcmp(s,"multiply_v1")) return 2;
  if(!strcmp(s,"divide_v1")) return 3;
  if(!strcmp(s,"maximum_v1")) return 4;
  return -1;
}
/* Parse the bytecode and, if the function body is a strict linear chain of
 * binary elementwise ops (add/sub/mul/div) arranged as any DAG (linear chains AND
 * trees), fill ex graph fields + return 1. Else returns 0 (and sets parse_n_ops
 * so the caller can decide: single-op substring vs unsupported compile error). */
/* Extract a scalar fp16 constant value from a constant_v1 op's properties blob.
 * props_idx = the op's properties varint (index into the properties offset table).
 * The properties blob is [dataSize varint][data]; the data's first varint is the
 * attr INDEX of the 'value' DenseElementsAttr. A scalar fp16 dense attr is
 * `1f 07 05 <2-byte LE fp16>`. Returns 1 and sets *out (fp16 bits) on success. */
static int extract_const_fp16(size_t props_idx, uint16_t* out){
  if(!g_num_props || props_idx >= (size_t)g_num_props) return 0;
  size_t p = g_prop_off[props_idx];
  if(p >= g_bc_cap) return 0;
  size_t ds = bc_rvi(&p);  /* dataSize (bytes of the properties data) */
  (void)ds;
  if(p >= g_bc_cap) return 0;
  size_t aidx = bc_rvi(&p);  /* first attr index = the "value" attr */
  if(aidx >= (size_t)g_numA || aidx >= 512) return 0;
  /* Decode the DenseElementsAttr entry (it lives in g_bc): kind varint (must
   * be 15 = DenseElements), then an element-type-index varint (VARIES per
   * bytecode -- NOT a fixed byte), then a flag/numel varint, then the raw
   * element bytes. For a scalar fp16 splat the trailing 2 bytes are the LE
   * fp16 value and nothing follows. Decode varints with bc_rvi so multi-byte
   * type indices are handled, then require exactly 2 trailing bytes. */
  size_t q = g_attr_base + g_aentry_off[aidx];
  size_t qend = q + g_aentry_sz[aidx];
  if(qend > g_bc_cap || q >= qend) return 0;
  size_t kind = bc_rvi(&q);  /* DenseElements kind */
  if(kind != 15) return 0;
  if(q >= qend) return 0;
  bc_rvi(&q);  /* element type index (skip -- varies per bytecode) */
  if(q >= qend) return 0;
  bc_rvi(&q);  /* flag/numel (skip) */
  if(q + 2 > qend) return 0;
  if(q + 2 != qend) return 0;  /* scalar fp16: exactly 2 trailing bytes */
  *out = (uint16_t)g_bc[q] | ((uint16_t)g_bc[q+1] << 8);
  return 1;
}
static int npu_parse_graph(const uint8_t* bc, size_t n, NpuExecutable* ex){
  g_bc=bc; g_bc_cap=n; g_nstr=0; g_nopname=0; g_body_found=0; g_nbody=0; g_collecting=0;
  g_numA=0; g_attr_base=0; g_num_props=0;
  size_t p=0;
  if(n<4 || memcmp(bc,"ML\xefR",4)) return 0; p+=4;
  bc_rvi(&p); /* version */
  while(g_bc[p]) p++; p++; /* producer null-terminated */
  size_t sec_off[16], sec_len[16]; memset(sec_off,0,sizeof(sec_off)); memset(sec_len,0,sizeof(sec_len));
  while(p<n){
    uint8_t idA=g_bc[p]; p++; int id=idA&0x7f; int ha=(idA>>7)&1;
    size_t len=bc_rvi(&p);
    if(ha){ size_t al=bc_rvi(&p); while((p&(al-1))) p++; }
    if(id<16){ sec_off[id]=p; sec_len[id]=len; }
    p+=len;
  }
  if(!sec_len[0]||!sec_len[1]||!sec_len[4]) return 0;
  /* attr offset table (id=3) + attr data base (id=2) — for constant value
   * extraction. Each attr entry: (size, hasCustom) in the offset section, data
   * in the attr section at a cumulative offset. */
  if(sec_len[2]&&sec_len[3]){
    g_attr_base=sec_off[2];
    size_t op3=sec_off[3];
    size_t numA=bc_rvi(&op3), numT=bc_rvi(&op3); (void)numT;
    size_t co=0; size_t idx=0; size_t tot=numA+numT;
    while(idx<tot && idx<512){
      size_t di=bc_rvi(&op3); size_t no=bc_rvi(&op3);
      for(size_t i=0;i<no && idx<tot;i++){
        size_t enc=bc_rvi(&op3); size_t es=enc>>1; (void)(enc&1);
        if(idx<512){ g_aentry_off[idx]=co; g_aentry_sz[idx]=(uint32_t)es; }
        co+=es; idx++;
      }
    }
    g_numA=(int)numA;
  }
  /* properties section (id=8): count + offset table of (dataSize+data) blobs.
   * g_prop_off[i] = absolute offset in g_bc of entry i's dataSize varint. */
  if(sec_len[8]){
    size_t pp=sec_off[8]; size_t count=bc_rvi(&pp); /* num property blobs */
    size_t pend=sec_off[8]+sec_len[8];
    for(size_t i=0;i<count && i<256 && pp<pend;i++){
      g_prop_off[i]=pp;
      size_t ds=bc_rvi(&pp); pp+=ds; /* skip dataSize + data */
    }
    g_num_props=(int)count;
  }
  size_t sp=sec_off[0]; g_nstr=bc_rvi(&sp);
  size_t rlen[256]; for(size_t i=0;i<g_nstr;i++) rlen[i]=bc_rvi(&sp);
  for(size_t i=0;i<g_nstr;i++) g_slen[i]=rlen[g_nstr-1-i];
  size_t sd=sp, acc=0; for(size_t i=0;i<g_nstr;i++){ g_soff[i]=sd+acc; acc+=g_slen[i]; }
  size_t dp=sec_off[1], dend=dp+sec_len[1];
  size_t nd=bc_rvi(&dp);
  for(size_t i=0;i<nd;i++){ size_t nv=bc_rvi(&dp); if(nv&1){ size_t vsz=bc_rvi(&dp); dp+=vsz; } }
  size_t topN=bc_rvi(&dp);
  while(g_nopname<topN && dp<dend){
    bc_rvi(&dp); size_t no=bc_rvi(&dp);
    for(size_t i=0;i<no && g_nopname<topN;i++){ size_t nr=bc_rvi(&dp); if(g_nopname<64) g_opname_str[g_nopname]=nr>>1; g_nopname++; }
  }
  size_t ip=sec_off[4]; bc_parse_block(&ip, 0);
  ex->parse_n_ops = g_body_found ? (g_nbody>0 ? g_nbody-1 : 0) : 0;  /* excl return */
  if(!g_body_found || g_nbody<2) return 0;  /* need >=1 op + return */
  int ri=g_nbody-1;
  if(strcmp(bc_str(g_opname_str[g_body[ri].nameIdx]),"return_v1")) return 0;
  if(g_body[ri].nops!=1) return 0;
  int k=g_nbody-1;
  if(k>30) return 0;
  /* result value index for each op (args occupy 0..nargs-1, then op results in def order) */
  int nextval=g_body_nargs;
  int resval[32];
  for(int i=0;i<k;i++){
    if(g_body[i].nres!=1) return 0;
    resval[i]=nextval; nextval+=1;
  }
  /* Map bytecode op -> executable op. Some bytecode ops are BAKED and skipped
   * (constant_v1, broadcast_in_dim_v1 of a constant): the NPU relu kernel bakes
   * max-with-0, so we recognize maximum(x, broadcast(constant)) as relu(x). We
   * track which op-result VALUES are constants so maximum's const operand can be
   * dropped. g_ops indices (ng) differ from bytecode op indices (i). */
  int is_const_v[64]; int gop_of[64]; uint16_t const_val[64]; int has_cval[64];
  for(int i=0;i<64;i++){ is_const_v[i]=0; gop_of[i]=-1; const_val[i]=0; has_cval[i]=0; }
  int ng=0, has_relu=0, has_tanh=0, has_const_opnd=0, has_matmul=0;
  for(int i=0;i<k;i++){
    const char* nm=bc_str(g_opname_str[g_body[i].nameIdx]);
    int rv=resval[i];
    if(!strcmp(nm,"constant_v1")){            /* baked constant: extract scalar fp16 value if present */
      if(g_body[i].nops!=0) return 0;
      is_const_v[rv]=1;
      if(g_body[i].has_props){ uint16_t cv; if(extract_const_fp16(g_body[i].props_idx, &cv)){ const_val[rv]=cv; has_cval[rv]=1; } }
      continue;
    }
    if(!strcmp(nm,"broadcast_in_dim_v1")){    /* broadcast of a constant -> still const (value propagates) */
      if(g_body[i].nops!=1) return 0;
      int ov=(int)g_body[i].ops[0];
      is_const_v[rv] = (ov<g_body_nargs)?0:is_const_v[ov];
      const_val[rv] = (ov<g_body_nargs)?0:const_val[ov];
      has_cval[rv] = (ov<g_body_nargs)?0:has_cval[ov];
      continue;
    }
    if(!strcmp(nm,"maximum_v1")){              /* relu = max(x, broadcast(constant(0))) */
      if(g_body[i].nops!=2) return 0;
      int v0=(int)g_body[i].ops[0], v1=(int)g_body[i].ops[1];
      int c0=(v0<g_body_nargs)?0:is_const_v[v0];
      int c1=(v1<g_body_nargs)?0:is_const_v[v1];
      int inv=-1, cval_ok=0;
      if(c0 && !c1){ inv=v1; cval_ok = (v0<g_body_nargs)?0:(has_cval[v0] && const_val[v0]==0); }
      else if(c1 && !c0){ inv=v0; cval_ok = (v1<g_body_nargs)?0:(has_cval[v1] && const_val[v1]==0); }
      else return 0;  /* need exactly one const operand */
      if(!cval_ok) return 0;  /* relu requires the const to be 0 (kernel bakes max-with-0); non-zero max is unsupported */
      if(ng>=16) return 0;
      ex->g_ops[ng]=4; has_relu=1;
      if(inv<g_body_nargs){ ex->g_opnd_isarg[ng][0]=1; ex->g_opnd_idx[ng][0]=inv; }
      else { int oi=gop_of[inv]; if(oi<0) return 0; ex->g_opnd_isarg[ng][0]=0; ex->g_opnd_idx[ng][0]=oi; }
      ex->g_opnd_isarg[ng][1]=0; ex->g_opnd_idx[ng][1]=-1;
      gop_of[rv]=ng; ng++; continue;
    }
    if(!strncmp(nm,"tanh",4)){           /* tanh: unary (1 operand), op=5. Matches tanh/tanh_v1/tanh_v2 (jax sends tanh_v2). jnp.tanh/jax.nn.tanh lower to a single stablehlo.tanh (flat, no call). */
      if(g_body[i].nops!=1) return 0;
      int inv=(int)g_body[i].ops[0];
      if(ng>=16) return 0;
      ex->g_ops[ng]=5; has_tanh=1;
      if(inv<g_body_nargs){ ex->g_opnd_isarg[ng][0]=1; ex->g_opnd_idx[ng][0]=inv; }
      else { int oi=gop_of[inv]; if(oi<0) return 0; ex->g_opnd_isarg[ng][0]=0; ex->g_opnd_idx[ng][0]=oi; }
      ex->g_opnd_isarg[ng][1]=0; ex->g_opnd_idx[ng][1]=-1;
      gop_of[rv]=ng; ng++; continue;
    }
    if(!strncmp(nm,"dot_general",11)){  /* matmul: 2 tensor operands (X 3D, W 2D); dimension_numbers is an attribute */
      if(g_body[i].nops!=2) return 0;
      if(ng>=16) return 0;
      ex->g_ops[ng]=6; has_matmul=1;
      for(int j=0;j<2;j++){
        int v=(int)g_body[i].ops[j];
        ex->g_opnd_isconst[ng][j]=0;
        if(v<g_body_nargs){ ex->g_opnd_isarg[ng][j]=1; ex->g_opnd_idx[ng][j]=v; }
        else { int oi=gop_of[v]; if(oi<0) return 0; ex->g_opnd_isarg[ng][j]=0; ex->g_opnd_idx[ng][j]=oi; }
      }
      gop_of[rv]=ng; ng++; continue;
    }
    int t=opname_to_type(nm);                 /* add/sub/mul/div (binary) */
    if(t<0) return 0;
    if(g_body[i].nops!=2) return 0;
    if(ng>=16) return 0;
    ex->g_ops[ng]=t;
    for(int j=0;j<2;j++){
      int v=(int)g_body[i].ops[j];
      ex->g_opnd_isconst[ng][j]=0;
      if(v<g_body_nargs){ ex->g_opnd_isarg[ng][j]=1; ex->g_opnd_idx[ng][j]=v; }
      else if(is_const_v[v]){
        if(!has_cval[v]) return 0;  /* non-scalar-fp16 const operand unsupported */
        ex->g_opnd_isarg[ng][j]=0; ex->g_opnd_idx[ng][j]=-1;
        ex->g_opnd_isconst[ng][j]=1; ex->g_const_val[ng][j]=const_val[v]; has_const_opnd=1;
      }
      else {
        int oi=gop_of[v]; if(oi<0) return 0;
        ex->g_opnd_isarg[ng][j]=0; ex->g_opnd_idx[ng][j]=oi;
      }
    }
    gop_of[rv]=ng; ng++;
  }
  /* return operand must map to an executable op result */
  int retv=(int)g_body[ri].ops[0];
  int out_op=gop_of[retv];
  if(out_op<0) return 0;  /* returning a bare arg / const = identity, not a graph */
  if(ng<2 && !has_const_opnd) return 0;  /* single plain op -> dedicated path; single-op-with-const -> graph */
  ex->n_ops=ng; ex->n_args=g_body_nargs; ex->g_out_op=out_op; ex->g_has_relu=has_relu; ex->g_has_tanh=has_tanh; ex->is_graph=1; ex->g_has_matmul=has_matmul;
  return 1;
}
static PJRT_Error* fn_Client_Compile(void* a) {
  fprintf(stderr, "REAL Client_Compile\n"); fflush(stderr);
  PJRT_Client_Compile_Args* args = (PJRT_Client_Compile_Args*)a;
  NpuClient* c = (NpuClient*)args->client;
  NpuExecutable* ex = (NpuExecutable*)calloc(1, sizeof(NpuExecutable));
  ex->cli = c;
  ex->op = -1;  /* unknown until detected */
  /* --- parse MLIR bytecode for a DAG of binary elementwise ops --- */
  {
    const PJRT_Program* p = args->program;
    if (p && p->code && p->code_size) {
      npu_parse_graph((const uint8_t*)p->code, p->code_size, ex);
    }
  }
  /* Single-op fallback ONLY for genuine single-op functions (relu/tanh): detect
   * from MLIR stablehlo op-name strings. A multi-op function that did NOT parse
   * as a graph is UNSUPPORTED (mixed elementwise+activation, or unsupported ops)
   * -> fail loudly with a compile error rather than silently producing wrong
   * output by running just one op. */
  if (!ex->is_graph) {
    const PJRT_Program* p = args->program;
    if (p && p->code && p->code_size) {
      const char* c = p->code; size_t n = p->code_size;
      #define HAS(needle, k) ({ const char* _h=c; size_t _i,_j,_f=0; \
        for(_i=0;_i+k<=n && !_f;_i++){ _f=1; for(_j=0;_j<k;_j++) if(_h[_i+_j]!=needle[_j]){_f=0;break;} } _f; })
      int single_ok = 0;
      /* cached-weight matmul custom op: stablehlo.custom_call @npu_cached_mm
       * with backend_config "w_idx=N". W is a plugin-managed cached handle (no
       * jax device buffer for W -> no cross-fd / device-put issues); only X is a
       * jax buffer. Detect by target-name substring, parse w_idx from config. */
      if (HAS("npu_cached_mm", 13)) {
        ex->op = 7; ex->w_idx = -1;
        const char* wp = NULL;
        for (size_t i=0; i+6<=n; i++) if (!memcmp(c+i,"w_idx=",6)){ wp=c+i+6; break; }
        if (wp) { int v=0; while(*wp>='0'&&*wp<='9'){ v=v*10+(*wp-'0'); wp++; } ex->w_idx=v; }
        single_ok = 1;
      } else
      /* only accept a single-op substring match if the bytecode genuinely
       * contains exactly ONE result-producing op (else it is a multi-op
       * function we cannot execute correctly as a single op). Single add/sub/
       * mul/div run via the dedicated elementwise path; relu/tanh via theirs. */
      if (ex->parse_n_ops <= 1) {
        if (HAS("subtract_v1", 11)){ ex->op = 1; single_ok = 1; }
        else if (HAS("multiply_v1", 11)){ ex->op = 2; single_ok = 1; }
        else if (HAS("divide_v1", 9)){ ex->op = 3; single_ok = 1; }
        else if (HAS("maximum_v1", 10)){ ex->op = 4; single_ok = 1; }   /* relu = max(x,0) */
        else if (HAS("tanh_v1", 7)){ ex->op = 5; single_ok = 1; }
        else if (HAS("tanh_v2", 7)){ ex->op = 5; single_ok = 1; }
        else if (HAS("tanh", 4)){ ex->op = 5; single_ok = 1; }
        else if (HAS("add_v1", 5)){ ex->op = 0; single_ok = 1; }
        else if (HAS("dot_general", 11)){ ex->op = 6; single_ok = 1; }  /* matmul (shape-pinned 1x64x768 @ 768x3072) */
      }
      #undef HAS
      if (!single_ok && ex->parse_n_ops >= 2) {
        return make_err("npu: unsupported multi-op program (not a pure binary-elementwise DAG)");
      }
      if (!single_ok && ex->parse_n_ops == 0) {
        return make_err("npu: unsupported program (no NPU ops)");
      }
      if (!single_ok && ex->parse_n_ops == 1) {
        return make_err("npu: unsupported single-op program");
      }
    }
  }
  /* Pre-allocate the activation kernel BOs NOW (at compile time, before any
   * input transfer / device_put) so they are the FIRST allocation -> top of the
   * IOVA pool. Relu and tanh chains are IOVA-position-sensitive: they need their
   * kernel BOs above any device buffer (else wrong output / hang), and the window
   * only holds ONE set, so pre-alloc only the set matching the detected op. This
   * requires the jit to compile BEFORE the input is transferred to the device
   * (i.e. jax.jit(f)(host_array), the common pattern -- NOT device_put then jit). */
  if (ex->op == 4) {
    uint64_t rsz[5] = {RELU_SZ_TASK, RELU_SZ_REGCMD, RELU_SZ_SCRATCH, RELU_SZ_IN, RELU_SZ_OUT};
    uint32_t rflg[5] = {CREATE_FLAGS_TASK, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF};
    ex->has_relu_bos = 1;
    for (int i=0;i<5;i++){ if (npu_create_bo(c->fd, rsz[i], rflg[i], &ex->relu_bos[i])<0){ ex->has_relu_bos=0; break; } }
    for (int i=0;i<5 && ex->has_relu_bos;i++) npu_sync_bo(c->fd, ex->relu_bos[i].obj_addr, ex->relu_bos[i].size, 3);
  } else if (ex->op == 5) {
    uint64_t tsz[5] = {TANH_SZ_TASK, TANH_SZ_REGCMD, TANH_SZ_SCRATCH, TANH_SZ_IN, TANH_SZ_OUT};
    uint32_t tflg[5] = {CREATE_FLAGS_TASK, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF};
    ex->has_tanh_bos = 1;
    for (int i=0;i<5;i++){ if (npu_create_bo(c->fd, tsz[i], tflg[i], &ex->tanh_bos[i])<0){ ex->has_tanh_bos=0; break; } }
    for (int i=0;i<5 && ex->has_tanh_bos;i++) npu_sync_bo(c->fd, ex->tanh_bos[i].obj_addr, ex->tanh_bos[i].size, 3);
  }
  /* A multi-op graph containing a relu op also needs the relu kernel BOs
   * pre-allocated at the top of the IOVA pool (relu is IOVA-position-sensitive).
   * Same compile-before-transfer requirement as single relu. A graph containing
   * BOTH relu and tanh cannot run: their kernel footprints (503808 + 614400)
   * exceed the single IOVA window, so only ONE activation family fits at the
   * pool top. Reject such graphs at compile (loud error, not wrong output). */
  if (ex->is_graph && ex->g_has_relu && ex->g_has_tanh) {
    return make_err("npu: graph has both relu and tanh (IOVA window too small)");
  }
  if (ex->is_graph && ex->g_has_matmul==0 && ex->g_has_relu && !ex->has_relu_bos) {
    uint64_t rsz[5] = {RELU_SZ_TASK, RELU_SZ_REGCMD, RELU_SZ_SCRATCH, RELU_SZ_IN, RELU_SZ_OUT};
    uint32_t rflg[5] = {CREATE_FLAGS_TASK, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF};
    ex->has_relu_bos = 1;
    for (int i=0;i<5;i++){ if (npu_create_bo(c->fd, rsz[i], rflg[i], &ex->relu_bos[i])<0){ ex->has_relu_bos=0; break; } }
    for (int i=0;i<5 && ex->has_relu_bos;i++) npu_sync_bo(c->fd, ex->relu_bos[i].obj_addr, ex->relu_bos[i].size, 3);
  }
  if (ex->is_graph && ex->g_has_matmul==0 && ex->g_has_tanh && !ex->has_tanh_bos) {
    uint64_t tsz[5] = {TANH_SZ_TASK, TANH_SZ_REGCMD, TANH_SZ_SCRATCH, TANH_SZ_IN, TANH_SZ_OUT};
    uint32_t tflg[5] = {CREATE_FLAGS_TASK, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF};
    ex->has_tanh_bos = 1;
    for (int i=0;i<5;i++){ if (npu_create_bo(c->fd, tsz[i], tflg[i], &ex->tanh_bos[i])<0){ ex->has_tanh_bos=0; break; } }
    for (int i=0;i<5 && ex->has_tanh_bos;i++) npu_sync_bo(c->fd, ex->tanh_bos[i].obj_addr, ex->tanh_bos[i].size, 3);
  }
  /* default shape (1,64,768) fp16 — overridden at execute by buffer shapes */
  ex->dtype = PJRT_Buffer_Type_F16;
  ex->out_ndims = 3; ex->out_dims[0]=1; ex->out_dims[1]=64; ex->out_dims[2]=768; ex->out_dim_sizes=3;
  args->executable = (PJRT_LoadedExecutable*)ex;
  return NULL;
}
static PJRT_Error* fn_Executable_NumReplicas(void* a){ fprintf(stderr,"REAL fn_Executable_NumReplicas\n"); fflush(stderr); PJRT_Executable_NumReplicas_Args* q=(PJRT_Executable_NumReplicas_Args*)a; q->num_replicas=1; return NULL; }
static PJRT_Error* fn_Executable_NumPartitions(void* a){ fprintf(stderr,"REAL fn_Executable_NumPartitions\n"); fflush(stderr); PJRT_Executable_NumPartitions_Args* q=(PJRT_Executable_NumPartitions_Args*)a; q->num_partitions=1; return NULL; }
static PJRT_Error* fn_Executable_NumOutputs(void* a){ fprintf(stderr,"REAL fn_Executable_NumOutputs\n"); fflush(stderr); PJRT_Executable_NumOutputs_Args* q=(PJRT_Executable_NumOutputs_Args*)a; q->num_outputs=1; return NULL; }
static PJRT_Error* fn_Executable_OutputElementTypes(void* a){ fprintf(stderr,"REAL fn_Executable_OutputElementTypes\n"); fflush(stderr); PJRT_Executable_OutputElementTypes_Args* q=(PJRT_Executable_OutputElementTypes_Args*)a; NpuExecutable* ex=(NpuExecutable*)q->executable; q->output_types=&ex->dtype; q->num_output_types=1; return NULL; }
static PJRT_Error* fn_Executable_OutputDimensions(void* a){ fprintf(stderr,"REAL fn_Executable_OutputDimensions\n"); fflush(stderr); PJRT_Executable_OutputDimensions_Args* q=(PJRT_Executable_OutputDimensions_Args*)a; NpuExecutable* ex=(NpuExecutable*)q->executable; q->dims=ex->out_dims; q->dim_sizes=&ex->out_dim_sizes; q->num_outputs=1; return NULL; }
static PJRT_Error* fn_Executable_OutputMemoryKinds(void* a){ fprintf(stderr,"REAL fn_Executable_OutputMemoryKinds\n"); fflush(stderr); PJRT_Executable_OutputMemoryKinds_Args* q=(PJRT_Executable_OutputMemoryKinds_Args*)a; static const char* mk[1]={"device"}; static const size_t ms[1]={6}; q->memory_kinds=mk; q->memory_kind_sizes=ms; q->num_outputs=1; return NULL; }
static PJRT_Error* fn_Executable_Name(void* a){ fprintf(stderr,"REAL fn_Executable_Name\n"); fflush(stderr); PJRT_Executable_Name_Args* q=(PJRT_Executable_Name_Args*)a; static const char* n="npu_add"; q->executable_name=n; q->executable_name_size=7; return NULL; }
static PJRT_Error* fn_Executable_SizeOfGeneratedCodeInBytes(void* a){ fprintf(stderr,"REAL fn_Executable_SizeOfGeneratedCodeInBytes\n"); fflush(stderr); PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* q=(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args*)a; q->size_in_bytes=0; return NULL; }
static PJRT_Error* fn_Executable_GetCostAnalysis(void* a){ fprintf(stderr,"REAL fn_Executable_GetCostAnalysis\n"); fflush(stderr); PJRT_Executable_GetCostAnalysis_Args* q=(PJRT_Executable_GetCostAnalysis_Args*)a; q->properties=NULL; q->num_properties=0; return NULL; }
static PJRT_Error* fn_Executable_OptimizedProgram(void* a){ fprintf(stderr,"REAL Executable_OptimizedProgram\n"); fflush(stderr); (void)a; return make_err("OptimizedProgram UNIMPLEMENTED"); }

static PJRT_Error* fn_LoadedExecutable_Destroy(void* a) {
  PJRT_LoadedExecutable_Destroy_Args* args = (PJRT_LoadedExecutable_Destroy_Args*)a;
  NpuExecutable* ex = (NpuExecutable*)args->executable;
  /* free pre-alloc'd activation kernel BOs so their IOVAs are released back to
   * the aperture -- otherwise they leak (close(fd) does NOT free mmap'd large
   * BO IOVAs) and a later matmul-graph's aperture-top requirement is violated. */
  if (ex) {
    if (ex->has_relu_bos) { for (int i=0;i<5;i++) npu_bo_free(&ex->relu_bos[i]); ex->has_relu_bos=0; }
    if (ex->has_tanh_bos) { for (int i=0;i<5;i++) npu_bo_free(&ex->tanh_bos[i]); ex->has_tanh_bos=0; }
    free(ex);
  }
  return NULL;
}
static PJRT_Error* fn_LoadedExecutable_GetExecutable(void* a){
  fprintf(stderr, "REAL LoadedExecutable_GetExecutable\n"); fflush(stderr);
  PJRT_LoadedExecutable_GetExecutable_Args* args=(PJRT_LoadedExecutable_GetExecutable_Args*)a;
  args->executable=(PJRT_Executable*)args->loaded_executable; return NULL;
}
static PJRT_Error* fn_LoadedExecutable_AddressableDevices(void* a){
  fprintf(stderr, "REAL LoadedExecutable_AddressableDevices\n"); fflush(stderr);
  PJRT_LoadedExecutable_AddressableDevices_Args* args=(PJRT_LoadedExecutable_AddressableDevices_Args*)a;
  args->addressable_devices=g_dev_arr; args->num_addressable_devices=1; return NULL;
}
static void dev_assign_deleter(PJRT_DeviceAssignmentSerialized* s){ (void)s; }
static PJRT_Error* fn_LoadedExecutable_GetDeviceAssignment(void* a){
  fprintf(stderr, "REAL LoadedExecutable_GetDeviceAssignment\n"); fflush(stderr);
  PJRT_LoadedExecutable_GetDeviceAssignment_Args* args=(PJRT_LoadedExecutable_GetDeviceAssignment_Args*)a;
  args->serialized_bytes=""; args->serialized_bytes_size=0;
  args->serialized_device_assignment=NULL; args->serialized_device_assignment_deleter=dev_assign_deleter;
  return NULL;
}
static PJRT_Error* fn_LoadedExecutable_AddressableDeviceLogicalIds(void* a){
  PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args* args=(PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args*)a;
  args->addressable_device_logical_ids=NULL; args->num_addressable_device_logical_ids=0; return NULL;
}

/* baked add regcmd+task templates (DMAs=0, patched at runtime) — generated */
#include "pjrt_npu_addtmpl.c"
/* ---- regcmd/task DMA patchers (port of Python _patch_h2/_patch_h1) ---- */
/* patch a DMA entry's val+tag (bytes 2..7), keeping reg (bytes 0..1) and cid */
static void patch_dma_entry(uint8_t* p, uint64_t dma, uint32_t cid) {
  uint16_t v = (uint16_t)(dma & 0xFFFF);
  uint32_t t = (cid << 16) | ((dma >> 16) & 0xFFFF);
  memcpy(p+2, &v, 2); memcpy(p+4, &t, 4);
}
/* General regcmd patcher. roles: 0=task,1=regcmd,2=scratch,3=in,4=out.
 * cid 0x1001/0x2001/0x0201 = DMA (repoint to matching cap range); reg 0x0010
 * (cid 0x0101) = chain pointer into regcmd range. */
static int patch_regcmd_tmpl(uint8_t* h2, int len,
        const uint64_t* cap, const uint64_t* my, const uint32_t* sz, int n_bo) {
  int n=0;
  for (int i=0;i<len/8;i++){
    uint8_t* p=h2+i*8;
    uint16_t reg,val; uint32_t tag;
    memcpy(&reg,p,2); memcpy(&val,p+2,2); memcpy(&tag,p+4,4);
    uint32_t cid=(tag>>16)&0xFFFF;
    uint64_t dma=(((uint64_t)(tag&0xFFFF))<<16)|val;
    if (cid==0x1001||cid==0x2001||cid==0x0201){
      for(int r=0;r<n_bo;r++) if(cap[r]<=dma && dma<cap[r]+sz[r]){ patch_dma_entry(p, my[r]+(dma-cap[r]), cid); n++; break; }
    } else if (reg==0x0010){
      if(cap[1]<=dma && dma<cap[1]+sz[1]){ uint64_t nd=my[1]+(dma-cap[1]); uint16_t v=(uint16_t)(nd&0xFFFF); uint32_t t=(0x0101<<16)|((nd>>16)&0xFFFF); memcpy(p+2,&v,2); memcpy(p+4,&t,4); n++; }
    }
  }
  return n;
}
/* task patcher: repoint regcmd_addr (8 bytes at i*40+32) to fresh regcmd BO */
static void patch_task_tmpl(uint8_t* h1, int len, uint64_t cap_rc, uint64_t my_rc, uint32_t rc_sz){
  int nt=len/40;
  for(int i=0;i<nt;i++){ uint8_t* p=h1+i*40+32; uint64_t ra; memcpy(&ra,p,8);
    if(cap_rc<=ra && ra<cap_rc+rc_sz){ uint64_t na=my_rc+(ra-cap_rc); memcpy(p,&na,8);} }
}

/* ── host-backed fresh-fd-per-op graph executor (supports matmul in-graph).
 * Each op runs on a FRESH fd (aperture top each time) so matmul's aperture-top
 * requirement is met and the large-IOVA leak doesn't accumulate across ops. All
 * args are read to host first; intermediate results flow through host buffers
 * mid[k]. Used when the graph contains a matmul (g_has_matmul); elementwise-only
 * graphs use the shared-fd executor (BO reuse, faster). */
static void npu_init_fd(int fd){
  npu_action(fd,0,0xFFFFFFFFu); npu_action(fd,1,0); npu_action(fd,19,0xFFFFFFEDu); npu_action(fd,1,0); npu_action(fd,18,0);
}
#define HKELEM 49152
#define HKTSZ  98304
static PJRT_Error* npu_exec_graph_host(void* a, NpuExecutable* ex, NpuClient* c, NpuBuffer** ins){
  PJRT_LoadedExecutable_Execute_Args* args=(PJRT_LoadedExecutable_Execute_Args*)a;
  PJRT_Buffer** outlist=args->output_lists[0];
  int na=ex->n_args; if(na>8) na=8;
  /* 1. read all args to host (mmaps valid while c->fd open), then free the input
   *    BOs (munmap+MEM_DESTROY) so their IOVAs are released BEFORE we close c->fd
   *    -- otherwise the aperture top stays occupied and the matmul fd lands below
   *    the reach window. JAX's later Buffer_Destroy on these is a no-op (handle=0). */
  uint8_t* hin[8]; int64_t hin_elems[8];
  for(int i=0;i<8;i++){ hin[i]=NULL; hin_elems[i]=0; }
  for(int i=0;i<na;i++){
    int64_t e=1; for(int d=0;d<ins[i]->ndims;d++) e*=ins[i]->dims[d];
    hin_elems[i]=e; hin[i]=(uint8_t*)malloc((size_t)e*2);
    if(!hin[i]){ for(int j=0;j<i;j++) free(hin[j]); return make_err("graph_host: alloc hin failed"); }
    npu_sync_bo(c->fd, ins[i]->bo.obj_addr, ins[i]->bo.size, 2);
    if(ins[i]->bo.mm) memcpy(hin[i], ins[i]->bo.mm, (size_t)e*2);
  }
  for(int i=0;i<na;i++) npu_bo_free(&ins[i]->bo);
  /* also free any pre-alloc'd activation BOs on this executable (a matmul-graph
   * normally skips pre-alloc, but free them if present so the aperture is clean
   * before close(c->fd)). */
  if(ex->has_relu_bos){ for(int i=0;i<5;i++) npu_bo_free(&ex->relu_bos[i]); ex->has_relu_bos=0; }
  if(ex->has_tanh_bos){ for(int i=0;i<5;i++) npu_bo_free(&ex->tanh_bos[i]); ex->has_tanh_bos=0; }
  /* 2. per-op output element counts + shapes + matmul template match */
  int64_t out_elems[16]; int out_ndims[16]; int64_t out_dims[16][8];
  const MMTemplate* op_tpl[16]; for(int k=0;k<16;k++){ op_tpl[k]=NULL; out_elems[k]=0; out_ndims[k]=0; }
  for(int k=0;k<ex->n_ops;k++){
    int t=ex->g_ops[k];
    if(t==6){ /* matmul: operand0=X(3D), operand1=W(2D) */
      int64_t Xd[8], Wd[8]; int Xn=0, Wn=0;
      if(ex->g_opnd_isarg[k][0]){ Xn=ins[ex->g_opnd_idx[k][0]]->ndims; for(int d=0;d<Xn;d++) Xd[d]=ins[ex->g_opnd_idx[k][0]]->dims[d]; }
      else { Xn=out_ndims[ex->g_opnd_idx[k][0]]; for(int d=0;d<Xn;d++) Xd[d]=out_dims[ex->g_opnd_idx[k][0]][d]; }
      if(ex->g_opnd_isarg[k][1]){ Wn=ins[ex->g_opnd_idx[k][1]]->ndims; for(int d=0;d<Wn;d++) Wd[d]=ins[ex->g_opnd_idx[k][1]]->dims[d]; }
      else { Wn=out_ndims[ex->g_opnd_idx[k][1]]; for(int d=0;d<Wn;d++) Wd[d]=out_dims[ex->g_opnd_idx[k][1]][d]; }
      const MMTemplate* tpl=NULL;
      for(int i=0;i<MM_NSHAPES;i++){ const MMTemplate* tt=&mm_templates[i];
        if((int)Xd[0]==tt->x0&&(int)Xd[1]==tt->x1&&(int)Xd[2]==tt->x2&&(int)Wd[0]==tt->w0&&(int)Wd[1]==tt->w1){ tpl=tt; break; } }
      if(!tpl){ for(int i=0;i<na;i++) free(hin[i]); return make_err("graph_host: no matmul template for X/W shape"); }
      op_tpl[k]=tpl; out_ndims[k]=3; out_dims[k][0]=tpl->z0; out_dims[k][1]=tpl->z1; out_dims[k][2]=tpl->z2;
      out_elems[k]=(int64_t)tpl->z0*tpl->z1*tpl->z2;
    } else { /* elementwise/relu/tanh: shape-preserving from the non-const data
           * operand. A binary op may have a const operand 0 (e.g. 0.044715*x ->
           * multiply(const, x)); the const operand's g_opnd_idx is -1, so we must
           * pick the non-const operand (its g_opnd_idx >= 0). */
      int srcj=-1;
      for(int j=0;j<2;j++){ if(!ex->g_opnd_isconst[k][j]){ srcj=j; break; } }
      if(srcj<0) srcj=0;
      int srcop=ex->g_opnd_idx[k][srcj];
      if(ex->g_opnd_isarg[k][srcj]){ out_ndims[k]=ins[srcop]->ndims; for(int d=0;d<out_ndims[k];d++) out_dims[k][d]=ins[srcop]->dims[d]; out_elems[k]=hin_elems[srcop]; }
      else { out_ndims[k]=out_ndims[srcop]; for(int d=0;d<out_ndims[k];d++) out_dims[k][d]=out_dims[srcop][d]; out_elems[k]=out_elems[srcop]; }
    }
  }
  /* 3. alloc mid[k] (intermediate host buffers) */
  uint8_t* mid[16]; for(int k=0;k<16;k++) mid[k]=NULL;
  for(int k=0;k<ex->n_ops;k++){ mid[k]=(uint8_t*)malloc((size_t)out_elems[k]*2); if(!mid[k]){ for(int i=0;i<na;i++)free(hin[i]); for(int i=0;i<16;i++)free(mid[i]); return make_err("graph_host: alloc mid failed"); } }
  /* 4. close the client fd (inputs staged to host; their BOs freed above) */
  close(c->fd); c->fd=-1;
  static const uint16_t alu_e23v[4]={0x02c0,0x02c0,0x03c4,0x03c0};
  static const uint32_t alu_e23t[4]={(0x1001<<16)|0x1082,(0x1001<<16)|0x1084,(0x1001<<16)|0x1080,(0x1001<<16)|0x1083};
  static const uint16_t alu_e28t[4]={0x0001,0x0001,0x0001,0x0000};
  static const uint16_t alu_e63v[4]={0x7849,0x7849,0x7849,0x7841};
  int ok=1;
  /* 5. per op: fresh fd at aperture top, run, read mid[k], close fd */
  for(int k=0;k<ex->n_ops && ok;k++){
    int t=ex->g_ops[k];
    int mfd=open(NPU_DEV,O_RDWR); if(mfd<0){ ok=0; break; } npu_init_fd(mfd);
    if(t==6){ /* matmul */
      const MMTemplate* tpl=op_tpl[k];
      uint8_t* Xh = ex->g_opnd_isarg[k][0] ? hin[ex->g_opnd_idx[k][0]] : mid[ex->g_opnd_idx[k][0]];
      uint8_t* Wh = ex->g_opnd_isarg[k][1] ? hin[ex->g_opnd_idx[k][1]] : mid[ex->g_opnd_idx[k][1]];
      struct npu_bo mb[6];
      uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
      uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
      int ce=0; for(int i=0;i<6;i++){ if(npu_create_bo(mfd,msz[i],mfl[i],&mb[i])<0){ ce=1; break; } }
      if(ce){ for(int i=0;i<6;i++) npu_bo_free(&mb[i]); close(mfd); ok=0; break; }
      uint64_t cap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
      uint64_t my[6]; for(int i=0;i<6;i++) my[i]=mb[i].dma_addr;
      uint32_t sz32[6]; for(int i=0;i<6;i++) sz32[i]=(uint32_t)msz[i];
      if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch);
      if(mb[3].mm) memcpy(mb[3].mm, Xh, tpl->sz_in);
      if(mb[4].mm) memcpy(mb[4].mm, Wh, tpl->sz_w);
      if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out);
      if(mb[1].mm) memcpy(mb[1].mm, tpl->regcmd_tmpl, tpl->sz_regcmd<mb[1].size?tpl->sz_regcmd:mb[1].size);
      if(mb[0].mm) memcpy(mb[0].mm, tpl->task_tmpl, tpl->sz_task<mb[0].size?tpl->sz_task:mb[0].size);
      patch_regcmd_tmpl(mb[1].mm, tpl->sz_regcmd, cap, my, sz32, 6);
      patch_task_tmpl(mb[0].mm, tpl->sz_task, cap[1], my[1], tpl->sz_regcmd);
      for(int i=0;i<6;i++) npu_sync_bo(mfd, mb[i].obj_addr, mb[i].size, 1);
      uint8_t sub[104]; memset(sub,0,104);
      *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks;
      *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
      for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2]; *(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1]; }
      if(ioctl(mfd,IOCTL_SUBMIT,sub)<0){ for(int i=0;i<6;i++) npu_bo_free(&mb[i]); close(mfd); ok=0; break; }
      npu_sync_bo(mfd, mb[5].obj_addr, mb[5].size, 2);
      if(mb[5].mm) memcpy(mid[k], mb[5].mm, (size_t)out_elems[k]*2);
      for(int i=0;i<6;i++) npu_bo_free(&mb[i]);
      close(mfd);
    } else if(t<4){ /* binary elementwise, tiled */
      struct npu_bo cb[6];
      uint64_t csz[6]={4096,8192,0x60000,HKTSZ,HKTSZ,HKTSZ};
      uint32_t cfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
      int ce=0; for(int i=0;i<6;i++){ if(npu_create_bo(mfd,csz[i],cfl[i],&cb[i])<0){ ce=1; break; } }
      if(ce){ for(int i=0;i<6;i++) npu_bo_free(&cb[i]); close(mfd); ok=0; break; }
      uint64_t dm[6]; for(int i=0;i<6;i++) dm[i]=cb[i].dma_addr;
      if(cb[2].mm) memset(cb[2].mm,0,cb[2].size);
      if(cb[1].mm) memcpy(cb[1].mm,g_regcmd_tmpl,sizeof(g_regcmd_tmpl)<cb[1].size?sizeof(g_regcmd_tmpl):cb[1].size);
      if(cb[0].mm) memcpy(cb[0].mm,g_task_tmpl,sizeof(g_task_tmpl)<cb[0].size?sizeof(g_task_tmpl):cb[0].size);
      uint64_t rcb=dm[1]+0xC0;
      static const int blk_off[4]={0x0,0x280,0x500,0x780};
      #define HCPDMA(bo_off,entry,dma) do{ uint8_t* p=(uint8_t*)cb[1].mm+(bo_off)+(entry)*8; uint16_t v=(uint16_t)((dma)&0xFFFF); uint32_t tt=(0x2001<<16)|(((dma)>>16)&0xFFFF); if((entry)==5){tt=(0x1001<<16)|(((dma)>>16)&0xFFFF);} memcpy(p+2,&v,2); memcpy(p+4,&tt,4);}while(0)
      HCPDMA(0xC0+blk_off[0],5,dm[3]); HCPDMA(0xC0+blk_off[0],55,dm[2]+0x18000);
      HCPDMA(0xC0+blk_off[1],5,dm[4]); HCPDMA(0xC0+blk_off[1],55,dm[2]+0x0);
      HCPDMA(0xC0+blk_off[2],5,dm[2]+0x30000); HCPDMA(0xC0+blk_off[2],55,dm[3]); HCPDMA(0xC0+blk_off[2],61,dm[4]);
      HCPDMA(0xC0+blk_off[3],5,dm[5]); HCPDMA(0xC0+blk_off[3],55,dm[2]+0x30000);
      #define HCPCHAIN(bo_off,nextoff) do{ uint8_t* p=(uint8_t*)cb[1].mm+(bo_off)+69*8; uint64_t nd=rcb+(nextoff); uint16_t rv=0x0010,vv=(uint16_t)(nd&0xFFFF); uint32_t tt=(0x0101<<16)|((nd>>16)&0xFFFF); memcpy(p+0,&rv,2); memcpy(p+2,&vv,2); memcpy(p+4,&tt,4);}while(0)
      HCPCHAIN(0xC0+blk_off[0],blk_off[1]); HCPCHAIN(0xC0+blk_off[1],blk_off[2]); HCPCHAIN(0xC0+blk_off[2],blk_off[3]);
      for(int ti=0;ti<12;ti++){ uint8_t* p=(uint8_t*)cb[0].mm+ti*40+32; uint64_t aa=rcb+blk_off[ti%4]; memcpy(p,&aa,8); }
      #undef HCPDMA
      #undef HCPCHAIN
      for(int i=0;i<6;i++) npu_sync_bo(mfd,cb[i].obj_addr,cb[i].size,3);
      for(int i=0;i<6;i++) if(i!=2) npu_sync_bo(mfd,cb[i].obj_addr,cb[i].size,1);
      int64_t nc=(out_elems[k]+HKELEM-1)/HKELEM;
      for(int64_t ci=0;ci<nc && ok;ci++){
        int64_t s=ci*HKELEM, ne=(s+HKELEM>out_elems[k])?(out_elems[k]-s):HKELEM; int64_t bytes=ne*2;
        if(cb[2].mm){ memset((uint8_t*)cb[2].mm+0x18000,0,HKTSZ); memset((uint8_t*)cb[2].mm+0x0,0,HKTSZ);
          for(int j=0;j<2;j++){ uint8_t* dst=(j==0)?(uint8_t*)cb[2].mm+0x18000:(uint8_t*)cb[2].mm+0x0;
            if(ex->g_opnd_isconst[k][j]){ uint16_t cv=ex->g_const_val[k][j]; uint8_t lo=cv&0xff,hi=cv>>8; for(int64_t e=0;e<HKTSZ/2;e++){dst[e*2]=lo;dst[e*2+1]=hi;} }
            else if(ex->g_opnd_isarg[k][j]){ memcpy(dst, hin[ex->g_opnd_idx[k][j]]+s*2, bytes); }
            else memcpy(dst, mid[ex->g_opnd_idx[k][j]]+s*2, bytes);
          }
        }
        npu_sync_bo(mfd,cb[2].obj_addr,cb[2].size,1);
        { uint8_t* base=(uint8_t*)cb[1].mm+0xC0+0x500; int op=t;
          #define HCSETALU(e,val,tag) do{ uint8_t* pp=base+(e)*8; uint16_t _v=(val); uint32_t _t=(tag); memcpy(pp+2,&_v,2); memcpy(pp+4,&_t,4);}while(0)
          HCSETALU(23,alu_e23v[op],alu_e23t[op]); HCSETALU(28,0x0001,(0x1001<<16)|alu_e28t[op]); HCSETALU(63,alu_e63v[op],(0x2001<<16)|0x0001);
          #undef HCSETALU
        }
        npu_sync_bo(mfd,cb[1].obj_addr,cb[1].size,1);
        uint8_t sub[104]; memset(sub,0,104);
        *(uint32_t*)(sub+0)=5; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=0; *(uint32_t*)(sub+12)=4;
        *(uint32_t*)(sub+16)=4; *(uint64_t*)(sub+24)=cb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
        *(uint32_t*)(sub+64)=0; *(uint32_t*)(sub+68)=4;
        if(ioctl(mfd,IOCTL_SUBMIT,sub)<0){ ok=0; break; }
        npu_sync_bo(mfd,cb[5].obj_addr,cb[5].size,2);
        if(cb[5].mm) memcpy(mid[k]+s*2, cb[5].mm, bytes);
      }
      for(int i=0;i<6;i++) npu_bo_free(&cb[i]);
      close(mfd);
    } else if(t==4){ /* relu, tiled */
      struct npu_bo rb[5];
      uint64_t rsz[5]={RELU_SZ_TASK,RELU_SZ_REGCMD,RELU_SZ_SCRATCH,RELU_SZ_IN,RELU_SZ_OUT};
      uint32_t rfl[5]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
      int ce=0; for(int i=0;i<5;i++){ if(npu_create_bo(mfd,rsz[i],rfl[i],&rb[i])<0){ ce=1; break; } }
      if(ce){ for(int i=0;i<5;i++) npu_bo_free(&rb[i]); close(mfd); ok=0; break; }
      uint32_t rsz32[5]={RELU_SZ_TASK,RELU_SZ_REGCMD,RELU_SZ_SCRATCH,RELU_SZ_IN,RELU_SZ_OUT};
      uint64_t rcap[5]={RELU_CAP_TASK,RELU_CAP_REGCMD,RELU_CAP_SCRATCH,RELU_CAP_IN,RELU_CAP_OUT};
      uint64_t rmy[5]; for(int i=0;i<5;i++) rmy[i]=rb[i].dma_addr;
      if(rb[2].mm) memset(rb[2].mm,0,rb[2].size);
      npu_sync_bo(mfd,rb[2].obj_addr,rb[2].size,1);
      if(rb[1].mm) memcpy(rb[1].mm,g_relu_regcmd,RELU_REGCMD_SZ<rb[1].size?RELU_REGCMD_SZ:rb[1].size);
      if(rb[0].mm) memcpy(rb[0].mm,g_relu_task,RELU_TASK_SZ<rb[0].size?RELU_TASK_SZ:rb[0].size);
      patch_regcmd_tmpl(rb[1].mm,RELU_REGCMD_SZ,rcap,rmy,rsz32,5);
      patch_task_tmpl(rb[0].mm,RELU_TASK_SZ,rcap[1],rmy[1],RELU_SZ_REGCMD);
      npu_sync_bo(mfd,rb[1].obj_addr,rb[1].size,1); npu_sync_bo(mfd,rb[0].obj_addr,rb[0].size,1);
      int64_t nc=(out_elems[k]+HKELEM-1)/HKELEM;
      for(int64_t ci=0;ci<nc && ok;ci++){
        int64_t s=ci*HKELEM, ne=(s+HKELEM>out_elems[k])?(out_elems[k]-s):HKELEM; int64_t bytes=ne*2;
        if(rb[3].mm){ memset(rb[3].mm,0,RELU_SZ_IN);
          if(ex->g_opnd_isarg[k][0]) memcpy(rb[3].mm, hin[ex->g_opnd_idx[k][0]]+s*2, bytes);
          else memcpy(rb[3].mm, mid[ex->g_opnd_idx[k][0]]+s*2, bytes);
        }
        if(rb[4].mm) memset(rb[4].mm,0,RELU_SZ_OUT);
        npu_sync_bo(mfd,rb[3].obj_addr,rb[3].size,1); npu_sync_bo(mfd,rb[4].obj_addr,rb[4].size,1);
        uint8_t sub[104]; memset(sub,0,104);
        *(uint32_t*)(sub+0)=RELU_FLAGS; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=RELU_TASK_START; *(uint32_t*)(sub+12)=RELU_N_TASKS;
        *(uint32_t*)(sub+16)=RELU_N_TASKS; *(uint64_t*)(sub+24)=rb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
        for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=RELU_SUBCORE[si*2]; *(uint32_t*)(sub+64+si*8+4)=RELU_SUBCORE[si*2+1]; }
        if(ioctl(mfd,IOCTL_SUBMIT,sub)<0){ ok=0; break; }
        npu_sync_bo(mfd,rb[4].obj_addr,rb[4].size,2);
        if(rb[4].mm) memcpy(mid[k]+s*2, rb[4].mm, bytes);
      }
      for(int i=0;i<5;i++) npu_bo_free(&rb[i]);
      close(mfd);
    } else if(t==5){ /* tanh, tiled, 3 submits */
      struct npu_bo tb[5];
      uint64_t tsz[5]={TANH_SZ_TASK,TANH_SZ_REGCMD,TANH_SZ_SCRATCH,TANH_SZ_IN,TANH_SZ_OUT};
      uint32_t tfl[5]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
      int ce=0; for(int i=0;i<5;i++){ if(npu_create_bo(mfd,tsz[i],tfl[i],&tb[i])<0){ ce=1; break; } }
      if(ce){ for(int i=0;i<5;i++) npu_bo_free(&tb[i]); close(mfd); ok=0; break; }
      static const unsigned char* const th2[3]={g_tanh_regcmd0,g_tanh_regcmd1,g_tanh_regcmd2};
      static const unsigned char* const th1[3]={g_tanh_task0,g_tanh_task1,g_tanh_task2};
      static const uint32_t tstart[3]={TANH_SUB0_START,TANH_SUB1_START,TANH_SUB2_START};
      static const uint32_t tntask[3]={TANH_SUB0_NTASK,TANH_SUB1_NTASK,TANH_SUB2_NTASK};
      static const uint32_t tflags[3]={TANH_SUB0_FLAGS,TANH_SUB1_FLAGS,TANH_SUB2_FLAGS};
      static const uint32_t* const tsc[3]={TANH_SUB0_SUBCORE,TANH_SUB1_SUBCORE,TANH_SUB2_SUBCORE};
      uint32_t tsz32[5]={TANH_SZ_TASK,TANH_SZ_REGCMD,TANH_SZ_SCRATCH,TANH_SZ_IN,TANH_SZ_OUT};
      uint64_t tcap[5]={TANH_CAP_TASK,TANH_CAP_REGCMD,TANH_CAP_SCRATCH,TANH_CAP_IN,TANH_CAP_OUT};
      uint64_t tmy[5]; for(int i=0;i<5;i++) tmy[i]=tb[i].dma_addr;
      int64_t nc=(out_elems[k]+HKELEM-1)/HKELEM;
      for(int64_t ci=0;ci<nc && ok;ci++){
        int64_t s=ci*HKELEM, ne=(s+HKELEM>out_elems[k])?(out_elems[k]-s):HKELEM; int64_t bytes=ne*2;
        if(tb[2].mm) memset(tb[2].mm,0,tb[2].size);
        npu_sync_bo(mfd,tb[2].obj_addr,tb[2].size,1);
        if(tb[3].mm){ memset(tb[3].mm,0,TANH_SZ_IN);
          if(ex->g_opnd_isarg[k][0]) memcpy(tb[3].mm, hin[ex->g_opnd_idx[k][0]]+s*2, bytes);
          else memcpy(tb[3].mm, mid[ex->g_opnd_idx[k][0]]+s*2, bytes);
        }
        if(tb[4].mm) memset(tb[4].mm,0,TANH_SZ_OUT);
        npu_sync_bo(mfd,tb[3].obj_addr,tb[3].size,1); npu_sync_bo(mfd,tb[4].obj_addr,tb[4].size,1);
        for(int sd=0;sd<TANH_NSUBMITS && ok;sd++){
          if(tb[1].mm) memcpy(tb[1].mm,th2[sd],TANH_SZ_REGCMD);
          if(tb[0].mm) memcpy(tb[0].mm,th1[sd],TANH_SZ_TASK);
          patch_regcmd_tmpl(tb[1].mm,TANH_SZ_REGCMD,tcap,tmy,tsz32,5);
          patch_task_tmpl(tb[0].mm,TANH_SZ_TASK,tcap[1],tmy[1],TANH_SZ_REGCMD);
          npu_sync_bo(mfd,tb[1].obj_addr,tb[1].size,1); npu_sync_bo(mfd,tb[0].obj_addr,tb[0].size,1);
          uint8_t sub[104]; memset(sub,0,104);
          *(uint32_t*)(sub+0)=tflags[sd]; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tstart[sd]; *(uint32_t*)(sub+12)=tntask[sd];
          *(uint32_t*)(sub+16)=tntask[sd]; *(uint64_t*)(sub+24)=tb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
          for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=tsc[sd][si*2]; *(uint32_t*)(sub+64+si*8+4)=tsc[sd][si*2+1]; }
          if(ioctl(mfd,IOCTL_SUBMIT,sub)<0){ ok=0; break; }
        }
        if(!ok) break;
        npu_sync_bo(mfd,tb[4].obj_addr,tb[4].size,2);
        if(tb[4].mm) memcpy(mid[k]+s*2, tb[4].mm, bytes);
      }
      for(int i=0;i<5;i++) npu_bo_free(&tb[i]);
      close(mfd);
    } else { close(mfd); ok=0; break; }
  }
  /* 6. reopen c->fd (fresh domain), create output buffer, write mid[g_out_op] */
  c->fd=open(NPU_DEV,O_RDWR); npu_init_fd(c->fd);
  for(int i=0;i<na;i++) free(hin[i]);
  if(!ok){ for(int k=0;k<16;k++) free(mid[k]); return make_err("graph_host: op failed"); }
  int o=ex->g_out_op;
  uint64_t out_sz=(uint64_t)out_elems[o]*2;
  NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
  ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=out_ndims[o];
  for(int d=0;d<out_ndims[o];d++) ob->dims[d]=out_dims[o][d];
  if(npu_create_bo(c->fd,out_sz,CREATE_FLAGS_BUF,&ob->bo)<0){ for(int k=0;k<16;k++)free(mid[k]); free(ob); return make_err("graph_host: out create_bo failed"); }
  if(ob->bo.mm) memcpy(ob->bo.mm, mid[o], out_sz);
  npu_sync_bo(c->fd,ob->bo.obj_addr,ob->bo.size,1);
  for(int k=0;k<16;k++) free(mid[k]);
  outlist[0]=(PJRT_Buffer*)ob;
  if(args->device_complete_events) args->device_complete_events[0]=NULL;
  return NULL;
}
#undef HKELEM
#undef HKTSZ


static PJRT_Error* fn_LoadedExecutable_Execute(void* a) {
  fprintf(stderr, "REAL LoadedExecutable_Execute\n"); fflush(stderr);
  PJRT_LoadedExecutable_Execute_Args* args = (PJRT_LoadedExecutable_Execute_Args*)a;
  NpuExecutable* ex = (NpuExecutable*)args->executable;
  NpuClient* c = ex->cli;
  /* inputs: args->argument_lists[0][0], [0][1] (2 inputs, 1 device) */
  NpuBuffer* in0 = (NpuBuffer*)args->argument_lists[0][0];
  NpuBuffer* in1 = (NpuBuffer*)args->argument_lists[0][1];
  PJRT_Buffer** outlist = args->output_lists[0];
  /* host-backed fresh-fd-per-op graph executor: used when the graph contains a
   * matmul (which needs the aperture top -- a fresh fd per op). Elementwise-only
   * graphs use the shared-fd executor below (BO reuse, faster). */
  if (ex->is_graph && ex->g_has_matmul) {
    NpuBuffer* gins[8];
    for (int i=0;i<ex->n_args && i<8;i++) gins[i]=(NpuBuffer*)args->argument_lists[0][i];
    return npu_exec_graph_host(a, ex, c, gins);
  }
  /* element count from in0 */
  int64_t n = 1; for (int i=0;i<in0->ndims;i++) n*=in0->dims[i];
  uint64_t io_sz = (uint64_t)n*2;        /* full output bytes (fp16) */
  /* kernel pinned to (1,64,768)=49152 elems; tile into chunks reusing fixed BOs */
  #define KELEM 49152
  #define KTSZ  98304
  int64_t n_chunks = (n + KELEM - 1) / KELEM;
  /* ── relu: captured template + tiled execution (1 input, 1 submit) ─── */
  if (ex->op == 4) {  /* relu: captured template + tiled execution (pre-allocated BOs) */
    struct npu_bo* rb = ex->relu_bos;   /* pre-allocated at Client_Compile (top of IOVA pool) */
    uint32_t rsz32[5] = {RELU_SZ_TASK, RELU_SZ_REGCMD, RELU_SZ_SCRATCH, RELU_SZ_IN, RELU_SZ_OUT};
    uint64_t cap[5]={RELU_CAP_TASK,RELU_CAP_REGCMD,RELU_CAP_SCRATCH,RELU_CAP_IN,RELU_CAP_OUT};
    uint64_t my[5]={rb[0].dma_addr,rb[1].dma_addr,rb[2].dma_addr,rb[3].dma_addr,rb[4].dma_addr};
    if (rb[2].mm) memset(rb[2].mm,0,rb[2].size);   /* zeroed scratch suffices */
    npu_sync_bo(c->fd, rb[2].obj_addr, rb[2].size, 1);
    NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
    ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=in0->ndims;
    for(int i=0;i<in0->ndims;i++) ob->dims[i]=in0->dims[i];
    if (npu_create_bo(c->fd, io_sz, CREATE_FLAGS_BUF, &ob->bo)<0){ free(ob); return make_err("relu out create_bo failed"); }
    int ok=1;
    for (int64_t ci=0; ci<n_chunks; ci++){
      int64_t s=ci*KELEM, ne=(s+KELEM>n)?(n-s):KELEM;
      if (rb[3].mm){ memset(rb[3].mm,0,RELU_SZ_IN); if(in0->bo.mm) memcpy(rb[3].mm,(uint8_t*)in0->bo.mm+s*2, ne*2); }
      if (rb[4].mm) memset(rb[4].mm,0,RELU_SZ_OUT);
      npu_sync_bo(c->fd, rb[3].obj_addr, rb[3].size, 1);
      npu_sync_bo(c->fd, rb[4].obj_addr, rb[4].size, 1);
      if (rb[1].mm) memcpy(rb[1].mm, g_relu_regcmd, RELU_REGCMD_SZ<rb[1].size?RELU_REGCMD_SZ:rb[1].size);
      if (rb[0].mm) memcpy(rb[0].mm, g_relu_task, RELU_TASK_SZ<rb[0].size?RELU_TASK_SZ:rb[0].size);
      patch_regcmd_tmpl(rb[1].mm, RELU_REGCMD_SZ, cap, my, rsz32, 5);
      patch_task_tmpl(rb[0].mm, RELU_TASK_SZ, cap[1], my[1], RELU_SZ_REGCMD);
      npu_sync_bo(c->fd, rb[1].obj_addr, rb[1].size, 1);
      npu_sync_bo(c->fd, rb[0].obj_addr, rb[0].size, 1);
      uint8_t sub[104]; memset(sub,0,104);
      *(uint32_t*)(sub+0)=RELU_FLAGS; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=RELU_TASK_START; *(uint32_t*)(sub+12)=RELU_N_TASKS;
      *(uint32_t*)(sub+16)=RELU_N_TASKS; *(uint64_t*)(sub+24)=rb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
      for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=RELU_SUBCORE[si*2]; *(uint32_t*)(sub+64+si*8+4)=RELU_SUBCORE[si*2+1]; }
      if (ioctl(c->fd, IOCTL_SUBMIT, sub)<0){ ok=0; break; }
      npu_sync_bo(c->fd, rb[4].obj_addr, rb[4].size, 2);
      if (ob->bo.mm && rb[4].mm) memcpy((uint8_t*)ob->bo.mm+s*2, rb[4].mm, ne*2);
    }
    npu_sync_bo(c->fd, ob->bo.obj_addr, ob->bo.size, 1);
    outlist[0]=(PJRT_Buffer*)ob;
    if (args->device_complete_events) args->device_complete_events[0]=NULL;
    if(!ok) return make_err("relu submit failed during tiling");
    return NULL;
  }
  if (ex->op == 5) {  /* tanh: captured template + tiled execution (3 submits) */
    struct npu_bo* rb = ex->tanh_bos;   /* pre-allocated at Client_Compile (top of IOVA pool) */
    uint32_t tsz32[5] = {TANH_SZ_TASK, TANH_SZ_REGCMD, TANH_SZ_SCRATCH, TANH_SZ_IN, TANH_SZ_OUT};
    uint64_t cap[5]={TANH_CAP_TASK,TANH_CAP_REGCMD,TANH_CAP_SCRATCH,TANH_CAP_IN,TANH_CAP_OUT};
    uint64_t my[5]={rb[0].dma_addr,rb[1].dma_addr,rb[2].dma_addr,rb[3].dma_addr,rb[4].dma_addr};
    static const unsigned char* const th2[3]={g_tanh_regcmd0,g_tanh_regcmd1,g_tanh_regcmd2};
    static const unsigned char* const th1[3]={g_tanh_task0,g_tanh_task1,g_tanh_task2};
    static const uint32_t tstart[3]={TANH_SUB0_START,TANH_SUB1_START,TANH_SUB2_START};
    static const uint32_t tntask[3]={TANH_SUB0_NTASK,TANH_SUB1_NTASK,TANH_SUB2_NTASK};
    static const uint32_t tflags[3]={TANH_SUB0_FLAGS,TANH_SUB1_FLAGS,TANH_SUB2_FLAGS};
    static const uint32_t* const tsc[3]={TANH_SUB0_SUBCORE,TANH_SUB1_SUBCORE,TANH_SUB2_SUBCORE};
    NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
    ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=in0->ndims;
    for(int i=0;i<in0->ndims;i++) ob->dims[i]=in0->dims[i];
    if (npu_create_bo(c->fd, io_sz, CREATE_FLAGS_BUF, &ob->bo)<0){ free(ob); return make_err("tanh out create_bo failed"); }
    int ok=1;
    for (int64_t ci=0; ci<n_chunks; ci++){
      int64_t s=ci*KELEM, ne=(s+KELEM>n)?(n-s):KELEM;
      if (rb[2].mm) memset(rb[2].mm,0,rb[2].size);   /* scratch carries state across the 3 submits */
      npu_sync_bo(c->fd, rb[2].obj_addr, rb[2].size, 1);
      if (rb[3].mm){ memset(rb[3].mm,0,TANH_SZ_IN); if(in0->bo.mm) memcpy(rb[3].mm,(uint8_t*)in0->bo.mm+s*2, ne*2); }
      if (rb[4].mm) memset(rb[4].mm,0,TANH_SZ_OUT);
      npu_sync_bo(c->fd, rb[3].obj_addr, rb[3].size, 1);
      npu_sync_bo(c->fd, rb[4].obj_addr, rb[4].size, 1);
      for (int sd=0; sd<TANH_NSUBMITS; sd++){
        if (rb[1].mm) memcpy(rb[1].mm, th2[sd], TANH_SZ_REGCMD);
        if (rb[0].mm) memcpy(rb[0].mm, th1[sd], TANH_SZ_TASK);
        patch_regcmd_tmpl(rb[1].mm, TANH_SZ_REGCMD, cap, my, tsz32, 5);
        patch_task_tmpl(rb[0].mm, TANH_SZ_TASK, cap[1], my[1], TANH_SZ_REGCMD);
        npu_sync_bo(c->fd, rb[1].obj_addr, rb[1].size, 1);
        npu_sync_bo(c->fd, rb[0].obj_addr, rb[0].size, 1);
        uint8_t sub[104]; memset(sub,0,104);
        *(uint32_t*)(sub+0)=tflags[sd]; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tstart[sd]; *(uint32_t*)(sub+12)=tntask[sd];
        *(uint32_t*)(sub+16)=tntask[sd]; *(uint64_t*)(sub+24)=rb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
        for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=tsc[sd][si*2]; *(uint32_t*)(sub+64+si*8+4)=tsc[sd][si*2+1]; }
        if (ioctl(c->fd, IOCTL_SUBMIT, sub)<0){ ok=0; break; }
      }
      if(!ok) break;
      npu_sync_bo(c->fd, rb[4].obj_addr, rb[4].size, 2);
      if (ob->bo.mm && rb[4].mm) memcpy((uint8_t*)ob->bo.mm+s*2, rb[4].mm, ne*2);
    }
    npu_sync_bo(c->fd, ob->bo.obj_addr, ob->bo.size, 1);
    outlist[0]=(PJRT_Buffer*)ob;
    if (args->device_complete_events) args->device_complete_events[0]=NULL;
    if(!ok) return make_err("tanh submit failed during tiling");
    return NULL;
  }
  /* ── matmul: aperture-top-pinned template. The matmul BOs form a contiguous
   * block ending exactly at the 4GB boundary (0x100000000); the kernel only
   * runs when they occupy the very TOP of the IOMMU aperture. Any BO above
   * them (even 4KB) pushes io2 below 0xfed64000 -> submit ETIMEDOUT. So we
   * CLOSE the client fd (which releases its whole IOMMU domain -- unlike
   * freeing individual BOs, which leak large IOVAs), read X/W to host first
   * (their mmaps keep the GEM memory alive after the fd closes), open a fresh
   * matmul fd at the aperture top, run, read Z to host, then REOPEN the client
   * fd for the output buffer. Shape-pinned: X[1,64,768] @ W[768,3072] ->
   * Z[1,64,3072]; both inputs runtime (W NOT baked); zeroed scratch suffices. */
  if (ex->op == 6) {
    if (!in0 || !in1) return make_err("matmul needs 2 inputs (X, W)");
    NpuBuffer* Xb = in0; NpuBuffer* Wb = in1;
    if (Xb->ndims!=3 || Wb->ndims!=2) return make_err("matmul: X must be 3D, W 2D");
    int M=Xb->dims[1], K=Xb->dims[2], Nv=Wb->dims[1];
    int match = (s_wc_mfd >= 0 && s_wc_shape[0]==M && s_wc_shape[1]==K && s_wc_shape[2]==Nv);
    if (match) {
      /* FAST PATH: no close(c->fd), no host staging. Fingerprint straight from
       * the device buffer mmap; load W from mmap on first sight, else hit; run
       * from the X mmap. Device buffers stay alive -> jax.device_put-once works. */
      npu_sync_bo(c->fd, Wb->bo.obj_addr, Wb->bo.size, 2);
      uint64_t fp0=0, fp1=0;
      if (Wb->bo.mm && Wb->bo.size>=16){ memcpy(&fp0, Wb->bo.mm, 8); memcpy(&fp1, Wb->bo.mm+8, 8); }
      int w_idx=-1;
      for (int i=0;i<c->mm_nw;i++) if (c->mm_wsrc[i]==fp0 && c->mm_wsrc2[i]==fp1){ w_idx=i; break; }
      if (w_idx<0){
        w_idx=c->mm_nw++;
        if (w_idx>=MM_NWPRE) return make_err("matmul: W cache full");
        npu_mm_cache_load(w_idx, Wb->bo.mm);
        c->mm_wsrc[w_idx]=fp0; c->mm_wsrc2[w_idx]=fp1;
      }
      npu_sync_bo(c->fd, Xb->bo.obj_addr, Xb->bo.size, 2);
      uint64_t out_sz=(uint64_t)s_wc_tpl->z0*s_wc_tpl->z1*s_wc_tpl->z2*2;
      uint8_t* Zh=(uint8_t*)malloc(out_sz);
      if (!Zh) return make_err("matmul: Z alloc failed");
      int rr=npu_mm_cache_run(w_idx, Xb->bo.mm, Zh);
      if (rr){ free(Zh); return make_err("matmul: cache_run failed"); }
      NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
      ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=3;
      ob->dims[0]=s_wc_tpl->z0; ob->dims[1]=s_wc_tpl->z1; ob->dims[2]=s_wc_tpl->z2;
      if (npu_create_bo(c->fd, out_sz, CREATE_FLAGS_BUF, &ob->bo)<0){ free(ob); free(Zh); return make_err("matmul out create_bo failed"); }
      if (ob->bo.mm) memcpy(ob->bo.mm, Zh, out_sz<ob->bo.size?out_sz:ob->bo.size);
      npu_sync_bo(c->fd, ob->bo.obj_addr, ob->bo.size, 1);
      free(Zh);
      outlist[0]=(PJRT_Buffer*)ob;
      if(args->device_complete_events) args->device_complete_events[0]=NULL;
      return NULL;
    }
    /* SLOW PATH: shape mismatch (non-decomposed shapes). Free device buffers +
     * close c->fd so the new s_wc_mfd claims the top, stage X/W to host, setup,
     * reopen c->fd. NOTE: this frees the device buffers, so the jax device_put
     * cache for THIS shape is invalidated (the decomposed 768x768 path never
     * hits this -- it stays on the fast path). */
    npu_sync_bo(c->fd, Xb->bo.obj_addr, Xb->bo.size, 2);
    uint8_t* Xh=(uint8_t*)malloc(Xb->bo.size);
    if (!Xh) return make_err("matmul: X alloc failed");
    if (Xb->bo.mm) memcpy(Xh, Xb->bo.mm, Xb->bo.size);
    npu_sync_bo(c->fd, Wb->bo.obj_addr, Wb->bo.size, 2);
    uint8_t* Wh=(uint8_t*)malloc(Wb->bo.size);
    if (!Wh){ free(Xh); return make_err("matmul: W alloc failed"); }
    if (Wb->bo.mm) memcpy(Wh, Wb->bo.mm, Wb->bo.size);
    uint64_t w_sz=Wb->bo.size;
    npu_bo_free(&Xb->bo); npu_bo_free(&Wb->bo);
    close(c->fd); c->fd=-1;
    if (s_wc_mfd>=0) npu_mm_cache_close();
    int sr=npu_mm_cache_setup(MM_NWPRE, M, K, Nv);
    if (sr<0){ free(Xh); free(Wh); c->fd=open(NPU_DEV,O_RDWR); npu_action(c->fd,0,0xFFFFFFFFu); npu_action(c->fd,1,0); npu_action(c->fd,19,0xFFFFFFEDu); npu_action(c->fd,1,0); npu_action(c->fd,18,0); return make_err("matmul: cache_setup failed"); }
    s_wc_shape[0]=M; s_wc_shape[1]=K; s_wc_shape[2]=Nv; c->mm_nw=0;
    c->fd=open(NPU_DEV, O_RDWR);
    npu_action(c->fd,0,0xFFFFFFFFu); npu_action(c->fd,1,0); npu_action(c->fd,19,0xFFFFFFEDu); npu_action(c->fd,1,0); npu_action(c->fd,18,0);
    uint64_t fp0=0, fp1=0;
    if (w_sz>=16){ memcpy(&fp0, Wh, 8); memcpy(&fp1, Wh+8, 8); }
    int w_idx=0;
    npu_mm_cache_load(0, Wh);
    c->mm_wsrc[0]=fp0; c->mm_wsrc2[0]=fp1; c->mm_nw=1;
    free(Wh);
    uint64_t out_sz=(uint64_t)s_wc_tpl->z0*s_wc_tpl->z1*s_wc_tpl->z2*2;
    uint8_t* Zh=(uint8_t*)malloc(out_sz);
    if (!Zh){ free(Xh); return make_err("matmul: Z alloc failed"); }
    int rr=npu_mm_cache_run(0, Xh, Zh);
    free(Xh);
    if (rr){ free(Zh); return make_err("matmul: cache_run failed"); }
    NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
    ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=3;
    ob->dims[0]=s_wc_tpl->z0; ob->dims[1]=s_wc_tpl->z1; ob->dims[2]=s_wc_tpl->z2;
    if (npu_create_bo(c->fd, out_sz, CREATE_FLAGS_BUF, &ob->bo)<0){ free(ob); free(Zh); return make_err("matmul out create_bo failed"); }
    if (ob->bo.mm) memcpy(ob->bo.mm, Zh, out_sz<ob->bo.size?out_sz:ob->bo.size);
    npu_sync_bo(c->fd, ob->bo.obj_addr, ob->bo.size, 1);
    free(Zh);
    outlist[0]=(PJRT_Buffer*)ob;
    if(args->device_complete_events) args->device_complete_events[0]=NULL;
    return NULL;
  }
  /* op=7: cached-weight matmul custom op (stablehlo.custom_call @npu_cached_mm).
   * W is a plugin-managed cached weight (loaded at Client_Create from
   * $NPU_WEIGHTS_DIR/w{N}.bin into s_wc_wb[N]); only X is a jax buffer. No
   * device_put of W -> no cross-fd / device-put-of-W issues. */
  if (ex->op == 7) {
    if (!in0) return make_err("npu_cached_mm needs 1 input (X)");
    NpuBuffer* Xb = in0;
    /* Use CNA descriptor cache if available (variable shapes, fp32 output) */
    if (npu_cna_ready()) {
      if (ex->w_idx < 0 || ex->w_idx >= 144) return make_err("npu_cached_mm: bad w_idx");
      int K = npu_cna_wt_k(ex->w_idx);
      int N = npu_cna_wt_n(ex->w_idx);
      if (K <= 0 || N <= 0) return make_err("npu_cached_mm: weight not loaded");
      int M = Xb->bo.size / 2 / K;  /* fp16 input: size/2 = M*K */
      npu_sync_bo(c->fd, Xb->bo.obj_addr, Xb->bo.size, 2);
      uint64_t out_sz = (uint64_t)M * N * 2;  /* fp16 output */
      float* Z32 = (float*)malloc((size_t)M * N * 4);
      if (!Z32) return make_err("npu_cached_mm: Z alloc failed");
      int rr = npu_cna_cache_run_m(ex->w_idx, M, Xb->bo.mm, Z32);
      if (rr) { free(Z32); return make_err("npu_cached_mm: CNA run failed"); }
      uint16_t* Zh16 = (uint16_t*)malloc(out_sz);
      if (!Zh16) { free(Z32); return make_err("npu_cached_mm: Zh alloc failed"); }
      for (int i = 0; i < M * N; i++) {
        float v = Z32[i];
        v = v > 65504.0f ? 65504.0f : (v < -65504.0f ? -65504.0f : v);
        Zh16[i] = (uint16_t)f32_to_fp16(v);
      }
      free(Z32);
      NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
      ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=3;
      ob->dims[0]=1; ob->dims[1]=M; ob->dims[2]=N;
      if (npu_create_bo(c->fd, out_sz, CREATE_FLAGS_BUF, &ob->bo)<0){ free(ob); free(Zh16); return make_err("npu_cached_mm out create_bo failed"); }
      if (ob->bo.mm) memcpy(ob->bo.mm, Zh16, out_sz<ob->bo.size?out_sz:ob->bo.size);
      npu_sync_bo(c->fd, ob->bo.obj_addr, ob->bo.size, 1);
      free(Zh16);
      outlist[0]=(PJRT_Buffer*)ob;
      if(args->device_complete_events) args->device_complete_events[0]=NULL;
      return NULL;
    }
    /* Fallback to template-based weight cache (s_wc) */
    if (s_wc_mfd < 0) return make_err("npu_cached_mm: weight cache not set up");
    if (ex->w_idx < 0 || ex->w_idx >= s_wc_nw) return make_err("npu_cached_mm: bad w_idx");
    npu_sync_bo(c->fd, Xb->bo.obj_addr, Xb->bo.size, 2);
    uint64_t out_sz = (uint64_t)s_wc_tpl->z0*s_wc_tpl->z1*s_wc_tpl->z2*2;
    uint8_t* Zh = (uint8_t*)malloc(out_sz);
    if (!Zh) return make_err("npu_cached_mm: Z alloc failed");
    int rr = npu_mm_cache_run(ex->w_idx, Xb->bo.mm, Zh);
    if (rr) { free(Zh); return make_err("npu_cached_mm: run failed"); }
    NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
    ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=3;
    ob->dims[0]=s_wc_tpl->z0; ob->dims[1]=s_wc_tpl->z1; ob->dims[2]=s_wc_tpl->z2;
    if (npu_create_bo(c->fd, out_sz, CREATE_FLAGS_BUF, &ob->bo)<0){ free(ob); free(Zh); return make_err("npu_cached_mm out create_bo failed"); }
    if (ob->bo.mm) memcpy(ob->bo.mm, Zh, out_sz<ob->bo.size?out_sz:ob->bo.size);
    npu_sync_bo(c->fd, ob->bo.obj_addr, ob->bo.size, 1);
    free(Zh);
    outlist[0]=(PJRT_Buffer*)ob;
    if(args->device_complete_events) args->device_complete_events[0]=NULL;
    return NULL;
  }
  /* ── multi-op DAG of binary elementwise (add/sub/mul/div) + relu ──────── */
  /* Generalizes linear chains to arbitrary DAGs: each op's operands may be any
   * args or any PREVIOUS op results. Binary ops use the 6-BO elementwise kernel
   * (scratch IN1/IN2 -> bos[5]); relu (unary) uses the pre-allocated 5-BO relu
   * kernel (in io BO -> reader -> scratch -> out io BO). Per 49152-chunk we run
   * ops in definition (=topological) order, holding each result in a host buffer
   * mid[k]; the final op's result (g_out_op) goes to the output chunk. Single-op
   * functions are handled by the dedicated paths; this branch is multi-op only. */
  if (ex->is_graph) {
    NpuBuffer* ins[8];
    for (int i=0;i<ex->n_args && i<8;i++) ins[i]=(NpuBuffer*)args->argument_lists[0][i];
    int64_t n=1; for(int i=0;i<ins[0]->ndims;i++) n*=ins[0]->dims[i];
    uint64_t io_sz=(uint64_t)n*2;
    int64_t n_chunks=(n+KELEM-1)/KELEM;
    int has_binary=0; for(int k=0;k<ex->n_ops;k++) if(ex->g_ops[k]<4) has_binary=1;
    int has_relu=ex->g_has_relu;
    int has_tanh=ex->g_has_tanh;
    /* elementwise kernel BOs (only if a binary op is present) */
    struct npu_bo cb[6]; int have_cb=0;
    if (has_binary) {
      uint64_t csz[6]={4096,8192,0x60000,KTSZ,KTSZ,KTSZ};
      uint32_t cfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
      for(int i=0;i<6;i++) if(npu_create_bo(c->fd,csz[i],cfl[i],&cb[i])<0){ for(int j=0;j<i;j++) npu_bo_free(&cb[j]); return make_err("graph: create_bo failed"); }
      have_cb=1;
      uint64_t dm[6]; for(int i=0;i<6;i++) dm[i]=cb[i].dma_addr;
      if(cb[2].mm) memset(cb[2].mm,0,cb[2].size);
      if(cb[1].mm) memcpy(cb[1].mm,g_regcmd_tmpl,sizeof(g_regcmd_tmpl)<cb[1].size?sizeof(g_regcmd_tmpl):cb[1].size);
      if(cb[0].mm) memcpy(cb[0].mm,g_task_tmpl,sizeof(g_task_tmpl)<cb[0].size?sizeof(g_task_tmpl):cb[0].size);
      uint64_t rcb=dm[1]+0xC0;
      static const int blk_off[4]={0x0,0x280,0x500,0x780};
      #define CPDMA(bo_off,entry,dma) do{ uint8_t* p=(uint8_t*)cb[1].mm+(bo_off)+(entry)*8; uint16_t v=(uint16_t)((dma)&0xFFFF); uint32_t t=(0x2001<<16)|(((dma)>>16)&0xFFFF); if((entry)==5){t=(0x1001<<16)|(((dma)>>16)&0xFFFF);} memcpy(p+2,&v,2); memcpy(p+4,&t,4);}while(0)
      CPDMA(0xC0+blk_off[0],5,dm[3]); CPDMA(0xC0+blk_off[0],55,dm[2]+0x18000);
      CPDMA(0xC0+blk_off[1],5,dm[4]); CPDMA(0xC0+blk_off[1],55,dm[2]+0x0);
      CPDMA(0xC0+blk_off[2],5,dm[2]+0x30000); CPDMA(0xC0+blk_off[2],55,dm[3]); CPDMA(0xC0+blk_off[2],61,dm[4]);
      CPDMA(0xC0+blk_off[3],5,dm[5]); CPDMA(0xC0+blk_off[3],55,dm[2]+0x30000);
      #define CPCHAIN(bo_off,nextoff) do{ uint8_t* p=(uint8_t*)cb[1].mm+(bo_off)+69*8; uint64_t nd=rcb+(nextoff); uint16_t rv=0x0010,vv=(uint16_t)(nd&0xFFFF); uint32_t tt=(0x0101<<16)|((nd>>16)&0xFFFF); memcpy(p+0,&rv,2); memcpy(p+2,&vv,2); memcpy(p+4,&tt,4);}while(0)
      CPCHAIN(0xC0+blk_off[0],blk_off[1]); CPCHAIN(0xC0+blk_off[1],blk_off[2]); CPCHAIN(0xC0+blk_off[2],blk_off[3]);
      for(int ti=0;ti<12;ti++){ uint8_t* p=(uint8_t*)cb[0].mm+ti*40+32; uint64_t a=rcb+blk_off[ti%4]; memcpy(p,&a,8); }
      #undef CPDMA
      #undef CPCHAIN
      for(int i=0;i<6;i++) npu_sync_bo(c->fd,cb[i].obj_addr,cb[i].size,3);
      for(int i=0;i<6;i++) if(i!=2) npu_sync_bo(c->fd,cb[i].obj_addr,cb[i].size,1);
    }
    /* relu kernel BOs (pre-allocated at Client_Compile, top of pool). Patch the
     * captured template's DMAs to these BOs ONCE (BOs are fixed across ops/chunks). */
    struct npu_bo* rb = ex->relu_bos;
    if (has_relu) {
      uint32_t rsz32[5]={RELU_SZ_TASK,RELU_SZ_REGCMD,RELU_SZ_SCRATCH,RELU_SZ_IN,RELU_SZ_OUT};
      uint64_t rcap[5]={RELU_CAP_TASK,RELU_CAP_REGCMD,RELU_CAP_SCRATCH,RELU_CAP_IN,RELU_CAP_OUT};
      uint64_t rmy[5]={rb[0].dma_addr,rb[1].dma_addr,rb[2].dma_addr,rb[3].dma_addr,rb[4].dma_addr};
      if(rb[2].mm) memset(rb[2].mm,0,rb[2].size);
      npu_sync_bo(c->fd, rb[2].obj_addr, rb[2].size, 1);
      if(rb[1].mm) memcpy(rb[1].mm, g_relu_regcmd, RELU_REGCMD_SZ<rb[1].size?RELU_REGCMD_SZ:rb[1].size);
      if(rb[0].mm) memcpy(rb[0].mm, g_relu_task, RELU_TASK_SZ<rb[0].size?RELU_TASK_SZ:rb[0].size);
      patch_regcmd_tmpl(rb[1].mm, RELU_REGCMD_SZ, rcap, rmy, rsz32, 5);
      patch_task_tmpl(rb[0].mm, RELU_TASK_SZ, rcap[1], rmy[1], RELU_SZ_REGCMD);
      npu_sync_bo(c->fd, rb[1].obj_addr, rb[1].size, 1);
      npu_sync_bo(c->fd, rb[0].obj_addr, rb[0].size, 1);
    }
    /* tanh kernel BOs (pre-allocated at Client_Compile, top of pool). The 3
     * submits each carry their own regcmd+task template; patching is done per
     * submit per chunk (the regcmd differs per submit, unlike relu's 1 submit).
     * Scratch is zeroed per tanh op (carries state across the 3 submits). */
    struct npu_bo* tb = ex->tanh_bos;
    if (has_tanh) {
      if(tb[2].mm) memset(tb[2].mm,0,tb[2].size);
      npu_sync_bo(c->fd, tb[2].obj_addr, tb[2].size, 1);
    }
    NpuBuffer* ob=(NpuBuffer*)calloc(1,sizeof(NpuBuffer));
    ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=ins[0]->ndims;
    for(int i=0;i<ins[0]->ndims;i++) ob->dims[i]=ins[0]->dims[i];
    if (npu_create_bo(c->fd, io_sz, CREATE_FLAGS_BUF, &ob->bo)<0){
      if(have_cb) for(int i=0;i<6;i++) npu_bo_free(&cb[i]); free(ob); return make_err("graph out create_bo failed"); }
    uint8_t* mid[16]; for(int i=0;i<16;i++) mid[i]=(uint8_t*)malloc(KTSZ);
    static const uint16_t alu_e23v[4]={0x02c0,0x02c0,0x03c4,0x03c0};
    static const uint32_t alu_e23t[4]={(0x1001<<16)|0x1082,(0x1001<<16)|0x1084,(0x1001<<16)|0x1080,(0x1001<<16)|0x1083};
    static const uint16_t alu_e28t[4]={0x0001,0x0001,0x0001,0x0000};
    static const uint16_t alu_e63v[4]={0x7849,0x7849,0x7849,0x7841};
    int ok=1;
    for(int64_t ci=0;ci<n_chunks && ok;ci++){
      int64_t s=ci*KELEM, e=s+KELEM; int64_t bytes=((e>n?n:e)-s)*2;
      for(int k=0;k<ex->n_ops && ok;k++){
        int t=ex->g_ops[k];
        if(t<4){  /* binary elementwise: operands -> scratch IN1/IN2 -> bos[5] */
          if(cb[2].mm){
            memset((uint8_t*)cb[2].mm+0x18000,0,KTSZ); memset((uint8_t*)cb[2].mm+0x0,0,KTSZ);
            for(int j=0;j<2;j++){
              uint8_t* dst=(j==0)?(uint8_t*)cb[2].mm+0x18000:(uint8_t*)cb[2].mm+0x0;
              if(ex->g_opnd_isconst[k][j]){  /* baked scalar fp16 const: broadcast-fill the region */
                uint16_t cv=ex->g_const_val[k][j]; uint8_t lo=(uint8_t)(cv&0xff), hi=(uint8_t)(cv>>8);
                for(int ee=0;ee<KTSZ/2;ee++){ dst[ee*2]=lo; dst[ee*2+1]=hi; }
              } else if(ex->g_opnd_isarg[k][j]){ NpuBuffer* ab=ins[ex->g_opnd_idx[k][j]]; if(ab->bo.mm) memcpy(dst,(uint8_t*)ab->bo.mm+s*2,bytes); }
              else memcpy(dst,mid[ex->g_opnd_idx[k][j]],bytes);
            }
          }
          npu_sync_bo(c->fd,cb[2].obj_addr,cb[2].size,1);
          { uint8_t* base=(uint8_t*)cb[1].mm+0xC0+0x500; int op=t;
            #define CSETALU(entry,val,tag) do{ uint8_t* pp=base+(entry)*8; uint16_t _v=(val); uint32_t _t=(tag); memcpy(pp+2,&_v,2); memcpy(pp+4,&_t,4);}while(0)
            CSETALU(23,alu_e23v[op],alu_e23t[op]); CSETALU(28,0x0001,(0x1001<<16)|alu_e28t[op]); CSETALU(63,alu_e63v[op],(0x2001<<16)|0x0001);
            #undef CSETALU
          }
          npu_sync_bo(c->fd,cb[1].obj_addr,cb[1].size,1);
          uint8_t sub[104]; memset(sub,0,104);
          *(uint32_t*)(sub+0)=5; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=0; *(uint32_t*)(sub+12)=4;
          *(uint32_t*)(sub+16)=4; *(uint64_t*)(sub+24)=cb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
          *(uint32_t*)(sub+64)=0; *(uint32_t*)(sub+68)=4;
          if(ioctl(c->fd,IOCTL_SUBMIT,sub)<0){ ok=0; break; }
          npu_sync_bo(c->fd,cb[5].obj_addr,cb[5].size,2);
          if(cb[5].mm) memcpy(mid[k],cb[5].mm,bytes);
        } else if(t==4){  /* relu: operand -> rb[3] in io -> rb[4] out io */
          if(rb[3].mm){
            memset(rb[3].mm,0,RELU_SZ_IN);
            if(ex->g_opnd_isarg[k][0]){ NpuBuffer* ab=ins[ex->g_opnd_idx[k][0]]; if(ab->bo.mm) memcpy(rb[3].mm,(uint8_t*)ab->bo.mm+s*2,bytes); }
            else memcpy(rb[3].mm,mid[ex->g_opnd_idx[k][0]],bytes);
          }
          if(rb[4].mm) memset(rb[4].mm,0,RELU_SZ_OUT);
          npu_sync_bo(c->fd, rb[3].obj_addr, rb[3].size, 1);
          npu_sync_bo(c->fd, rb[4].obj_addr, rb[4].size, 1);
          uint8_t sub[104]; memset(sub,0,104);
          *(uint32_t*)(sub+0)=RELU_FLAGS; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=RELU_TASK_START; *(uint32_t*)(sub+12)=RELU_N_TASKS;
          *(uint32_t*)(sub+16)=RELU_N_TASKS; *(uint64_t*)(sub+24)=rb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
          for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=RELU_SUBCORE[si*2]; *(uint32_t*)(sub+64+si*8+4)=RELU_SUBCORE[si*2+1]; }
          if(ioctl(c->fd,IOCTL_SUBMIT,sub)<0){ ok=0; break; }
          npu_sync_bo(c->fd, rb[4].obj_addr, rb[4].size, 2);
          if(rb[4].mm) memcpy(mid[k], rb[4].mm, bytes);
        } else if(t==5){  /* tanh: operand -> tb[3] in io -> 3 submits -> tb[4] out io */
          static const unsigned char* const th2[3]={g_tanh_regcmd0,g_tanh_regcmd1,g_tanh_regcmd2};
          static const unsigned char* const th1[3]={g_tanh_task0,g_tanh_task1,g_tanh_task2};
          static const uint32_t tstart[3]={TANH_SUB0_START,TANH_SUB1_START,TANH_SUB2_START};
          static const uint32_t tntask[3]={TANH_SUB0_NTASK,TANH_SUB1_NTASK,TANH_SUB2_NTASK};
          static const uint32_t tflags[3]={TANH_SUB0_FLAGS,TANH_SUB1_FLAGS,TANH_SUB2_FLAGS};
          static const uint32_t* const tsc[3]={TANH_SUB0_SUBCORE,TANH_SUB1_SUBCORE,TANH_SUB2_SUBCORE};
          uint32_t tsz32[5] = {TANH_SZ_TASK, TANH_SZ_REGCMD, TANH_SZ_SCRATCH, TANH_SZ_IN, TANH_SZ_OUT};
          uint64_t tcap[5]={TANH_CAP_TASK,TANH_CAP_REGCMD,TANH_CAP_SCRATCH,TANH_CAP_IN,TANH_CAP_OUT};
          uint64_t tmy[5]={tb[0].dma_addr,tb[1].dma_addr,tb[2].dma_addr,tb[3].dma_addr,tb[4].dma_addr};
          if(tb[2].mm){ memset(tb[2].mm,0,tb[2].size); }  /* scratch carries state across the 3 submits */
          npu_sync_bo(c->fd, tb[2].obj_addr, tb[2].size, 1);
          if(tb[3].mm){
            memset(tb[3].mm,0,TANH_SZ_IN);
            if(ex->g_opnd_isarg[k][0]){ NpuBuffer* ab=ins[ex->g_opnd_idx[k][0]]; if(ab->bo.mm) memcpy(tb[3].mm,(uint8_t*)ab->bo.mm+s*2,bytes); }
            else memcpy(tb[3].mm,mid[ex->g_opnd_idx[k][0]],bytes);
          }
          if(tb[4].mm) memset(tb[4].mm,0,TANH_SZ_OUT);
          npu_sync_bo(c->fd, tb[3].obj_addr, tb[3].size, 1);
          npu_sync_bo(c->fd, tb[4].obj_addr, tb[4].size, 1);
          for (int sd=0; sd<TANH_NSUBMITS && ok; sd++){
            if(tb[1].mm) memcpy(tb[1].mm, th2[sd], TANH_SZ_REGCMD);
            if(tb[0].mm) memcpy(tb[0].mm, th1[sd], TANH_SZ_TASK);
            patch_regcmd_tmpl(tb[1].mm, TANH_SZ_REGCMD, tcap, tmy, tsz32, 5);
            patch_task_tmpl(tb[0].mm, TANH_SZ_TASK, tcap[1], tmy[1], TANH_SZ_REGCMD);
            npu_sync_bo(c->fd, tb[1].obj_addr, tb[1].size, 1);
            npu_sync_bo(c->fd, tb[0].obj_addr, tb[0].size, 1);
            uint8_t sub[104]; memset(sub,0,104);
            *(uint32_t*)(sub+0)=tflags[sd]; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tstart[sd]; *(uint32_t*)(sub+12)=tntask[sd];
            *(uint32_t*)(sub+16)=tntask[sd]; *(uint64_t*)(sub+24)=tb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
            for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=tsc[sd][si*2]; *(uint32_t*)(sub+64+si*8+4)=tsc[sd][si*2+1]; }
            if(ioctl(c->fd,IOCTL_SUBMIT,sub)<0){ ok=0; break; }
          }
          if(!ok) break;
          npu_sync_bo(c->fd, tb[4].obj_addr, tb[4].size, 2);
          if(tb[4].mm) memcpy(mid[k], tb[4].mm, bytes);
        }
      }
      if(ok && ob->bo.mm) memcpy((uint8_t*)ob->bo.mm+s*2, mid[ex->g_out_op], bytes);
    }
    for(int i=0;i<16;i++) free(mid[i]);
    if(ob->bo.mm) npu_sync_bo(c->fd,ob->bo.obj_addr,ob->bo.size,1);
    outlist[0]=(PJRT_Buffer*)ob;
    if(args->device_complete_events) args->device_complete_events[0]=NULL;
    if(have_cb) for(int i=0;i<6;i++) npu_bo_free(&cb[i]);
    if(!ok) return make_err("graph submit failed during tiling");
    return NULL;
  }
  struct npu_bo bos[6];
  uint64_t sizes[6] = {4096, 8192, 0x60000, KTSZ, KTSZ, KTSZ};
  uint32_t flgs[6] = {CREATE_FLAGS_TASK, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF, CREATE_FLAGS_BUF};
  for (int i=0;i<6;i++) if (npu_create_bo(c->fd, sizes[i], flgs[i], &bos[i])<0) return make_err("exec: create_bo failed");
  uint64_t* dm = (uint64_t*)alloca(sizeof(uint64_t)*6);
  for (int i=0;i<6;i++) dm[i]=bos[i].dma_addr;
  /* Zero the scratch: a zeroed scratch is sufficient for elementwise ops
   * (verified on-board: zeroed == scratch_init for add/sub/mul). This removes
   * the per-op capture dependency. */
  if (bos[2].mm) memset(bos[2].mm, 0, bos[2].size);
  /* load regcmd + task templates */
  if (bos[1].mm) memcpy(bos[1].mm, g_regcmd_tmpl, sizeof(g_regcmd_tmpl)<bos[1].size?sizeof(g_regcmd_tmpl):bos[1].size);
  if (bos[0].mm) memcpy(bos[0].mm, g_task_tmpl, sizeof(g_task_tmpl)<bos[0].size?sizeof(g_task_tmpl):bos[0].size);
  /* Patch the compute block's ALU selectors (entries 23, 28, 63) for the
   * detected op. The template is baked with add selectors; sub/mul/div differ
   * only in these 3 entries. Compute block is block 2 at 0xC0+0x500. */
  do {
    uint8_t* base = (uint8_t*)bos[1].mm + 0xC0 + 0x500;
    uint16_t e23v; uint32_t e23t; uint16_t e28t; uint16_t e63v;
    switch (ex->op) {
      case 1: e23v=0x02c0; e23t=(0x1001<<16)|0x1084; e28t=0x0001; e63v=0x7849; break; /* sub */
      case 2: e23v=0x03c4; e23t=(0x1001<<16)|0x1080; e28t=0x0001; e63v=0x7849; break; /* mul */
      case 3: e23v=0x03c0; e23t=(0x1001<<16)|0x1083; e28t=0x0000; e63v=0x7841; break; /* div */
      default: e23v=0x02c0; e23t=(0x1001<<16)|0x1082; e28t=0x0001; e63v=0x7849; break; /* add */
    }
    /* entry: reg(u16) val(u16) tag(u32) — patch val(bytes 2-3) + tag(bytes 4-7) */
    #define SETALU(entry, val, tag) do{ uint8_t* p=base+(entry)*8; uint16_t _v=(val); uint32_t _t=(tag); memcpy(p+2,&_v,2); memcpy(p+4,&_t,4);}while(0)
    SETALU(23, e23v, e23t);
    SETALU(28, 0x0001, (0x1001<<16)|e28t);
    SETALU(63, e63v, (0x2001<<16)|0x0001);
    #undef SETALU
  } while(0);
  /* patch regcmd DMAs + chain ptrs + task regcmd_addrs */
  uint64_t rcb = dm[1]+0xC0;
  static const int blk_off[4]={0x0,0x280,0x500,0x780};
  /* scratch offsets */
  #define IN1_OFF 0x18000
  #define IN2_OFF 0x0
  #define RES_OFF 0x30000
  /* patch a DMA entry: byte offset within regcmd BO, val+tag */
  #define PDMA(bo_off, entry, dma) do{ uint8_t* p=(uint8_t*)bos[1].mm+(bo_off)+(entry)*8; uint16_t v=(uint16_t)((dma)&0xFFFF); uint32_t t=(0x2001<<16)|(((dma)>>16)&0xFFFF); if((entry)==5){t=(0x1001<<16)|(((dma)>>16)&0xFFFF);} memcpy(p+2,&v,2); memcpy(p+4,&t,4);}while(0)
  /* block 0 (reader1) at 0xC0: e5=in1, e55=scratch+IN1_OFF */
  PDMA(0xC0+blk_off[0],5,dm[3]); PDMA(0xC0+blk_off[0],55,dm[2]+IN1_OFF);
  /* block 1 (reader2): e5=in2, e55=scratch+IN2_OFF */
  PDMA(0xC0+blk_off[1],5,dm[4]); PDMA(0xC0+blk_off[1],55,dm[2]+IN2_OFF);
  /* block 2 (compute): e5=scratch+RES_OFF, e55=in1, e61=in2 */
  PDMA(0xC0+blk_off[2],5,dm[2]+RES_OFF); PDMA(0xC0+blk_off[2],55,dm[3]); PDMA(0xC0+blk_off[2],61,dm[4]);
  /* block 3 (writer): e5=out, e55=scratch+RES_OFF */
  PDMA(0xC0+blk_off[3],5,dm[5]); PDMA(0xC0+blk_off[3],55,dm[2]+RES_OFF);
  /* chain pointers (entry 69): reg 0x0010, cid 0x0101, val=absolute DMA of next block */
  #define PCHAIN(bo_off, nextoff) do{ uint8_t* p=(uint8_t*)bos[1].mm+(bo_off)+69*8; uint64_t nd=rcb+(nextoff); uint16_t regv=0x0010, vv=(uint16_t)(nd&0xFFFF); uint32_t tt=(0x0101<<16)|((nd>>16)&0xFFFF); memcpy(p+0,&regv,2); memcpy(p+2,&vv,2); memcpy(p+4,&tt,4);}while(0)
  PCHAIN(0xC0+blk_off[0], blk_off[1]);
  PCHAIN(0xC0+blk_off[1], blk_off[2]);
  PCHAIN(0xC0+blk_off[2], blk_off[3]);
  /* block 3 chain ptr stays 0 (end) */
  /* patch task regcmd_addrs: task ti -> rcb + blk_off[ti%4] */
  for (int ti=0; ti<12; ti++) { uint8_t* p=(uint8_t*)bos[0].mm+ti*40+32; uint64_t a=rcb+blk_off[ti%4]; memcpy(p,&a,8); }
  /* one-time syncs for fixed kernel BOs (scratch synced per chunk below) */
  for (int i=0;i<6;i++) npu_sync_bo(c->fd, bos[i].obj_addr, bos[i].size, 3);
  for (int i=0;i<6;i++) if(i!=2) npu_sync_bo(c->fd, bos[i].obj_addr, bos[i].size, 1);
  /* build output PJRT_Buffer (full size; CPU-copied into, never NPU-DMA'd) */
  NpuBuffer* ob = (NpuBuffer*)calloc(1, sizeof(NpuBuffer));
  ob->cli=c; ob->type=PJRT_Buffer_Type_F16; ob->ndims=in0->ndims;
  for (int i=0;i<in0->ndims;i++) ob->dims[i]=in0->dims[i];
  if (npu_create_bo(c->fd, io_sz, CREATE_FLAGS_BUF, &ob->bo)<0){ for(int i=0;i<6;i++) npu_bo_free(&bos[i]); free(ob); return make_err("out create_bo failed"); }
  /* tile loop: per chunk, overlay inputs into scratch, submit, read result */
  int ok = 1;
  for (int64_t ci=0; ci<n_chunks; ci++){
    int64_t s = ci*KELEM, e = s+KELEM;
    int64_t bytes = ((e>n ? n : e) - s) * 2;   /* bytes this chunk (<=KTSZ) */
    if (bos[2].mm) {
      memset((uint8_t*)bos[2].mm+IN1_OFF, 0, KTSZ);
      memset((uint8_t*)bos[2].mm+IN2_OFF, 0, KTSZ);
      if (in0->bo.mm) memcpy((uint8_t*)bos[2].mm+IN1_OFF, (uint8_t*)in0->bo.mm + s*2, bytes);
      if (in1->bo.mm) memcpy((uint8_t*)bos[2].mm+IN2_OFF, (uint8_t*)in1->bo.mm + s*2, bytes);
    }
    npu_sync_bo(c->fd, bos[2].obj_addr, bos[2].size, 1);
    uint8_t sub[104]; memset(sub,0,104);
    *(uint32_t*)(sub+0)=5; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=0; *(uint32_t*)(sub+12)=4;
    *(uint32_t*)(sub+16)=4; *(uint64_t*)(sub+24)=bos[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1;
    *(uint32_t*)(sub+64)=0; *(uint32_t*)(sub+68)=4;
    if (ioctl(c->fd, IOCTL_SUBMIT, sub) < 0) { ok=0; break; }
    npu_sync_bo(c->fd, bos[5].obj_addr, bos[5].size, 2);
    if (ob->bo.mm && bos[5].mm) memcpy((uint8_t*)ob->bo.mm + s*2, bos[5].mm, bytes);
  }
  if (ob->bo.mm) npu_sync_bo(c->fd, ob->bo.obj_addr, ob->bo.size, 1);
  outlist[0] = (PJRT_Buffer*)ob;
  if (args->device_complete_events) args->device_complete_events[0]=NULL;
  for (int i=0;i<6;i++) npu_bo_free(&bos[i]);
  if (!ok) return make_err("submit failed during tiling");
  return NULL;
}
/* ===== Raw matmul (no jax, no PJRT) — called via ctypes from Python =====
 * Keeps a persistent matmul fd + 6 BOs cached per shape, reused for same-shape
 * matmuls (no close+reopen, no BO realloc, no DMA re-patch). Host X/W are written
 * DIRECTLY to the mfd io0/io1 mmaps (1 hop, no device_put). This cuts the
 * per-matmul cost from ~23ms (jax path: device_put + close+reopen c->fd +
 * close+reopen mfd) to ~4ms reuse / ~13ms shape-switch. */
static int s_mm_mfd = -1;
static int s_mm_shape[5] = {0};
static const MMTemplate* s_mm_tpl = NULL;
static struct npu_bo s_mm_bo[6];

__attribute__((visibility("default")))
int npu_matmul_raw(const void* Xh, const void* Wh, int M, int K, int N, void* Zh) {
  const MMTemplate* tpl = NULL;
  for (int i=0;i<MM_NSHAPES;i++){ const MMTemplate* t=&mm_templates[i];
    if (t->x0==1 && t->x1==M && t->x2==K && t->w0==K && t->w1==N){ tpl=t; break; } }
  if (!tpl) return -1;
  int shape[5] = {1, M, K, K, N};
  int reuse = (s_mm_mfd >= 0 && s_mm_tpl == tpl && memcmp(s_mm_shape, shape, 20)==0);
  int mfd; struct npu_bo* mb;
  if (reuse) {
    mfd = s_mm_mfd; mb = s_mm_bo;
  } else {
    if (s_mm_mfd >= 0) { for(int i=0;i<6;i++) npu_bo_free(&s_mm_bo[i]); close(s_mm_mfd); s_mm_mfd=-1; }
    mfd = open(NPU_DEV, O_RDWR);
    if (mfd < 0) return -2;
    npu_action(mfd,0,0xFFFFFFFFu); npu_action(mfd,1,0); npu_action(mfd,19,0xFFFFFFEDu); npu_action(mfd,1,0); npu_action(mfd,18,0);
    uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
    uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
    int ce=0; for(int i=0;i<6;i++){ if(npu_create_bo(mfd,msz[i],mfl[i],&s_mm_bo[i])<0){ ce=1; break; } }
    if (ce){ for(int i=0;i<6;i++) npu_bo_free(&s_mm_bo[i]); close(mfd); return -3; }
    s_mm_mfd=mfd; s_mm_tpl=tpl; memcpy(s_mm_shape, shape, sizeof(shape));
    mb = s_mm_bo;
    uint64_t cap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
    uint64_t my[6]; for(int i=0;i<6;i++) my[i]=mb[i].dma_addr;
    uint32_t sz32[6]; for(int i=0;i<6;i++) sz32[i]=(uint32_t)msz[i];
    if(mb[1].mm) memcpy(mb[1].mm, tpl->regcmd_tmpl, tpl->sz_regcmd<mb[1].size?tpl->sz_regcmd:mb[1].size);
    if(mb[0].mm) memcpy(mb[0].mm, tpl->task_tmpl, tpl->sz_task<mb[0].size?tpl->sz_task:mb[0].size);
    patch_regcmd_tmpl(mb[1].mm, tpl->sz_regcmd, cap, my, sz32, 6);
    patch_task_tmpl(mb[0].mm, tpl->sz_task, cap[1], my[1], tpl->sz_regcmd);
  }
  /* (re)write inputs + zero scratch/output, sync, submit, read. */
  if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch);
  if(mb[3].mm) memcpy(mb[3].mm, Xh, tpl->sz_in);
  if(mb[4].mm) memcpy(mb[4].mm, Wh, tpl->sz_w);
  if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out);
  for(int i=0;i<6;i++) npu_sync_bo(mfd, mb[i].obj_addr, mb[i].size, 1);
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks;
  *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint64_t*)(sub+40)=0; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2]; *(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1]; }
  if(ioctl(mfd, IOCTL_SUBMIT, sub)<0){ for(int i=0;i<6;i++) npu_bo_free(&s_mm_bo[i]); close(s_mm_mfd); s_mm_mfd=-1; return -4; }
  npu_sync_bo(mfd, mb[5].obj_addr, mb[5].size, 2);
  uint64_t out_sz = (uint64_t)tpl->z0*tpl->z1*tpl->z2*2;
  if (mb[5].mm) memcpy(Zh, mb[5].mm, out_sz);
  return 0;
}

__attribute__((visibility("default")))
void npu_matmul_close(void) {
  if (s_mm_mfd >= 0) { for(int i=0;i<6;i++) npu_bo_free(&s_mm_bo[i]); close(s_mm_mfd); s_mm_mfd=-1; }
}

/* Test NPU DMA reach: alloc n_filler 1.1MB BOs (push the pool down), then alloc
 * the 6 mm_768x768 BOs, run a known matmul, report io2 IOVA + correctness.
 * If the matmul works with io2 < 0xfed64000, the reach window is > 19.5MB. */
__attribute__((visibility("default")))
int npu_test_reach(int n_filler, int M, int K, int N,
                   const void* Xh, const void* Wh, void* Zh, uint64_t* io2_iova_out) {
  const MMTemplate* tpl = NULL;
  for (int i=0;i<MM_NSHAPES;i++){ const MMTemplate* t=&mm_templates[i];
    if (t->x0==1 && t->x1==M && t->x2==K && t->w0==K && t->w1==N){ tpl=t; break; } }
  if (!tpl) return -1;
  int mfd = open(NPU_DEV, O_RDWR);
  if (mfd < 0) return -2;
  npu_action(mfd,0,0xFFFFFFFFu); npu_action(mfd,1,0); npu_action(mfd,19,0xFFFFFFEDu); npu_action(mfd,1,0); npu_action(mfd,18,0);
  struct npu_bo filler[64];
  if (n_filler > 64) n_filler = 64;
  for (int i=0;i<n_filler;i++){ if(npu_create_bo(mfd,1179648,CREATE_FLAGS_BUF,&filler[i])<0){ n_filler=i; break; } }
  uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
  uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  struct npu_bo mb[6]; int ce=0;
  for(int i=0;i<6;i++){ if(npu_create_bo(mfd,msz[i],mfl[i],&mb[i])<0){ ce=1; break; } }
  if (ce){ for(int i=0;i<6;i++) npu_bo_free(&mb[i]); for(int i=0;i<n_filler;i++) npu_bo_free(&filler[i]); close(mfd); return -3; }
  *io2_iova_out = mb[5].dma_addr;
  uint64_t cap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
  uint64_t my[6]; for(int i=0;i<6;i++) my[i]=mb[i].dma_addr;
  uint32_t sz32[6]; for(int i=0;i<6;i++) sz32[i]=(uint32_t)msz[i];
  if(mb[1].mm) memcpy(mb[1].mm, tpl->regcmd_tmpl, tpl->sz_regcmd<mb[1].size?tpl->sz_regcmd:mb[1].size);
  if(mb[0].mm) memcpy(mb[0].mm, tpl->task_tmpl, tpl->sz_task<mb[0].size?tpl->sz_task:mb[0].size);
  patch_regcmd_tmpl(mb[1].mm, tpl->sz_regcmd, cap, my, sz32, 6);
  patch_task_tmpl(mb[0].mm, tpl->sz_task, cap[1], my[1], tpl->sz_regcmd);
  if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch);
  if(mb[3].mm) memcpy(mb[3].mm, Xh, tpl->sz_in);
  if(mb[4].mm) memcpy(mb[4].mm, Wh, tpl->sz_w);
  if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out);
  for(int i=0;i<6;i++) npu_sync_bo(mfd, mb[i].obj_addr, mb[i].size, 1);
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks;
  *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2]; *(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1]; }
  int sr = ioctl(mfd, IOCTL_SUBMIT, sub);
  if (sr < 0) { for(int i=0;i<6;i++) npu_bo_free(&mb[i]); for(int i=0;i<n_filler;i++) npu_bo_free(&filler[i]); close(mfd); return -4; }
  npu_sync_bo(mfd, mb[5].obj_addr, mb[5].size, 2);
  uint64_t out_sz = (uint64_t)tpl->z0*tpl->z1*tpl->z2*2;
  if (mb[5].mm) memcpy(Zh, mb[5].mm, out_sz);
  for(int i=0;i<6;i++) npu_bo_free(&mb[i]);
  for(int i=0;i<n_filler;i++) npu_bo_free(&filler[i]);
  close(mfd);
  return 0;
}

/* Test: alloc 6 matmul BOs at TOP, then n_extra W BOs BELOW. Point io1 DMA to the
 * LAST extra W BO (below the matmul BOs). If the matmul is correct, the NPU can
 * DMA to a W BO below the matmul BOs -> weight caching below is feasible. */
__attribute__((visibility("default")))
int npu_test_wbelow(int n_extra, int M, int K, int N,
                     const void* Xh, const void* Wh, void* Zh, uint64_t* w_iova_out) {
  const MMTemplate* tpl = NULL;
  for (int i=0;i<MM_NSHAPES;i++){ const MMTemplate* t=&mm_templates[i];
    if (t->x0==1 && t->x1==M && t->x2==K && t->w0==K && t->w1==N){ tpl=t; break; } }
  if (!tpl) return -1;
  int mfd = open(NPU_DEV, O_RDWR);
  if (mfd < 0) return -2;
  npu_action(mfd,0,0xFFFFFFFFu); npu_action(mfd,1,0); npu_action(mfd,19,0xFFFFFFEDu); npu_action(mfd,1,0); npu_action(mfd,18,0);
  /* 6 matmul BOs FIRST (top) */
  uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
  uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  struct npu_bo mb[6]; int ce=0;
  for(int i=0;i<6;i++){ if(npu_create_bo(mfd,msz[i],mfl[i],&mb[i])<0){ ce=1; break; } }
  if (ce){ for(int i=0;i<6;i++) npu_bo_free(&mb[i]); close(mfd); return -3; }
  /* n_extra W BOs BELOW */

  struct npu_bo wb[256];
  for (int i=0;i<n_extra;i++){ if(npu_create_bo(mfd,tpl->sz_w,CREATE_FLAGS_BUF,&wb[i])<0){ n_extra=i; break; } }
  if (n_extra < 1){ for(int i=0;i<6;i++) npu_bo_free(&mb[i]); close(mfd); return -5; }
  *w_iova_out = wb[n_extra-1].dma_addr;
  /* copy W into the last extra W BO (the one we'll read from) */
  if(wb[n_extra-1].mm) memcpy(wb[n_extra-1].mm, Wh, tpl->sz_w);
  /* patch regcmd: io1 (cap[4]) -> the extra W BO, NOT mb[4] */
  uint64_t cap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
  uint64_t my[6]; for(int i=0;i<6;i++) my[i]=mb[i].dma_addr;
  my[4] = wb[n_extra-1].dma_addr;  /* OVERRIDE io1 to the extra W BO below */
  uint32_t sz32[6]; for(int i=0;i<6;i++) sz32[i]=(uint32_t)msz[i];
  if(mb[1].mm) memcpy(mb[1].mm, tpl->regcmd_tmpl, tpl->sz_regcmd<mb[1].size?tpl->sz_regcmd:mb[1].size);
  if(mb[0].mm) memcpy(mb[0].mm, tpl->task_tmpl, tpl->sz_task<mb[0].size?tpl->sz_task:mb[0].size);
  patch_regcmd_tmpl(mb[1].mm, tpl->sz_regcmd, cap, my, sz32, 6);
  patch_task_tmpl(mb[0].mm, tpl->sz_task, cap[1], my[1], tpl->sz_regcmd);
  if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch);
  if(mb[3].mm) memcpy(mb[3].mm, Xh, tpl->sz_in);
  /* mb[4] (the real io1 BO) is NOT written -- W is in wb[n_extra-1] below */
  if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out);
  /* sync all 6 matmul BOs + the W BO we use */
  for(int i=0;i<6;i++) npu_sync_bo(mfd, mb[i].obj_addr, mb[i].size, 1);
  npu_sync_bo(mfd, wb[n_extra-1].obj_addr, wb[n_extra-1].size, 1);
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks;
  *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2]; *(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1]; }
  int sr = ioctl(mfd, IOCTL_SUBMIT, sub);
  if (sr < 0) { for(int i=0;i<6;i++) npu_bo_free(&mb[i]); for(int i=0;i<n_extra;i++) npu_bo_free(&wb[i]); close(mfd); return -4; }
  npu_sync_bo(mfd, mb[5].obj_addr, mb[5].size, 2);
  uint64_t out_sz = (uint64_t)tpl->z0*tpl->z1*tpl->z2*2;
  if (mb[5].mm) memcpy(Zh, mb[5].mm, out_sz);
  for(int i=0;i<6;i++) npu_bo_free(&mb[i]);
  for(int i=0;i<n_extra;i++) npu_bo_free(&wb[i]);
  close(mfd);
  return 0;
}

/* s_wc_* globals moved to the top (before the matmul Execute) */

__attribute__((visibility("default")))
int npu_mm_cache_setup(int n_w, int M, int K, int N) {
  if (s_wc_mfd >= 0) return -99;
  const MMTemplate* tpl = NULL;
  for (int i=0;i<MM_NSHAPES;i++){ const MMTemplate* t=&mm_templates[i];
    if (t->x0==1 && t->x1==M && t->x2==K && t->w0==K && t->w1==N){ tpl=t; break; } }
  if (!tpl) return -1;
  int mfd = open(NPU_DEV, O_RDWR);
  if (mfd < 0) return -2;
  npu_action(mfd,0,0xFFFFFFFFu); npu_action(mfd,1,0); npu_action(mfd,19,0xFFFFFFEDu); npu_action(mfd,1,0); npu_action(mfd,18,0);
  uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
  uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  int ce=0; for(int i=0;i<6;i++){ if(npu_create_bo(mfd,msz[i],mfl[i],&s_wc_mb[i])<0){ ce=1; break; } }
  if (ce){ for(int i=0;i<6;i++) npu_bo_free(&s_wc_mb[i]); close(mfd); return -3; }
  s_wc_wb = (struct npu_bo*)calloc(n_w, sizeof(struct npu_bo));
  if (!s_wc_wb){ for(int i=0;i<6;i++) npu_bo_free(&s_wc_mb[i]); close(mfd); return -4; }
  int got=0; for(int i=0;i<n_w;i++){ if(npu_create_bo(mfd,tpl->sz_w,CREATE_FLAGS_BUF,&s_wc_wb[i])<0) break; got++; }
  if (got < n_w){ for(int i=0;i<got;i++) npu_bo_free(&s_wc_wb[i]); free(s_wc_wb); s_wc_wb=NULL;
    for(int i=0;i<6;i++) npu_bo_free(&s_wc_mb[i]); close(mfd); return -5; }
  s_wc_nw = n_w; s_wc_tpl = tpl; s_wc_mfd = mfd;
  /* patch regcmd with io1 -> wb[0] (so we can find io1 entries) */
  uint64_t cap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
  uint64_t my[6]; for(int i=0;i<6;i++) my[i]=s_wc_mb[i].dma_addr;
  my[4] = s_wc_wb[0].dma_addr;
  uint32_t sz32[6]; for(int i=0;i<6;i++) sz32[i]=(uint32_t)msz[i];
  if(s_wc_mb[1].mm) memcpy(s_wc_mb[1].mm, tpl->regcmd_tmpl, tpl->sz_regcmd<msz[1]?tpl->sz_regcmd:msz[1]);
  if(s_wc_mb[0].mm) memcpy(s_wc_mb[0].mm, tpl->task_tmpl, tpl->sz_task<msz[0]?tpl->sz_task:msz[0]);
  patch_regcmd_tmpl(s_wc_mb[1].mm, tpl->sz_regcmd, cap, my, sz32, 6);
  patch_task_tmpl(s_wc_mb[0].mm, tpl->sz_task, cap[1], my[1], tpl->sz_regcmd);
  /* record io1 DMA entries: scan regcmd for DMA in [wb[0].dma, wb[0].dma+sz_w) */
  uint8_t* rc = s_wc_mb[1].mm; uint64_t w0 = s_wc_wb[0].dma_addr; uint64_t wsz = tpl->sz_w;
  s_wc_n_io1 = 0;
  for (int off=0; off+8 <= (int)msz[1]; off+=8) {
    uint16_t val; memcpy(&val, rc+off+2, 2);
    uint32_t tag; memcpy(&tag, rc+off+4, 4);
    uint32_t cid = tag >> 16;
    if (cid==0x1001 || cid==0x2001 || cid==0x0201) {
      uint64_t dma = ((tag & 0xffff)<<16) | val;
      if (dma >= w0 && dma < w0 + wsz && s_wc_n_io1 < 32) {
        s_wc_io1_off[s_wc_n_io1] = off;
        s_wc_io1_suboff[s_wc_n_io1] = (uint32_t)(dma - w0);
        s_wc_n_io1++;
      }
    }
  }
  return s_wc_n_io1;  /* return #io1 entries found (should be 2) */
}

__attribute__((visibility("default")))
int npu_mm_cache_load(int w_idx, const void* Wh) {
  if (w_idx < 0 || w_idx >= s_wc_nw || s_wc_mfd < 0) return -1;
  if(s_wc_wb[w_idx].mm) memcpy(s_wc_wb[w_idx].mm, Wh, s_wc_tpl->sz_w);
  npu_sync_bo(s_wc_mfd, s_wc_wb[w_idx].obj_addr, s_wc_wb[w_idx].size, 1);
  return 0;
}

__attribute__((visibility("default")))
int npu_mm_cache_run(int w_idx, const void* Xh, void* Zh) {
  if (w_idx < 0 || w_idx >= s_wc_nw || s_wc_mfd < 0) return -1;
  const MMTemplate* tpl = s_wc_tpl;
  uint8_t* rc = s_wc_mb[1].mm;
  uint64_t wbase = s_wc_wb[w_idx].dma_addr;
  /* repoint io1 DMA entries -> this W BO */
  for (int i=0;i<s_wc_n_io1;i++) {
    uint64_t dma = wbase + s_wc_io1_suboff[i];
    int off = s_wc_io1_off[i];
    uint16_t val = (uint16_t)(dma & 0xffff);
    uint16_t tag_lo = (uint16_t)(dma >> 16);
    memcpy(rc+off+2, &val, 2);
    memcpy(rc+off+4, &tag_lo, 2);  /* preserve cid in high 16 of tag */
  }
  { static int zsz=-1; if(zsz<0){ const char*e=getenv("NPU_ZSZ"); zsz=e?atoi(e):0; }
    int zbytes = zsz>0 ? zsz : (int)tpl->sz_scratch;
    if(s_wc_mb[2].mm) memset(s_wc_mb[2].mm,0,zbytes<(int)tpl->sz_scratch?zbytes:(int)tpl->sz_scratch); }
  if(s_wc_mb[3].mm) memcpy(s_wc_mb[3].mm, Xh, tpl->sz_in);
  if(s_wc_mb[5].mm) memset(s_wc_mb[5].mm,0,tpl->sz_out);
  /* sync only changed BOs: regcmd(1, io1 repointed), scratch(2), io0(3), io2(5).
   * task(0) and the W BO are unchanged since cold/load -> no per-call sync. */
  npu_sync_bo(s_wc_mfd, s_wc_mb[1].obj_addr, s_wc_mb[1].size, 1);
  npu_sync_bo(s_wc_mfd, s_wc_mb[2].obj_addr, s_wc_mb[2].size, 1);
  npu_sync_bo(s_wc_mfd, s_wc_mb[3].obj_addr, s_wc_mb[3].size, 1);
  npu_sync_bo(s_wc_mfd, s_wc_mb[5].obj_addr, s_wc_mb[5].size, 1);
  static int _n1c=-1; if(_n1c<0){const char*_e=getenv("NPU_1COPY");_n1c=_e?atoi(_e):0;}
  uint32_t _ts=tpl->sub0_task_start,_nt=tpl->sub0_n_tasks; uint32_t _sc[10];
  for(int si=0;si<10;si++)_sc[si]=tpl->subcore[si];
  if(_n1c&&_nt>=3){uint32_t _b=_nt/3;_ts=_b*2;_nt=_b;_sc[0]=_ts;_sc[1]=_b;for(int si=2;si<10;si++)_sc[si]=0;}
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=_ts; *(uint32_t*)(sub+12)=_nt;
  *(uint32_t*)(sub+16)=_nt; *(uint64_t*)(sub+24)=s_wc_mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){ *(uint32_t*)(sub+64+si*8)=_sc[si*2]; *(uint32_t*)(sub+64+si*8+4)=_sc[si*2+1]; }
  if(ioctl(s_wc_mfd, IOCTL_SUBMIT, sub)<0) return -4;
  npu_sync_bo(s_wc_mfd, s_wc_mb[5].obj_addr, s_wc_mb[5].size, 2);
  uint64_t out_sz = (uint64_t)tpl->z0*tpl->z1*tpl->z2*2;
  if (s_wc_mb[5].mm) memcpy(Zh, s_wc_mb[5].mm, out_sz);
  return 0;
}

__attribute__((visibility("default")))
void npu_mm_cache_close(void) {
  if (s_wc_mfd >= 0) {
    for(int i=0;i<s_wc_nw;i++) npu_bo_free(&s_wc_wb[i]);
    free(s_wc_wb); s_wc_wb=NULL; s_wc_nw=0;
    for(int i=0;i<6;i++) npu_bo_free(&s_wc_mb[i]);
    close(s_wc_mfd); s_wc_mfd=-1;
  }
}

/* === Coexistence test: relu BOs at top + matmul BOs below + W below, one fd.
 *     Returns: bit0 = matmul ok, bit1 = relu ok, bit2 = alt ok. -ve = setup fail. === */
__attribute__((visibility("default")))
int npu_test_coexist(void) {
  const MMTemplate* tpl = &mm_templates[1]; /* mm_768x768 */
  int mfd = open(NPU_DEV, O_RDWR);
  if (mfd < 0) return -1;
  npu_action(mfd,0,0xFFFFFFFFu); npu_action(mfd,1,0); npu_action(mfd,19,0xFFFFFFEDu); npu_action(mfd,1,0); npu_action(mfd,18,0);
  /* 1. relu BOs (top) */
  struct npu_bo rb[5];
  uint64_t rsz[5]={RELU_SZ_TASK,RELU_SZ_REGCMD,RELU_SZ_SCRATCH,RELU_SZ_IN,RELU_SZ_OUT};
  uint32_t rfl[5]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  for(int i=0;i<5;i++) if(npu_create_bo(mfd,rsz[i],rfl[i],&rb[i])<0){ for(int j=0;j<i;j++) npu_bo_free(&rb[j]); close(mfd); return -2; }
  fprintf(stderr,"relu BOs: t=%llx rc=%llx sc=%llx in=%llx out=%llx\n",
    (unsigned long long)rb[0].dma_addr,(unsigned long long)rb[1].dma_addr,(unsigned long long)rb[2].dma_addr,
    (unsigned long long)rb[3].dma_addr,(unsigned long long)rb[4].dma_addr);
  /* 2. matmul BOs (below) */
  struct npu_bo mb[6];
  uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
  uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  for(int i=0;i<6;i++) if(npu_create_bo(mfd,msz[i],mfl[i],&mb[i])<0){ for(int j=0;j<i;j++) npu_bo_free(&mb[j]); for(int j=0;j<5;j++) npu_bo_free(&rb[j]); close(mfd); return -3; }
  fprintf(stderr,"mm BOs: t=%llx rc=%llx sc=%llx in=%llx w=%llx out=%llx\n",
    (unsigned long long)mb[0].dma_addr,(unsigned long long)mb[1].dma_addr,(unsigned long long)mb[2].dma_addr,
    (unsigned long long)mb[3].dma_addr,(unsigned long long)mb[4].dma_addr,(unsigned long long)mb[5].dma_addr);
  /* 3. W BO (below matmul) */
  struct npu_bo wb; if(npu_create_bo(mfd,tpl->sz_w,CREATE_FLAGS_BUF,&wb)<0){ for(int i=0;i<6;i++) npu_bo_free(&mb[i]); for(int i=0;i<5;i++) npu_bo_free(&rb[i]); close(mfd); return -4; }
  fprintf(stderr,"W BO: %llx (below mm by %llx)\n",(unsigned long long)wb.dma_addr,(unsigned long long)(mb[4].dma_addr-wb.dma_addr));
  /* patch matmul (io1 -> W BO) */
  uint64_t mcap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
  uint64_t mmy[6]; for(int i=0;i<6;i++) mmy[i]=mb[i].dma_addr; mmy[4]=wb.dma_addr;
  uint32_t msz32[6]; for(int i=0;i<6;i++) msz32[i]=(uint32_t)msz[i];
  memcpy(mb[1].mm,tpl->regcmd_tmpl,tpl->sz_regcmd<msz[1]?tpl->sz_regcmd:msz[1]);
  memcpy(mb[0].mm,tpl->task_tmpl,tpl->sz_task<msz[0]?tpl->sz_task:msz[0]);
  patch_regcmd_tmpl(mb[1].mm,tpl->sz_regcmd,mcap,mmy,msz32,6);
  patch_task_tmpl(mb[0].mm,tpl->sz_task,mcap[1],mmy[1],tpl->sz_regcmd);
  npu_sync_bo(mfd,mb[1].obj_addr,mb[1].size,1); npu_sync_bo(mfd,mb[0].obj_addr,mb[0].size,1);
  /* patch relu */
  uint64_t rcap[5]={RELU_CAP_TASK,RELU_CAP_REGCMD,RELU_CAP_SCRATCH,RELU_CAP_IN,RELU_CAP_OUT};
  uint64_t rmy[5]; for(int i=0;i<5;i++) rmy[i]=rb[i].dma_addr;
  uint32_t rsz32[5]={RELU_SZ_TASK,RELU_SZ_REGCMD,RELU_SZ_SCRATCH,RELU_SZ_IN,RELU_SZ_OUT};
  memcpy(rb[1].mm,g_relu_regcmd,RELU_REGCMD_SZ<rb[1].size?RELU_REGCMD_SZ:rb[1].size);
  memcpy(rb[0].mm,g_relu_task,RELU_TASK_SZ<rb[0].size?RELU_TASK_SZ:rb[0].size);
  patch_regcmd_tmpl(rb[1].mm,RELU_REGCMD_SZ,rcap,rmy,rsz32,5);
  patch_task_tmpl(rb[0].mm,RELU_TASK_SZ,rcap[1],rmy[1],RELU_SZ_REGCMD);
  npu_sync_bo(mfd,rb[1].obj_addr,rb[1].size,1); npu_sync_bo(mfd,rb[0].obj_addr,rb[0].size,1);
  /* fill X, W (fp16) */
  uint16_t* Xh=(uint16_t*)malloc(tpl->sz_in); uint16_t* Wh=(uint16_t*)malloc(tpl->sz_w);
  for(int i=0;i<tpl->sz_in/2;i++) Xh[i]=0x3c00;  /* 1.0 fp16 */
  for(int i=0;i<tpl->sz_w/2;i++) Wh[i]=0x3c00;
  memcpy(wb.mm,Wh,tpl->sz_w); npu_sync_bo(mfd,wb.obj_addr,wb.size,1);
  /* ---- matmul run ---- */
  int mm_ok=0, relu_ok=0, alt_ok=0;
  #define MM_RUN(lbl) do { \
    if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch); \
    if(mb[3].mm) memcpy(mb[3].mm,Xh,tpl->sz_in); \
    if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out); \
    npu_sync_bo(mfd,mb[2].obj_addr,mb[2].size,1); npu_sync_bo(mfd,mb[3].obj_addr,mb[3].size,1); npu_sync_bo(mfd,mb[5].obj_addr,mb[5].size,1); \
    uint8_t sub[104]; memset(sub,0,104); \
    *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks; \
    *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1; \
    for(int si=0;si<5;si++){*(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2];*(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1];} \
    int r=ioctl(mfd,IOCTL_SUBMIT,sub); fprintf(stderr,lbl" submit=%d ",r); \
    if(r<0){ fprintf(stderr,"MM HANG\n"); } else { npu_sync_bo(mfd,mb[5].obj_addr,mb[5].size,2); mm_ok=1; } } while(0)
  #define RELU_RUN(lbl) do { \
    if(rb[2].mm) memset(rb[2].mm,0,RELU_SZ_SCRATCH); \
    if(rb[3].mm){ memset(rb[3].mm,0,RELU_SZ_IN); memcpy(rb[3].mm,Xh,RELU_SZ_IN<tpl->sz_in?RELU_SZ_IN:tpl->sz_in); } \
    if(rb[4].mm) memset(rb[4].mm,0,RELU_SZ_OUT); \
    npu_sync_bo(mfd,rb[2].obj_addr,rb[2].size,1); npu_sync_bo(mfd,rb[3].obj_addr,rb[3].size,1); npu_sync_bo(mfd,rb[4].obj_addr,rb[4].size,1); \
    uint8_t sub[104]; memset(sub,0,104); \
    *(uint32_t*)(sub+0)=RELU_FLAGS; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=RELU_TASK_START; *(uint32_t*)(sub+12)=RELU_N_TASKS; \
    *(uint32_t*)(sub+16)=RELU_N_TASKS; *(uint64_t*)(sub+24)=rb[0].obj_addr; *(uint32_t*)(sub+56)=1; *(int32_t*)(sub+60)=-1; \
    for(int si=0;si<5;si++){*(uint32_t*)(sub+64+si*8)=RELU_SUBCORE[si*2];*(uint32_t*)(sub+64+si*8+4)=RELU_SUBCORE[si*2+1];} \
    int r=ioctl(mfd,IOCTL_SUBMIT,sub); fprintf(stderr,lbl" submit=%d ",r); \
    if(r<0){ fprintf(stderr,"RELU HANG\n"); } else { npu_sync_bo(mfd,rb[4].obj_addr,rb[4].size,2); relu_ok=1; } } while(0)
  MM_RUN("[mm1]"); fprintf(stderr,"mm_ok=%d\n",mm_ok);
  RELU_RUN("[relu1]"); fprintf(stderr,"relu_ok=%d\n",relu_ok);
  /* alternate 3x */
  for(int a=0;a<3 && mm_ok && relu_ok;a++){ MM_RUN("[mmA]"); RELU_RUN("[reluA]"); }
  if(mm_ok&&relu_ok) alt_ok=1;
  /* verify mm correctness (1 elem) */
  uint16_t z0; { uint16_t* z=(uint16_t*)mb[5].mm; z0=z[0]; }
  fprintf(stderr,"coexist: mm=%d relu=%d alt=%d z0=0x%04x (expect 0x6200=768.0)\n",mm_ok,relu_ok,alt_ok,z0);
  free(Xh); free(Wh);
  for(int i=0;i<6;i++) npu_bo_free(&mb[i]); for(int i=0;i<5;i++) npu_bo_free(&rb[i]); npu_bo_free(&wb); close(mfd);
  return (mm_ok?1:0)|(relu_ok?2:0)|(alt_ok?4:0);
}

/* === Test: push ONE mm BO low (alloc it last), keep others at top.
 *     which: 2=scratch, 3=in, 5=out. Returns submit result (0=ok, -1=hang). === */
__attribute__((visibility("default")))
int npu_test_bo_low(int which) {
  const MMTemplate* tpl = &mm_templates[1]; /* mm_768x768 */
  int mfd = open(NPU_DEV, O_RDWR);
  if (mfd < 0) return -100;
  npu_action(mfd,0,0xFFFFFFFFu); npu_action(mfd,1,0); npu_action(mfd,19,0xFFFFFFEDu); npu_action(mfd,1,0); npu_action(mfd,18,0);
  struct npu_bo mb[6];
  uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
  uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  /* alloc all except `which` first (top), then `which` last (low) */
  int order[6]; int n=0;
  for(int i=0;i<6;i++) if(i!=which) order[n++]=i;
  order[n++]=which;
  for(int k=0;k<6;k++){ int i=order[k]; if(npu_create_bo(mfd,msz[i],mfl[i],&mb[i])<0){ for(int j=0;j<k;j++) npu_bo_free(&mb[order[j]]); close(mfd); return -101; } }
  fprintf(stderr,"bo_low which=%d: t=%llx rc=%llx sc=%llx in=%llx w=%llx out=%llx (low=%llx is %s)\\n",
    which,(unsigned long long)mb[0].dma_addr,(unsigned long long)mb[1].dma_addr,(unsigned long long)mb[2].dma_addr,
    (unsigned long long)mb[3].dma_addr,(unsigned long long)mb[4].dma_addr,(unsigned long long)mb[5].dma_addr,
    (unsigned long long)mb[which].dma_addr, which==2?"scratch":which==3?"in":which==5?"out":"?");
  uint64_t mcap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
  uint64_t mmy[6]; for(int i=0;i<6;i++) mmy[i]=mb[i].dma_addr;
  uint32_t msz32[6]; for(int i=0;i<6;i++) msz32[i]=(uint32_t)msz[i];
  memcpy(mb[1].mm,tpl->regcmd_tmpl,tpl->sz_regcmd<msz[1]?tpl->sz_regcmd:msz[1]);
  memcpy(mb[0].mm,tpl->task_tmpl,tpl->sz_task<msz[0]?tpl->sz_task:msz[0]);
  patch_regcmd_tmpl(mb[1].mm,tpl->sz_regcmd,mcap,mmy,msz32,6);
  patch_task_tmpl(mb[0].mm,tpl->sz_task,mcap[1],mmy[1],tpl->sz_regcmd);
  npu_sync_bo(mfd,mb[1].obj_addr,mb[1].size,1); npu_sync_bo(mfd,mb[0].obj_addr,mb[0].size,1);
  /* fill X, W with 1.0 */
  for(int i=0;i<tpl->sz_in/2;i++) ((uint16_t*)mb[3].mm)[i]=0x3c00;
  for(int i=0;i<tpl->sz_w/2;i++) ((uint16_t*)mb[4].mm)[i]=0x3c00;
  if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch);
  if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out);
  for(int i=0;i<6;i++) npu_sync_bo(mfd,mb[i].obj_addr,mb[i].size,1);
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks;
  *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){*(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2];*(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1];}
  int r=ioctl(mfd,IOCTL_SUBMIT,sub);
  if(r>=0){ npu_sync_bo(mfd,mb[5].obj_addr,mb[5].size,2); uint16_t z0=((uint16_t*)mb[5].mm)[0]; fprintf(stderr,"  ok z0=0x%04x (expect 0x6200)\\n",z0); }
  else fprintf(stderr,"  HANG\\n");
  for(int i=0;i<6;i++) npu_bo_free(&mb[i]); close(mfd);
  return r;
}

/* fp16 -> fp32 soft conversion (no __builtin_float16 on this gcc) */
static float fp16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  if (exp == 0) {
    if (mant == 0) { uint32_t b = sign; float f; memcpy(&f,&b,4); return f; }
    exp = 1; while (!(mant & 0x400)) { mant <<= 1; if (exp) exp--; } mant &= 0x3ff;
    uint32_t b = sign | ((exp + 112) << 23) | (mant << 13); float f; memcpy(&f,&b,4); return f;
  }
  if (exp == 0x1f) { uint32_t b = sign | (0xffu << 23) | (mant << 13); float f; memcpy(&f,&b,4); return f; }
  uint32_t b = sign | ((exp + 112) << 23) | (mant << 13); float f; memcpy(&f,&b,4); return f;
}

/* fused: run matmul, read fp16 Z, convert to fp32, add bias. Zh32 = X@W + b.
 * Zh32 is caller-provided fp32 buffer (n_elem*4 bytes). Returns 0 ok. */
__attribute__((visibility("default")))
int npu_mm_cache_run_bias(int w_idx, const void* Xh, const float* bh, float* Zh32) {
  if (w_idx < 0 || w_idx >= s_wc_nw || s_wc_mfd < 0) return -1;
  const MMTemplate* tpl = s_wc_tpl;
  uint8_t* rc = s_wc_mb[1].mm; uint64_t wbase = s_wc_wb[w_idx].dma_addr;
  for (int i=0;i<s_wc_n_io1;i++) {
    uint64_t dma = wbase + s_wc_io1_suboff[i]; int off = s_wc_io1_off[i];
    uint16_t val=(uint16_t)(dma&0xffff), tl=(uint16_t)(dma>>16);
    memcpy(rc+off+2,&val,2); memcpy(rc+off+4,&tl,2);
  }
  if(s_wc_mb[2].mm) memset(s_wc_mb[2].mm,0,tpl->sz_scratch);
  if(s_wc_mb[3].mm) memcpy(s_wc_mb[3].mm,Xh,tpl->sz_in);
  if(s_wc_mb[5].mm) memset(s_wc_mb[5].mm,0,tpl->sz_out);
  npu_sync_bo(s_wc_mfd,s_wc_mb[1].obj_addr,s_wc_mb[1].size,1);
  npu_sync_bo(s_wc_mfd,s_wc_mb[2].obj_addr,s_wc_mb[2].size,1);
  npu_sync_bo(s_wc_mfd,s_wc_mb[3].obj_addr,s_wc_mb[3].size,1);
  npu_sync_bo(s_wc_mfd,s_wc_mb[5].obj_addr,s_wc_mb[5].size,1);
  static int _n1cb=-1; if(_n1cb<0){const char*_e=getenv("NPU_1COPY");_n1cb=_e?atoi(_e):0;}
  uint32_t _ts=tpl->sub0_task_start,_nt=tpl->sub0_n_tasks; uint32_t _sc[10];
  for(int si=0;si<10;si++)_sc[si]=tpl->subcore[si];
  if(_n1cb&&_nt>=3){uint32_t _b=_nt/3;_ts=_b*2;_nt=_b;_sc[0]=_ts;_sc[1]=_b;for(int si=2;si<10;si++)_sc[si]=0;}
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=_ts; *(uint32_t*)(sub+12)=_nt;
  *(uint32_t*)(sub+16)=_nt; *(uint64_t*)(sub+24)=s_wc_mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){*(uint32_t*)(sub+64+si*8)=_sc[si*2];*(uint32_t*)(sub+64+si*8+4)=_sc[si*2+1];}
  if(ioctl(s_wc_mfd,IOCTL_SUBMIT,sub)<0) return -4;
  npu_sync_bo(s_wc_mfd,s_wc_mb[5].obj_addr,s_wc_mb[5].size,2);
  int n = (tpl->z1)*(tpl->z2);  /* elements per row * rows = z1*z2 (z0=1) */
  int z2 = tpl->z2;  /* bias broadcasts over the last dim (z2) */
  uint16_t* z = (uint16_t*)s_wc_mb[5].mm;
  for (int i=0;i<n;i++) Zh32[i] = fp16_to_f32(z[i]) + (bh ? bh[i % z2] : 0.0f);
  return 0;
}

/* fused: Zh32 = X@W + b + res. For residual add (x = x + o + b). */
__attribute__((visibility("default")))
int npu_mm_cache_run_bias_res(int w_idx, const void* Xh, const float* bh, const float* resh, float* Zh32) {
  int r = npu_mm_cache_run_bias(w_idx, Xh, bh, Zh32);
  if (r) return r;
  const MMTemplate* tpl = s_wc_tpl;
  int n = (tpl->z1)*(tpl->z2);
  if (resh) for (int i=0;i<n;i++) Zh32[i] += resh[i];
  return 0;
}

/* === Cross-fd DMA test: io0 (X) lives on fd1, kernel+io1+io2 on fd2.
 *     Submit on fd2. Does the NPU read X from fd1's BO? Returns z0 (0x6200=ok). === */
__attribute__((visibility("default")))
int npu_test_crossfd(void) {
  const MMTemplate* tpl = &mm_templates[1]; /* mm_768x768 */
  int fd1 = open(NPU_DEV, O_RDWR);
  if (fd1 < 0) return -1;
  npu_action(fd1,0,0xFFFFFFFFu); npu_action(fd1,1,0); npu_action(fd1,19,0xFFFFFFEDu); npu_action(fd1,1,0); npu_action(fd1,18,0);
  struct npu_bo Xbo;  /* io0 on fd1 */
  if(npu_create_bo(fd1,tpl->sz_in,CREATE_FLAGS_BUF,&Xbo)<0){ close(fd1); return -2; }
  fprintf(stderr,"crossfd: X(io0) on fd1 @ %llx\n",(unsigned long long)Xbo.dma_addr);
  /* fill X with 1.0 */
  for(int i=0;i<tpl->sz_in/2;i++) ((uint16_t*)Xbo.mm)[i]=0x3c00;
  npu_sync_bo(fd1,Xbo.obj_addr,Xbo.size,1);
  /* fd2: kernel BOs (task,regcmd,scratch,io1=W,io2=out) -- io0 points to fd1's X */
  int fd2 = open(NPU_DEV, O_RDWR);
  if (fd2 < 0) { npu_bo_free(&Xbo); close(fd1); return -3; }
  npu_action(fd2,0,0xFFFFFFFFu); npu_action(fd2,1,0); npu_action(fd2,19,0xFFFFFFEDu); npu_action(fd2,1,0); npu_action(fd2,18,0);
  struct npu_bo mb[6];
  uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
  uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  /* alloc task,regcmd,scratch,io1,io2 on fd2 (skip io0 index 3 -- use fd1's X) */
  int order[5]={0,1,2,4,5}; for(int k=0;k<5;k++){ int i=order[k]; if(npu_create_bo(fd2,msz[i],mfl[i],&mb[i])<0){ for(int j=0;j<k;j++) npu_bo_free(&mb[order[j]]); close(fd2); npu_bo_free(&Xbo); close(fd1); return -4; } }
  fprintf(stderr,"crossfd: kernel on fd2: t=%llx rc=%llx sc=%llx w=%llx out=%llx\n",
    (unsigned long long)mb[0].dma_addr,(unsigned long long)mb[1].dma_addr,(unsigned long long)mb[2].dma_addr,
    (unsigned long long)mb[4].dma_addr,(unsigned long long)mb[5].dma_addr);
  /* patch: io0 -> fd1's X; others -> fd2's BOs */
  uint64_t mcap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
  uint64_t mmy[6]; for(int i=0;i<6;i++) mmy[i]=mb[i].dma_addr; mmy[3]=Xbo.dma_addr;  /* io0 -> fd1 */
  uint32_t msz32[6]; for(int i=0;i<6;i++) msz32[i]=(uint32_t)msz[i];
  memcpy(mb[1].mm,tpl->regcmd_tmpl,tpl->sz_regcmd<msz[1]?tpl->sz_regcmd:msz[1]);
  memcpy(mb[0].mm,tpl->task_tmpl,tpl->sz_task<msz[0]?tpl->sz_task:msz[0]);
  patch_regcmd_tmpl(mb[1].mm,tpl->sz_regcmd,mcap,mmy,msz32,6);
  patch_task_tmpl(mb[0].mm,tpl->sz_task,mcap[1],mmy[1],tpl->sz_regcmd);
  /* fill W (io1) with 1.0, zero scratch+out */
  for(int i=0;i<tpl->sz_w/2;i++) ((uint16_t*)mb[4].mm)[i]=0x3c00;
  if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch);
  if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out);
  /* sync: regcmd+task+scratch+W+out on fd2; X on fd1 */
  npu_sync_bo(fd2,mb[1].obj_addr,mb[1].size,1); npu_sync_bo(fd2,mb[0].obj_addr,mb[0].size,1);
  npu_sync_bo(fd2,mb[2].obj_addr,mb[2].size,1); npu_sync_bo(fd2,mb[4].obj_addr,mb[4].size,1); npu_sync_bo(fd2,mb[5].obj_addr,mb[5].size,1);
  /* submit on fd2 */
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks;
  *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){*(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2];*(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1];}
  int r=ioctl(fd2,IOCTL_SUBMIT,sub);
  fprintf(stderr,"crossfd: submit=%d ",r);
  if(r<0){ fprintf(stderr,"HANG\n"); for(int i=0;i<5;i++) npu_bo_free(&mb[order[i]]); close(fd2); npu_bo_free(&Xbo); close(fd1); return -5; }
  npu_sync_bo(fd2,mb[5].obj_addr,mb[5].size,2);
  uint16_t z0=((uint16_t*)mb[5].mm)[0];
  fprintf(stderr,"ok z0=0x%04x (expect 0x6200=768.0)\n",z0);
  for(int i=0;i<5;i++) npu_bo_free(&mb[order[i]]); close(fd2); npu_bo_free(&Xbo); close(fd1);
  return z0;
}

/* === Mimic the jax cold path: open fd1, alloc X+W (device buffers),
 *     free them, close fd1, open fd2 (mfd), alloc 6 matmul BOs, submit. === */
__attribute__((visibility("default")))
int npu_test_coldpath(void) {
  const MMTemplate* tpl = &mm_templates[1]; /* mm_768x768 */
  int fd1 = open(NPU_DEV, O_RDWR);
  if (fd1 < 0) return -1;
  npu_action(fd1,0,0xFFFFFFFFu); npu_action(fd1,1,0); npu_action(fd1,19,0xFFFFFFEDu); npu_action(fd1,1,0); npu_action(fd1,18,0);
  struct npu_bo Xb, Wb;
  if(npu_create_bo(fd1,tpl->sz_in,CREATE_FLAGS_BUF,&Xb)<0){close(fd1);return -2;}
  if(npu_create_bo(fd1,tpl->sz_w,CREATE_FLAGS_BUF,&Wb)<0){npu_bo_free(&Xb);close(fd1);return -3;}
  fprintf(stderr,"cold: fd1 X=%llx W=%llx -> free+close\n",(unsigned long long)Xb.dma_addr,(unsigned long long)Wb.dma_addr);
  npu_bo_free(&Xb); npu_bo_free(&Wb); close(fd1);
  int fd2 = open(NPU_DEV, O_RDWR);
  if (fd2 < 0) return -4;
  npu_action(fd2,0,0xFFFFFFFFu); npu_action(fd2,1,0); npu_action(fd2,19,0xFFFFFFEDu); npu_action(fd2,1,0); npu_action(fd2,18,0);
  struct npu_bo mb[6];
  uint64_t msz[6]={tpl->sz_task,tpl->sz_regcmd,tpl->sz_scratch,tpl->sz_in,tpl->sz_w,tpl->sz_out};
  uint32_t mfl[6]={CREATE_FLAGS_TASK,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF,CREATE_FLAGS_BUF};
  for(int i=0;i<6;i++) if(npu_create_bo(fd2,msz[i],mfl[i],&mb[i])<0){for(int j=0;j<i;j++)npu_bo_free(&mb[j]);close(fd2);return -5;}
  fprintf(stderr,"cold: fd2 mm t=%llx sc=%llx out=%llx\n",(unsigned long long)mb[0].dma_addr,(unsigned long long)mb[2].dma_addr,(unsigned long long)mb[5].dma_addr);
  uint64_t mcap[6]={tpl->cap_task,tpl->cap_regcmd,tpl->cap_scratch,tpl->cap_in,tpl->cap_w,tpl->cap_out};
  uint64_t mmy[6]; for(int i=0;i<6;i++) mmy[i]=mb[i].dma_addr;
  uint32_t msz32[6]; for(int i=0;i<6;i++) msz32[i]=(uint32_t)msz[i];
  memcpy(mb[1].mm,tpl->regcmd_tmpl,tpl->sz_regcmd<msz[1]?tpl->sz_regcmd:msz[1]);
  memcpy(mb[0].mm,tpl->task_tmpl,tpl->sz_task<msz[0]?tpl->sz_task:msz[0]);
  patch_regcmd_tmpl(mb[1].mm,tpl->sz_regcmd,mcap,mmy,msz32,6);
  patch_task_tmpl(mb[0].mm,tpl->sz_task,mcap[1],mmy[1],tpl->sz_regcmd);
  for(int i=0;i<tpl->sz_in/2;i++) ((uint16_t*)mb[3].mm)[i]=0x3c00;
  for(int i=0;i<tpl->sz_w/2;i++) ((uint16_t*)mb[4].mm)[i]=0x3c00;
  if(mb[2].mm) memset(mb[2].mm,0,tpl->sz_scratch);
  if(mb[5].mm) memset(mb[5].mm,0,tpl->sz_out);
  for(int i=0;i<6;i++) npu_sync_bo(fd2,mb[i].obj_addr,mb[i].size,1);
  uint8_t sub[104]; memset(sub,0,104);
  *(uint32_t*)(sub+0)=tpl->sub0_flags; *(uint32_t*)(sub+4)=6000; *(uint32_t*)(sub+8)=tpl->sub0_task_start; *(uint32_t*)(sub+12)=tpl->sub0_n_tasks;
  *(uint32_t*)(sub+16)=tpl->sub0_n_tasks; *(uint64_t*)(sub+24)=mb[0].obj_addr; *(uint32_t*)(sub+56)=tpl->core_mask; *(int32_t*)(sub+60)=-1;
  for(int si=0;si<5;si++){*(uint32_t*)(sub+64+si*8)=tpl->subcore[si*2];*(uint32_t*)(sub+64+si*8+4)=tpl->subcore[si*2+1];}
  int r=ioctl(fd2,IOCTL_SUBMIT,sub);
  fprintf(stderr,"cold: submit=%d ",r);
  if(r<0){ fprintf(stderr,"FAIL errno=%d\n",(*__errno_location())); for(int i=0;i<6;i++) npu_bo_free(&mb[i]); close(fd2); return -6; }
  npu_sync_bo(fd2,mb[5].obj_addr,mb[5].size,2);
  uint16_t z0=((uint16_t*)mb[5].mm)[0];
  fprintf(stderr,"ok z0=0x%04x (expect 0x6200)\n",z0);
  for(int i=0;i<6;i++) npu_bo_free(&mb[i]); close(fd2);
  return z0;
}
