# Kernel Patches for RK3588 NPU

These patches modify the rknpu driver in the Rockchip Linux kernel
(develop-6.1 branch) to support features needed by OpenNPU:

## 0001-ioctl-add-batch-submit-struct.patch
Adds `RKNPU_BATCH_SUBMIT` ioctl (0x06) and `struct rknpu_batch_submit`
to the ioctl header. Allows submitting up to 3 NPU jobs in one syscall.

## 0002-iommu-lock-free-domain-fast-path.patch
Optimizes `rknpu_iommu_domain_get_and_switch()`: when the requested
IOMMU domain is already active, uses `atomic_inc` without taking the
`domain_lock` mutex. Falls back to mutex for domain switches.

## 0003-drv-batch-submit-ioctl-DRM_UNLOCKED.patch
Registers the batch submit ioctl in the DRM ioctl table with
`DRM_UNLOCKED` flag. Adds `__rknpu_batch_submit_ioctl` wrapper.
Also adds `DRM_UNLOCKED` to `RKNPU_SUBMIT` (redundant — rknpu is not
`DRIVER_LEGACY`, so the DRM mutex is never held regardless).

## 0004-job-batch-submit-parallel-dispatch-kmem-cache.patch
Implements `rknpu_batch_submit()`: allocates all jobs upfront, takes
`irq_lock` once to add all jobs to todo_lists, then commits all jobs
to hardware back-to-back with no locks held. Adds `kmem_cache` for
`rknpu_job` pre-allocation. Initializes cache in `rknpu_init()`.

## Applying

```bash
# From the kernel source tree (drivers/rknpu/)
cd linux-rockchip/drivers/rknpu/
patch -p1 < 0001-ioctl-add-batch-submit-struct.patch
patch -p1 < 0002-iommu-lock-free-domain-fast-path.patch
patch -p1 < 0003-drv-batch-submit-ioctl-DRM_UNLOCKED.patch
patch -p1 < 0004-job-batch-submit-parallel-dispatch-kmem-cache.patch
```

Also enable in `.config`:
```
CONFIG_ROCKCHIP_RKNPU_FENCE=y
```

## Kernel version

Tested on Rockchip linux-rockchip develop-6.1 branch (6.1.141).
Built and running on Orange Pi 5 Max (RK3588).
