# OpenNPU Makefile — build the PJRT C plugin for JAX

PLUGIN_DIR = src/opennpu/pjrt_c
PLUGIN = $(PLUGIN_DIR)/libpjrt_npu.so
CC = gcc
CFLAGS = -shared -fPIC -O3 -ffast-math -I$(PLUGIN_DIR)
LDFLAGS = -lpthread

.PHONY: all plugin full clean test

all: plugin

# Basic plugin (elementwise + matmul, no GPT-2 forward)
plugin: $(PLUGIN_DIR)/pjrt_npu.c $(PLUGIN_DIR)/pjrt_npu_impl.c
	cd $(PLUGIN_DIR) && python3 gen_plugin.py
	$(CC) $(CFLAGS) -o $@ $(PLUGIN_DIR)/pjrt_npu.c $(LDFLAGS)

# Full plugin (adds GPT-2 npu_lm_forward with OpenMP + GELU overlap thread)
# lm_forward.c is concatenated to the end of pjrt_npu_impl.c because it relies
# on types/decls from the impl file (it was designed as an append, not a TU)
full: $(PLUGIN_DIR)/pjrt_npu.c $(PLUGIN_DIR)/pjrt_npu_impl.c $(PLUGIN_DIR)/cna_matmul.c $(PLUGIN_DIR)/lm_forward.c $(PLUGIN_DIR)/vision_forward.c
	cd $(PLUGIN_DIR) && python3 gen_plugin.py
	cat $(PLUGIN_DIR)/pjrt_npu_impl.c $(PLUGIN_DIR)/cna_matmul.c $(PLUGIN_DIR)/lm_forward.c $(PLUGIN_DIR)/vision_forward.c > $(PLUGIN_DIR)/_impl_full.c
	mv $(PLUGIN_DIR)/pjrt_npu_impl.c $(PLUGIN_DIR)/pjrt_npu_impl_orig.c
	cp $(PLUGIN_DIR)/_impl_full.c $(PLUGIN_DIR)/pjrt_npu_impl.c
	$(CC) $(CFLAGS) -fopenmp -o $(PLUGIN) $(PLUGIN_DIR)/pjrt_npu.c $(LDFLAGS)
	mv $(PLUGIN_DIR)/pjrt_npu_impl_orig.c $(PLUGIN_DIR)/pjrt_npu_impl.c
	rm -f $(PLUGIN_DIR)/_impl_full.c

test: $(PLUGIN)
	JAX_PLATFORMS=npu,cpu \
	PJRT_NAMES_AND_LIBRARY_PATHS=npu:$(PLUGIN) \
	python3 -c 'import jax; print(jax.devices())'

clean:
	rm -f $(PLUGIN) $(PLUGIN_DIR)/pjrt_npu.c $(PLUGIN_DIR)/_impl_full.c