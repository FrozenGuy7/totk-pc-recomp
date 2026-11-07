// AMD FSR 4 (ML) upscaler (vk_fsr4.c), the third engine of vk_dlaa.c's DLAA / DLSS point beside NVIDIA DLSS and FSR 3.1:
// the same scene colour, depth, motion vectors and jitter go to the FSR Upscaling provider of AMD's FSR SDK 2.4, loaded
// at run time from the signed amd_fidelityfx_loader_vk.dll and amd_fidelityfx_upscaler_vk.dll next to the executable.
#pragma once
#include <stdint.h>

#include "vk.h"

typedef struct Fsr4 Fsr4;

// aa_engine = fsr4 with DLAA on: vk_device.c then enables the device features the ML model needs (cooperative matrix,
// float8) when the GPU has them. Read before the device exists, so a change takes a restart.
int vk_fsr4_wanted(void);
// The device extensions FSR 4 needs (VK_KHR_cooperative_matrix, VK_EXT_shader_float8), added to ext[*n] when wanted and
// both present (at most max entries in all). *added is 1 when they were: the caller then chains their feature structs.
void vk_fsr4_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max,
                         int *added);

// 1 once FSR 4 could not be used this run (no DLLs, no provider for this GPU, a failed context): vk_dlaa.c then runs
// FSR 3.1 in its place.
int fsr4_failed(void);

// One context per scene size w x h and output size ow x oh, kept by vk_dlaa.c with that size's work images: mv (RG16F
// motion vectors at the scene size, render pixels, this frame's position to last frame's) and out (RGBA16F at the
// output size), both in VK_IMAGE_LAYOUT_GENERAL. NULL when the context cannot be made; the reason is logged.
Fsr4 *fsr4_create(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, VkImage mv, VkImage out);
// After the GPU finished every frame recorded with it (vk_dlaa.c free_size waits for the device first).
void fsr4_destroy(Fsr4 *f);

typedef struct {
    VkImage color, depth;   // the scene's colour (mip 0, layer 0; any float colour format a blit reads) and its D32F
                            // depth, both in VK_IMAGE_LAYOUT_GENERAL
    float jx, jy;           // the jitter the scene was drawn with, render pixels
    int reset;              // start a new history
} Fsr4Frame;

// Records the upscale into cb, writing the context's out image (no render pass open; the recorder drained). 1 on success.
int fsr4_dispatch(Fsr4 *f, VkCommandBuffer cb, const Fsr4Frame *in);
