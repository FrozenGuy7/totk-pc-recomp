// AMD FSR 4 (ML) upscaler for vk_dlaa.c's DLAA / DLSS point (vk_fsr4.h), through the public C interface (ffx-api) of
// AMD's FSR SDK 2.4. Nothing of AMD's is linked: amd_fidelityfx_loader_vk.dll is loaded at the first context and finds
// amd_fidelityfx_upscaler_vk.dll (the FSR Upscaling provider, with the ML models) beside itself. Both are AMD's signed
// binaries from the SDK's Kits/FidelityFX/signedbin, redistributable under its licence; the headers this file includes
// come from the same SDK (Kits/FidelityFX/api/include, upscalers/include) into the deps folder's include/ffx.
//
// The provider decides which upscaler runs: FSR 4.1 (the FP8 model on RDNA 4, INT8 models on RDNA 2/3 and some Ryzen
// APUs) where the GPU and driver support it, else an older FSR 3 version. The FP8 model needs the device features
// cooperativeMatrix and shaderFloat8, which vk_device.c enables when aa_engine is fsr4 at start (vk_fsr4_device_exts).
//
// Inputs and conventions are those of vk_fsr3.c (the FSR 3.1 bridge) and of DLSS: the scene's light before the bloom
// (linear, HDR), its depth (OpenGL range, not inverted, finite far plane), motion vectors in render pixels from this
// frame's position to last frame's (scale 1) and the jitter the scene was drawn with. The colour and depth are copied
// into an RGBA16F and an R32F image of this context first, as vk_fsr3.c does, so FSR reads images whose formats and
// usage this file controls. FSR wants its inputs in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: the inputs move there
// around the dispatch and back to GENERAL, the layout every image of the back end lives in; the output stays in GENERAL
// (FSR's unordered-access state). The provider records plain Vulkan commands into the command buffer and binds its own
// pipelines and descriptor sets: vk_dlaa.c drains the recorder before and makes the next draw bind everything again.
#include <windows.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_fsr4.h"

int vk_fsr4_wanted(void) {
    return cfg_bool("TOTK_DLAA") && !strcmp(cfg_value("TOTK_AA_ENGINE"), "fsr4");
}

static int has(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

void vk_fsr4_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max,
                         int *added) {
    *added = 0;
    if (!vk_fsr4_wanted()) return;
    int coop = has(have, nhave, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    int f8 = has(have, nhave, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    if (coop && f8 && *n + 2 <= max) {
        ext[(*n)++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        ext[(*n)++] = VK_EXT_SHADER_FLOAT8_EXTENSION_NAME;
        *added = 1;
    } else {
        rt_log("fsr4: the device lacks %s%s%s: FSR 4's FP8 model cannot run (the provider may pick another version)",
               coop ? "" : VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, !coop && !f8 ? " and " : "",
               f8 ? "" : VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    }
}

// A build without the SDK headers in its deps folder has no FSR 4: aa_engine fsr4 then falls back to FSR 3.1.
#if !__has_include("ffx/upscalers/include/ffx_upscale.h")
int fsr4_failed(void) { return 1; }
Fsr4 *fsr4_create(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, VkImage mv, VkImage out) {
    (void)w, (void)h, (void)ow, (void)oh, (void)mv, (void)out;
    return NULL;
}
void fsr4_destroy(Fsr4 *f) { (void)f; }
int fsr4_dispatch(Fsr4 *f, VkCommandBuffer cb, const Fsr4Frame *in) { (void)f, (void)cb, (void)in; return 0; }
#else
#include "ffx/upscalers/include/ffx_upscale.h"
#include "ffx/api/include/vk/ffx_api_vk.h"

#define LOADER_DLL "amd_fidelityfx_loader_vk.dll"

static struct {
    int tried, failed;
    HMODULE dll;
    PfnFfxCreateContext create;
    PfnFfxDestroyContext destroy;
    PfnFfxQuery query;
    PfnFfxDispatch dispatch;
} L;

int fsr4_failed(void) { return L.failed; }

static int fail(const char *why) {
    if (!L.failed) rt_log("fsr4: %s; FSR 3.1 runs instead", why);
    L.failed = 1;
    return 0;
}

// The loader from the executable's folder (not the DLL search path), and the upscaler beside it.
static int load(void) {
    if (L.tried) return L.dll != NULL && !L.failed;
    L.tried = 1;
    char exe[1024], path[1100];
    wchar_t wpath[1100];
    char *sep = plat_exe_path(exe, sizeof exe) ? plat_last_sep(exe) : NULL;
    if (!sep) return fail("the executable's folder is unknown");
    snprintf(path, sizeof path, "%.*s\\" LOADER_DLL, (int)(sep - exe), exe);
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1100)) return fail("the loader path does not convert");
    L.dll = LoadLibraryExW(wpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!L.dll) {
        rt_log("fsr4: %s not loaded (error %lu): put AMD's amd_fidelityfx_loader_vk.dll and "
               "amd_fidelityfx_upscaler_vk.dll (FSR SDK 2.4, Kits/FidelityFX/signedbin) next to the executable", path,
               (unsigned long)GetLastError());
        return fail("no FSR SDK loader");
    }
    L.create = (PfnFfxCreateContext)(void (*)(void))GetProcAddress(L.dll, "ffxCreateContext");
    L.destroy = (PfnFfxDestroyContext)(void (*)(void))GetProcAddress(L.dll, "ffxDestroyContext");
    L.query = (PfnFfxQuery)(void (*)(void))GetProcAddress(L.dll, "ffxQuery");
    L.dispatch = (PfnFfxDispatch)(void (*)(void))GetProcAddress(L.dll, "ffxDispatch");
    if (!L.create || !L.destroy || !L.query || !L.dispatch) return fail("the loader lacks the ffx-api entry points");
    rt_log("fsr4: %s loaded", path);
    return 1;
}

// The SDK's own messages (errors and warnings of the provider), a few per run.
static void message(uint32_t type, const wchar_t *text) {
    static int count;
    if (count++ >= 20) return;
    char buf[512];
    if (!text || !WideCharToMultiByte(CP_UTF8, 0, text, -1, buf, sizeof buf, NULL, NULL)) buf[0] = 0;
    buf[sizeof buf - 1] = 0;
    rt_log("fsr4: SDK %s: %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", buf);
}

enum { IN_COLOR, IN_DEPTH, NOWN };

struct Fsr4 {
    ffxContext ctx;
    uint32_t w, h, ow, oh;
    VkImage img[NOWN], mv, out;     // img: this context's copies of the colour and depth; mv, out: vk_dlaa.c's
    VmaAllocation mem[NOWN];
    VkBuffer zbuf;                  // the depth on its way from the D32F target to IN_DEPTH
    VmaAllocation zbuf_mem;
    int general;                    // own images moved from UNDEFINED to GENERAL (first dispatch)
    uint64_t last_ns;               // plat_now_ns of the previous dispatch (frameTimeDelta)
};

static int own_image(Fsr4 *f, int i, VkFormat fmt, uint32_t w, uint32_t h) {
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ii, &ai, &f->img[i], &f->mem[i], NULL) == VK_SUCCESS) return 1;
    rt_log("fsr4: image %d (%ux%u, format %d) could not be created", i, w, h, (int)fmt);
    return 0;
}

static float knob_float(const char *name, float def, float lo) {
    const char *s = getenv(name);   // test-only switches, read like rt_knob's: the environment, no settings row
    double v = s ? strtod(s, NULL) : 0;
    return s && v >= lo ? (float)v : def;
}

Fsr4 *fsr4_create(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, VkImage mv, VkImage out) {
    if (!load()) return NULL;
    const VkCtx *c = vk_ctx();
    Fsr4 *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    f->w = w, f->h = h, f->ow = ow, f->oh = oh, f->mv = mv, f->out = out;
    static int hdr = -1, autoexp = -1, debug = -1;
    if (hdr < 0) hdr = rt_knob("FSR4_HDR", "1");          // TOTK_FSR4_HDR=0: tell FSR the colour is display-referred
    if (autoexp < 0) autoexp = rt_knob("FSR4_AUTO_EXPOSURE", "0");   // TOTK_FSR4_AUTO_EXPOSURE=1: FSR's own exposure
    if (debug < 0) debug = rt_knob("FSR4_DEBUG", "0");    // TOTK_FSR4_DEBUG=1: the SDK checks the API values it gets
    struct ffxCreateBackendVKDesc be = {
        .header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, NULL},
        .deviceContext = {.vkDevice = c->device, .vkPhysicalDevice = c->phys, .vkDeviceProcAddr = vkGetDeviceProcAddr}};
    struct ffxCreateContextDescUpscaleVersion ver = {.header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION, &be.header},
                                                     .version = FFX_UPSCALER_VERSION};
    struct ffxCreateContextDescUpscale cd = {
        .header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, &ver.header},
        .flags = (hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0) | (autoexp ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0) |
                 (debug ? FFX_UPSCALE_ENABLE_DEBUG_CHECKING : 0),
        .maxRenderSize = {w, h}, .maxUpscaleSize = {ow, oh}, .fpMessage = message};
    ffxReturnCode_t r = L.create(&f->ctx, &cd.header, NULL);
    if (r != FFX_API_RETURN_OK || !f->ctx) {
        rt_log("fsr4: context creation failed (%u) at %ux%u -> %ux%u", (unsigned)r, w, h, ow, oh);
        f->ctx = NULL;
        fsr4_destroy(f);
        fail("no FSR context");
        return NULL;
    }
    struct ffxQueryGetProviderVersion pv = {.header = {FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION, NULL}};
    const char *name = L.query(&f->ctx, &pv.header) == FFX_API_RETURN_OK && pv.versionName ? pv.versionName : "?";
    int ok = own_image(f, IN_COLOR, VK_FORMAT_R16G16B16A16_SFLOAT, w, h) &&
             own_image(f, IN_DEPTH, VK_FORMAT_R32_SFLOAT, w, h);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * 4,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    ok = ok && vmaCreateBuffer(c->vma, &bi, &ai, &f->zbuf, &f->zbuf_mem, NULL) == VK_SUCCESS;
    if (!ok) {
        fsr4_destroy(f);
        fail("its work images could not be created");
        return NULL;
    }
    static int logged_version;
    rt_log("fsr4: FSR Upscaling %s context %ux%u -> %ux%u (%s colour%s)", name, w, h, ow, oh, hdr ? "HDR" : "display",
           autoexp ? ", auto exposure" : "");
    if (!logged_version++ && name[0] != '4')
        rt_log("fsr4: the provider picked FSR %s, not FSR 4: this GPU or driver has no FSR 4 here", name);
    return f;
}

void fsr4_destroy(Fsr4 *f) {
    if (!f) return;
    const VkCtx *c = vk_ctx();
    if (f->ctx) L.destroy(&f->ctx, NULL);
    for (int i = 0; i < NOWN; i++)
        if (f->img[i]) vmaDestroyImage(c->vma, f->img[i], f->mem[i]);
    if (f->zbuf) vmaDestroyBuffer(c->vma, f->zbuf, f->zbuf_mem);
    free(f);
}

static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// Every image in imgs from layout a to layout b, after all earlier work and before all later work.
static void layouts(VkCommandBuffer cb, const VkImage *imgs, int n, VkImageLayout a, VkImageLayout b) {
    VkImageMemoryBarrier2 ib[3];
    for (int k = 0; k < n; k++)
        ib[k] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = a, .newLayout = b,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = imgs[k],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = (uint32_t)n,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
}

static struct FfxApiResource resource(VkImage img, uint32_t format, uint32_t w, uint32_t h, uint32_t usage,
                                      uint32_t state) {
    struct FfxApiResource r;
    memset(&r, 0, sizeof r);
    r.resource = (void *)img;
    r.state = state;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = format;
    r.description.width = w;
    r.description.height = h;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage = usage;
    return r;
}

int fsr4_dispatch(Fsr4 *f, VkCommandBuffer cb, const Fsr4Frame *in) {
    if (!f->general) {   // the context's own images start in UNDEFINED; they live in GENERAL like every other image
        layouts(cb, f->img, NOWN, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        f->general = 1;
    }
    // the scene's colour and depth into this context's RGBA16F and R32F images
    VkImageBlit blit = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .srcOffsets = {{0, 0, 0}, {(int32_t)f->w, (int32_t)f->h, 1}},
                        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .dstOffsets = {{0, 0, 0}, {(int32_t)f->w, (int32_t)f->h, 1}}};
    vkCmdBlitImage(cb, in->color, VK_IMAGE_LAYOUT_GENERAL, f->img[IN_COLOR], VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                   VK_FILTER_NEAREST);
    VkBufferImageCopy zc = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, .imageExtent = {f->w, f->h, 1}};
    vkCmdCopyImageToBuffer(cb, in->depth, VK_IMAGE_LAYOUT_GENERAL, f->zbuf, 1, &zc);
    barrier(cb);
    zc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkCmdCopyBufferToImage(cb, f->zbuf, f->img[IN_DEPTH], VK_IMAGE_LAYOUT_GENERAL, 1, &zc);
    // FSR reads its inputs in SHADER_READ_ONLY_OPTIMAL (its compute-read state) and writes the output in GENERAL
    const VkImage inputs[3] = {f->img[IN_COLOR], f->img[IN_DEPTH], f->mv};
    layouts(cb, inputs, 3, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    uint64_t now = plat_now_ns();
    double dt = f->last_ns ? (now - f->last_ns) / 1e6 : 0;
    f->last_ns = now;
    const uint32_t rd = FFX_API_RESOURCE_STATE_COMPUTE_READ, ro = FFX_API_RESOURCE_USAGE_READ_ONLY;
    struct ffxDispatchDescUpscale d;
    memset(&d, 0, sizeof d);
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = (void *)cb;
    d.color = resource(f->img[IN_COLOR], FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, f->w, f->h, ro, rd);
    d.depth = resource(f->img[IN_DEPTH], FFX_API_SURFACE_FORMAT_R32_FLOAT, f->w, f->h, ro, rd);
    d.motionVectors = resource(f->mv, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, f->w, f->h, ro, rd);
    d.output = resource(f->out, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, f->ow, f->oh, FFX_API_RESOURCE_USAGE_UAV,
                        FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset.x = in->jx, d.jitterOffset.y = in->jy;
    d.motionVectorScale.x = 1.0f, d.motionVectorScale.y = 1.0f;
    d.renderSize.width = f->w, d.renderSize.height = f->h;
    d.upscaleSize.width = f->ow, d.upscaleSize.height = f->oh;
    // TOTK_FSR4_SHARPNESS=0..1: FSR's own sharpening after the upscale (off by default, as for DLSS)
    float sharp = knob_float("TOTK_FSR4_SHARPNESS", 0.0f, 0.0f);
    d.enableSharpening = sharp > 0.0f;
    d.sharpness = sharp > 1.0f ? 1.0f : sharp;
    d.frameTimeDelta = dt > 0 && dt < 100 ? (float)dt : 16.67f;
    d.preExposure = 1.0f;
    d.reset = in->reset != 0;
    // fixed camera planes and field of view, as for FSR 3.1 (the projection is only on the GPU); TOTK_FSR3_NEAR / FAR /
    // FOV override them for both
    d.cameraNear = knob_float("TOTK_FSR3_NEAR", 0.1f, 1e-6f);
    d.cameraFar = knob_float("TOTK_FSR3_FAR", 10000.0f, 1e-6f);
    d.cameraFovAngleVertical = knob_float("TOTK_FSR3_FOV", 50.0f, 1e-6f) * 3.14159265f / 180.0f;
    d.viewSpaceToMetersFactor = 1.0f;
    ffxReturnCode_t r = L.dispatch(&f->ctx, &d.header);
    layouts(cb, inputs, 3, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    if (r == FFX_API_RETURN_OK) return 1;
    static int errors;
    if (errors++ < 5) rt_log("fsr4: dispatch failed (%u)", (unsigned)r);
    return 0;
}
#endif   // the SDK headers
