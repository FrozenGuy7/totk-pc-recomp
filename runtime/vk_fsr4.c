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
// into an RGBA16F and an R32F image of the context first, as vk_fsr3.c does, so FSR reads images whose formats and
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

// The SDK's Vulkan back end (Kits/FidelityFX/backend/vk/ffx_vk.cpp, ExecuteGpuJobsVK) makes image views of the images it
// is given at every dispatch and destroys them three dispatches later (FFX_MAX_QUEUED_FRAMES 4); its descriptor sets and
// constants rotate the same way. It assumes the GPU is never more dispatches behind. TotK's frames can run further ahead
// at a high frame rate (asynchronous replay, split submissions), and then the GPU read views and memory already gone (a
// GPU page fault, or an access violation in the driver). So a dispatch first waits until the submission of the dispatch
// IN_FLIGHT back is done: the views it destroys at its end are that dispatch's.
enum { IN_FLIGHT = 3 };

// One FSR context for the run, made with the largest render and output size seen and FSR's dynamic resolution flag:
// TotK's dynamic resolution moves the scene size every few frames, and each frame passes the size it has. A second
// context per scene size (what vk_fsr3.c does) ended in a GPU page fault with the FSR 4 provider. The copies of the
// colour and depth are made at the largest size and FSR reads the render size's corner of them.
static struct {
    ffxContext ctx;
    uint32_t mw, mh, ow, oh;        // the largest render size and the output size the context was made for
    VkImage img[NOWN];
    VmaAllocation mem[NOWN];
    VkBuffer zbuf;                  // the depth on its way from the D32F target to IN_DEPTH
    VmaAllocation zbuf_mem;
    int general;                    // own images moved from UNDEFINED to GENERAL (first dispatch)
    int fresh;                      // made since the last dispatch: its history starts empty
    int users;                      // live Fsr4 handles (vk_dlaa.c's scene sizes)
    uint64_t last_ns;               // plat_now_ns of the previous dispatch (frameTimeDelta)
    uint64_t fence[IN_FLIGHT];      // the timeline values of the last dispatches' submissions (a ring)
    uint32_t dispatches;            // dispatches recorded so far
    uint64_t waits, wait_ns;        // dispatches that waited for an older one, and how long in all
} G;

struct Fsr4 {                       // one per vk_dlaa.c scene size: its size and work images
    uint32_t w, h, ow, oh;
    VkImage mv, out;
};

static int own_image(int i, VkFormat fmt, uint32_t w, uint32_t h) {
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ii, &ai, &G.img[i], &G.mem[i], NULL) == VK_SUCCESS) return 1;
    rt_log("fsr4: image %d (%ux%u, format %d) could not be created", i, w, h, (int)fmt);
    return 0;
}

static float knob_float(const char *name, float def, float lo) {
    const char *s = getenv(name);   // test-only switches, read like rt_knob's: the environment, no settings row
    double v = s ? strtod(s, NULL) : 0;
    return s && v >= lo ? (float)v : def;
}

// The context and its images, after the GPU finished everything recorded with them.
static void shared_free(void) {
    const VkCtx *c = vk_ctx();
    if (G.ctx) L.destroy(&G.ctx, NULL);
    for (int i = 0; i < NOWN; i++)
        if (G.img[i]) vmaDestroyImage(c->vma, G.img[i], G.mem[i]);
    if (G.zbuf) vmaDestroyBuffer(c->vma, G.zbuf, G.zbuf_mem);
    int users = G.users;
    memset(&G, 0, sizeof G);
    G.users = users;
}

// A context for render sizes up to mw x mh and output ow x oh. 0 when it cannot be made (FSR 4 is then off).
static int shared_make(uint32_t mw, uint32_t mh, uint32_t ow, uint32_t oh) {
    const VkCtx *c = vk_ctx();
    static int hdr = -1, autoexp = -1, debug = -1, dynres = -1;
    if (hdr < 0) hdr = rt_knob("FSR4_HDR", "1");          // TOTK_FSR4_HDR=0: tell FSR the colour is display-referred
    if (dynres < 0) dynres = rt_knob("FSR4_DYNRES", "1"); // TOTK_FSR4_DYNRES=0: no dynamic resolution flag (a test)
    if (autoexp < 0) autoexp = rt_knob("FSR4_AUTO_EXPOSURE", "0");   // TOTK_FSR4_AUTO_EXPOSURE=1: FSR's own exposure
    if (debug < 0) debug = rt_knob("FSR4_DEBUG", "0");    // TOTK_FSR4_DEBUG=1: the SDK checks the API values it gets
    struct ffxCreateBackendVKDesc be = {
        .header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, NULL},
        .deviceContext = {.vkDevice = c->device, .vkPhysicalDevice = c->phys, .vkDeviceProcAddr = vkGetDeviceProcAddr}};
    struct ffxCreateContextDescUpscaleVersion ver = {.header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION, &be.header},
                                                     .version = FFX_UPSCALER_VERSION};
    struct ffxCreateContextDescUpscale cd = {
        .header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, &ver.header},
        .flags = (dynres ? FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION : 0) | (hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0) |
                 (autoexp ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0) | (debug ? FFX_UPSCALE_ENABLE_DEBUG_CHECKING : 0),
        .maxRenderSize = {mw, mh}, .maxUpscaleSize = {ow, oh}, .fpMessage = message};
    ffxReturnCode_t r = L.create(&G.ctx, &cd.header, NULL);
    if (r != FFX_API_RETURN_OK || !G.ctx) {
        rt_log("fsr4: context creation failed (%u) for up to %ux%u -> %ux%u", (unsigned)r, mw, mh, ow, oh);
        G.ctx = NULL;
        shared_free();
        return fail("no FSR context");
    }
    G.mw = mw, G.mh = mh, G.ow = ow, G.oh = oh, G.fresh = 1;
    struct ffxQueryGetProviderVersion pv = {.header = {FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION, NULL}};
    const char *name = L.query(&G.ctx, &pv.header) == FFX_API_RETURN_OK && pv.versionName ? pv.versionName : "?";
    int ok = own_image(IN_COLOR, VK_FORMAT_R16G16B16A16_SFLOAT, mw, mh) &&
             own_image(IN_DEPTH, VK_FORMAT_R32_SFLOAT, mw, mh);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)mw * mh * 4,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    ok = ok && vmaCreateBuffer(c->vma, &bi, &ai, &G.zbuf, &G.zbuf_mem, NULL) == VK_SUCCESS;
    if (!ok) {
        shared_free();
        return fail("its work images could not be created");
    }
    static int logged_version;
    rt_log("fsr4: FSR Upscaling %s context for scenes up to %ux%u -> %ux%u (flags %#x:%s%s colour%s%s)", name, mw, mh, ow,
           oh, cd.flags, dynres ? " dynamic resolution," : "", hdr ? " HDR" : " display", autoexp ? ", auto exposure" : "",
           debug ? ", SDK checks" : "");
    if (!logged_version++ && name[0] != '4')
        rt_log("fsr4: the provider picked FSR %s, not FSR 4: this GPU or driver has no FSR 4 here", name);
    return 1;
}

Fsr4 *fsr4_create(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, VkImage mv, VkImage out) {
    if (!load()) return NULL;
    // the context covers at least the output size, so the usual sizes (DLAA, the quality modes, dynamic resolution
    // below the output) never need another; a larger scene (render_scale above 100) or another output size makes it anew
    uint32_t mw = w > ow ? w : ow, mh = h > oh ? h : oh;
    if (G.ctx && (mw > G.mw || mh > G.mh || ow != G.ow || oh != G.oh)) {
        vkDeviceWaitIdle(vk_ctx()->device);   // the old context's frames may still be in flight
        rt_log("fsr4: %ux%u -> %ux%u is outside the context (%ux%u -> %ux%u): a new one", w, h, ow, oh, G.mw, G.mh, G.ow,
               G.oh);
        if (mw < G.mw) mw = G.mw;
        if (mh < G.mh) mh = G.mh;
        shared_free();
    }
    if (!G.ctx && !shared_make(mw, mh, ow, oh)) return NULL;
    Fsr4 *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    f->w = w, f->h = h, f->ow = ow, f->oh = oh, f->mv = mv, f->out = out;
    G.users++;
    return f;
}

void fsr4_destroy(Fsr4 *f) {
    if (!f) return;
    free(f);
    if (--G.users <= 0) {   // vk_dlaa.c's free_size waited for the device: the last size's frames are done
        G.users = 0;
        shared_free();
    }
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
    if (!G.ctx || f->w > G.mw || f->h > G.mh || f->ow != G.ow || f->oh != G.oh) return 0;   // fsr4_create keeps it fitting
    uint64_t old = G.dispatches >= IN_FLIGHT ? G.fence[G.dispatches % IN_FLIGHT] : 0;
    if (old && old < in->frame && vk_sync_fence_done() < old) {
        uint64_t t0 = plat_now_ns();
        if (!vk_sync_fence_wait(old, 2000000000ull)) rt_log("fsr4: submission %llu not done after 2 s",
                                                             (unsigned long long)old);
        G.waits++, G.wait_ns += plat_now_ns() - t0;
    }
    G.fence[G.dispatches++ % IN_FLIGHT] = in->frame;
    if (G.dispatches == 600 || G.dispatches % 36000 == 0)
        rt_log("fsr4: %u dispatches; %llu waited for the GPU to finish an older one (%.2f ms in all)", G.dispatches,
               (unsigned long long)G.waits, G.wait_ns / 1e6);
    if (!G.general) {   // the context's own images start in UNDEFINED; they live in GENERAL like every other image
        layouts(cb, G.img, NOWN, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        G.general = 1;
    }
    // the scene's colour and depth into the top-left w x h of the context's RGBA16F and R32F images
    VkImageBlit blit = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .srcOffsets = {{0, 0, 0}, {(int32_t)f->w, (int32_t)f->h, 1}},
                        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .dstOffsets = {{0, 0, 0}, {(int32_t)f->w, (int32_t)f->h, 1}}};
    vkCmdBlitImage(cb, in->color, VK_IMAGE_LAYOUT_GENERAL, G.img[IN_COLOR], VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                   VK_FILTER_NEAREST);
    VkBufferImageCopy zc = {.bufferRowLength = f->w, .bufferImageHeight = f->h,
                            .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, .imageExtent = {f->w, f->h, 1}};
    vkCmdCopyImageToBuffer(cb, in->depth, VK_IMAGE_LAYOUT_GENERAL, G.zbuf, 1, &zc);
    barrier(cb);
    zc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkCmdCopyBufferToImage(cb, G.zbuf, G.img[IN_DEPTH], VK_IMAGE_LAYOUT_GENERAL, 1, &zc);
    // FSR reads its inputs in SHADER_READ_ONLY_OPTIMAL (its compute-read state) and writes the output in GENERAL
    const VkImage inputs[3] = {G.img[IN_COLOR], G.img[IN_DEPTH], f->mv};
    layouts(cb, inputs, 3, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    uint64_t now = plat_now_ns();
    double dt = G.last_ns ? (now - G.last_ns) / 1e6 : 0;
    G.last_ns = now;
    const uint32_t rd = FFX_API_RESOURCE_STATE_COMPUTE_READ, ro = FFX_API_RESOURCE_USAGE_READ_ONLY;
    struct ffxDispatchDescUpscale d;
    memset(&d, 0, sizeof d);
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = (void *)cb;
    // the descriptions give each image's own size; renderSize says how much of the inputs this frame fills
    d.color = resource(G.img[IN_COLOR], FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, G.mw, G.mh, ro, rd);
    d.depth = resource(G.img[IN_DEPTH], FFX_API_SURFACE_FORMAT_R32_FLOAT, G.mw, G.mh, ro, rd);
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
    d.reset = in->reset || G.fresh;
    G.fresh = 0;
    // fixed camera planes and field of view, as for FSR 3.1 (the projection is only on the GPU); TOTK_FSR3_NEAR / FAR /
    // FOV override them for both
    d.cameraNear = knob_float("TOTK_FSR3_NEAR", 0.1f, 1e-6f);
    d.cameraFar = knob_float("TOTK_FSR3_FAR", 10000.0f, 1e-6f);
    d.cameraFovAngleVertical = knob_float("TOTK_FSR3_FOV", 50.0f, 1e-6f) * 3.14159265f / 180.0f;
    d.viewSpaceToMetersFactor = 1.0f;
    ffxReturnCode_t r = L.dispatch(&G.ctx, &d.header);
    layouts(cb, inputs, 3, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    if (r == FFX_API_RETURN_OK) return 1;
    rt_log("fsr4: dispatch failed (%u): render %ux%u of %ux%u inputs, motion vectors %ux%u, output %ux%u (context up to "
           "%ux%u -> %ux%u), jitter %.3f %.3f, reset %d, frame time %.2f ms, near %g far %g fov %.3f rad, sharpening %d %.2f",
           (unsigned)r, f->w, f->h, G.mw, G.mh, f->w, f->h, f->ow, f->oh, G.mw, G.mh, G.ow, G.oh, d.jitterOffset.x,
           d.jitterOffset.y, (int)d.reset, d.frameTimeDelta, d.cameraNear, d.cameraFar, d.cameraFovAngleVertical,
           (int)d.enableSharpening, d.sharpness);
    fail("FSR refused a frame");
    return 0;
}
#endif   // the SDK headers
