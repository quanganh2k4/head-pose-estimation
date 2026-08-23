#include "pipeline.hpp"
#include "probe_processor.hpp"
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <map>
#include <gst/gst.h>
#include <glib.h>

static std::mutex g_pipeline_mutex;
static GstElement *g_pipeline = nullptr;
static GstElement *g_streammux = nullptr;
static GMainLoop *g_main_loop = nullptr;

struct SourceInfo {
    GstElement *src_bin;
    GstPad *mux_sink_pad;
    std::string url;
    bool active;
};

static std::map<int, SourceInfo> g_sources;
static int g_next_id = 0;

// Implementation note.
static void cb_newpad(GstElement *element, GstPad *pad, gpointer data) {
    GstPad *sink_pad = (GstPad *)data;
    
    // Implementation note.
    if (gst_pad_get_direction(pad) != GST_PAD_SRC) {
        return;
    }

    if (gst_pad_is_linked(sink_pad)) {
        std::cout << "[Pipeline] Sink pad already linked." << std::endl;
        return;
    }

    // Implementation note.
    GstCaps *caps = gst_pad_get_current_caps(pad);
    if (!caps) {
        caps = gst_pad_query_caps(pad, nullptr);
    }
    
    if (caps && gst_caps_get_size(caps) > 0) {
        GstStructure *str = gst_caps_get_structure(caps, 0);
        const gchar *name = gst_structure_get_name(str);
        if (g_strrstr(name, "video")) {
            GstPadLinkReturn ret = gst_pad_link(pad, sink_pad);
            if (ret != GST_PAD_LINK_OK) {
                std::cerr << "[Pipeline] Link failed: " << gst_pad_link_get_name(ret) << std::endl;
            } else {
                std::cout << "[Pipeline] Linked source pad to nvstreammux pad successfully" << std::endl;
            }
        }
    }
    if (caps) {
        gst_caps_unref(caps);
    }
}

bool pipeline_init() {
    std::cout << "[Pipeline] Initializing GStreamer..." << std::endl;
    gst_init(nullptr, nullptr);

    g_main_loop = g_main_loop_new(nullptr, FALSE);
    if (!g_main_loop) {
        std::cerr << "[Pipeline] Failed to create GLib main loop" << std::endl;
        return false;
    }

    g_pipeline = gst_pipeline_new("deepstream-grpc-pipeline");
    if (!g_pipeline) {
        std::cerr << "[Pipeline] Failed to create pipeline" << std::endl;
        return false;
    }

    // Implementation note.
    g_streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    if (!g_streammux) {
        std::cerr << "[Pipeline] Failed to create nvstreammux" << std::endl;
        return false;
    }

    // Implementation note.
    g_object_set(G_OBJECT(g_streammux), "batch-size", 8, nullptr);
    g_object_set(G_OBJECT(g_streammux), "width", 1920, nullptr); // Implementation note.
    g_object_set(G_OBJECT(g_streammux), "height", 1080, nullptr);
    g_object_set(G_OBJECT(g_streammux), "batched-push-timeout", 33000, nullptr);
    g_object_set(G_OBJECT(g_streammux), "live-source", 1, nullptr);

    gst_bin_add(GST_BIN(g_pipeline), g_streammux);

    // Implementation note.
    GstElement *pgie = gst_element_factory_make("nvinfer", "pgie");
    if (!pgie) {
        std::cerr << "[Pipeline] Failed to create nvinfer (pgie)" << std::endl;
        return false;
    }
    const char* pgie_config_env = std::getenv("PEOPLENET_CONFIG");
    std::string pgie_config_path = pgie_config_env ? pgie_config_env : "/models/peoplenet/config_infer_peoplenet.txt";
    g_object_set(G_OBJECT(pgie), "config-file-path", pgie_config_path.c_str(), nullptr);
    g_object_set(G_OBJECT(pgie), "batch-size", 8, nullptr);

    // Implementation note.
    GstElement *nvvideoconvert = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    GstElement *caps_rgba = gst_element_factory_make("capsfilter", "caps-rgba");
    GstElement *fakesink = gst_element_factory_make("fakesink", "fake-sink");

    if (!nvvideoconvert || !caps_rgba || !fakesink) {
        std::cerr << "[Pipeline] Failed to create converter, caps_rgba, or fakesink" << std::endl;
        return false;
    }

    // Implementation note.
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
    g_object_set(G_OBJECT(caps_rgba), "caps", caps, nullptr);
    gst_caps_unref(caps);

    // Implementation note.
    g_object_set(G_OBJECT(fakesink), "sync", FALSE, nullptr);
    g_object_set(G_OBJECT(fakesink), "async", FALSE, nullptr);

    gst_bin_add_many(GST_BIN(g_pipeline), pgie, nvvideoconvert, caps_rgba, fakesink, nullptr);

    // Link: streammux -> pgie -> nvvideoconvert -> caps_rgba -> fakesink
    if (!gst_element_link_many(g_streammux, pgie, nvvideoconvert, caps_rgba, fakesink, nullptr)) {
        std::cerr << "[Pipeline] Failed to link: streammux -> pgie -> converter -> caps_rgba -> fakesink" << std::endl;
        return false;
    }

    // Implementation note.
    GstPad *mpconv_src = gst_element_get_static_pad(nvvideoconvert, "src");
    if (!mpconv_src) {
        std::cerr << "[Pipeline] Failed to get converter src pad" << std::endl;
        return false;
    }
    gst_pad_add_probe(mpconv_src, GST_PAD_PROBE_TYPE_BUFFER, sgie_src_pad_probe, nullptr, nullptr);
    gst_object_unref(mpconv_src);

    std::cout << "[Pipeline] Base pipeline (with nvinfer & pad probe) constructed successfully." << std::endl;
    return true;
}

void pipeline_run() {
    std::cout << "[Pipeline] Setting pipeline state to PLAYING..." << std::endl;
    GstStateChangeReturn r = gst_element_set_state(g_pipeline, GST_STATE_PLAYING);
    if (r == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Pipeline] Failed to set pipeline to PLAYING state" << std::endl;
        return;
    }

    std::cout << "[Pipeline] Running main loop..." << std::endl;
    g_main_loop_run(g_main_loop);
}

void pipeline_stop() {
    std::cout << "[Pipeline] Stopping pipeline..." << std::endl;
    if (g_main_loop) {
        g_main_loop_quit(g_main_loop);
    }
    if (g_pipeline) {
        gst_element_set_state(g_pipeline, GST_STATE_NULL);
        gst_object_unref(g_pipeline);
        g_pipeline = nullptr;
    }
    if (g_main_loop) {
        g_main_loop_unref(g_main_loop);
        g_main_loop = nullptr;
    }
}

int pipeline_add_camera(const std::string& url) {
    std::lock_guard<std::mutex> lock(g_pipeline_mutex);
    
    // Implementation note.
    for (const auto& pair : g_sources) {
        if (pair.second.url == url && pair.second.active) {
            std::cout << "[Pipeline] URL already active with ID " << pair.first << std::endl;
            return pair.first;
        }
    }

    int src_id = g_next_id++;
    std::cout << "[Pipeline] Adding camera ID " << src_id << ", URL: " << url << std::endl;

    std::string bin_name = "src-bin-" + std::to_string(src_id);
    GstElement *src_bin = gst_element_factory_make("nvurisrcbin", bin_name.c_str());
    if (!src_bin) {
        std::cerr << "[Pipeline] Failed to create nvurisrcbin" << std::endl;
        return -1;
    }

    g_object_set(G_OBJECT(src_bin), "uri", url.c_str(), nullptr);
    g_object_set(G_OBJECT(src_bin), "type", 2, nullptr); // RTSP
    g_object_set(G_OBJECT(src_bin), "select-rtp-protocol", 4, nullptr); // TCP+UDP
    g_object_set(G_OBJECT(src_bin), "latency", 200, nullptr);
    g_object_set(G_OBJECT(src_bin), "rtsp-reconnect-interval", 10, nullptr);
    g_object_set(G_OBJECT(src_bin), "rtsp-reconnect-attempts", 0, nullptr);

    // Implementation note.
    std::string pad_name = "sink_" + std::to_string(src_id);
    GstPad *mux_sink_pad = gst_element_get_request_pad(g_streammux, pad_name.c_str());
    if (!mux_sink_pad) {
        std::cerr << "[Pipeline] Failed to request sink pad from streammux" << std::endl;
        gst_object_unref(src_bin);
        return -1;
    }

    gst_bin_add(GST_BIN(g_pipeline), src_bin);

    // Implementation note.
    g_signal_connect(src_bin, "pad-added", G_CALLBACK(cb_newpad), mux_sink_pad);

    // Implementation note.
    gst_element_sync_state_with_parent(src_bin);

    SourceInfo info;
    info.src_bin = src_bin;
    info.mux_sink_pad = mux_sink_pad;
    info.url = url;
    info.active = true;

    g_sources[src_id] = info;

    return src_id;
}

// Implementation note.
struct RemoveContext {
    int src_id;
    GstElement *src_bin;
    GstPad *mux_sink_pad;
};

// Implementation note.
static gboolean cb_remove_source(gpointer data) {
    auto *ctx = (RemoveContext *)data;
    
    std::cout << "[Pipeline] Safely removing camera ID: " << ctx->src_id << " in main thread..." << std::endl;

    // Implementation note.
    gst_element_set_state(ctx->src_bin, GST_STATE_NULL);

    // Implementation note.
    if (gst_pad_is_linked(ctx->mux_sink_pad)) {
        GstPad *peer = gst_pad_get_peer(ctx->mux_sink_pad);
        if (peer) {
            gst_pad_unlink(peer, ctx->mux_sink_pad);
            gst_object_unref(peer);
        }
    }

    // Implementation note.
    gst_element_release_request_pad(g_streammux, ctx->mux_sink_pad);

    // Implementation note.
    gst_bin_remove(GST_BIN(g_pipeline), ctx->src_bin);

    // Implementation note.
    delete ctx;
    std::cout << "[Pipeline] Camera removed successfully." << std::endl;
    return FALSE; // Implementation note.
}

bool pipeline_remove_camera(int src_id) {
    std::lock_guard<std::mutex> lock(g_pipeline_mutex);
    auto it = g_sources.find(src_id);
    if (it == g_sources.end() || !it->second.active) {
        std::cerr << "[Pipeline] Camera ID " << src_id << " not found or inactive" << std::endl;
        return false;
    }

    it->second.active = false;

    // Implementation note.
    auto *ctx = new RemoveContext();
    ctx->src_id = src_id;
    ctx->src_bin = it->second.src_bin;
    ctx->mux_sink_pad = it->second.mux_sink_pad;

    g_idle_add(cb_remove_source, ctx);

    g_sources.erase(it);
    return true;
}

std::vector<CameraSourceInfo> pipeline_list_cameras() {
    std::lock_guard<std::mutex> lock(g_pipeline_mutex);
    std::vector<CameraSourceInfo> active_list;
    for (const auto& pair : g_sources) {
        if (pair.second.active) {
            active_list.push_back({pair.first, pair.second.url});
        }
    }
    return active_list;
}
