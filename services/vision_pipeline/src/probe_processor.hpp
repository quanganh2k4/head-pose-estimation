#pragma once
#include <gst/gst.h>
#include <string>

// Khởi tạo các module Inference và MQTT
bool probe_processor_init(const std::string& mqtt_broker, int mqtt_port);
void probe_processor_cleanup();

// GStreamer Pad Probe Callback
GstPadProbeReturn sgie_src_pad_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
