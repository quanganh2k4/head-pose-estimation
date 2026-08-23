#pragma once
#include <gst/gst.h>
#include <string>

// Implementation note.
bool probe_processor_init(const std::string& mqtt_broker, int mqtt_port);
void probe_processor_cleanup();

// GStreamer Pad Probe Callback
GstPadProbeReturn sgie_src_pad_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
