# Dynamic RTSP Stream Manipulation in GStreamer

One of the key technical features of this system is the ability to dynamically connect and disconnect RTSP camera sources at runtime without restarting the running GStreamer pipeline.

## Mechanism

1. **gRPC Trigger:**
   The `analytics-api` sends an `AddCamera` or `RemoveCamera` RPC request to `vision-pipeline:50051`.

2. **GStreamer Bin Construction:**
   - For an `AddCamera` request:
     - Creates a `rtspsrc` element with low-latency jitter-buffer parameters.
     - Creates `rtph264depay` / `rtph265depay` and `nvv4l2decoder`.
     - Acquires a request pad `sink_%u` on `nvstreammux`.
     - Dynamically links the decoder `src` pad to the `nvstreammux` request pad upon `pad-added` signal.
     - Syncs state to `GST_STATE_PLAYING`.

3. **Pad Unlinking & Element Cleanup:**
   - For a `RemoveCamera` request:
     - Sends an EOS event or blocks the pad using `gst_pad_add_probe` (`GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM`).
     - Unlinks the bin from `nvstreammux` request sink pad.
     - Releases the request pad back to `nvstreammux`.
     - Transitions the source bin to `GST_STATE_NULL` and unrefs it cleanly from the parent pipeline.
