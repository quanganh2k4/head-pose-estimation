#pragma once
#include <string>
#include <vector>

struct CameraSourceInfo {
    int src_id;
    std::string url;
};

// Khởi chạy GStreamer pipeline
bool pipeline_init();
void pipeline_run();
void pipeline_stop();

// API Thread-safe cho gRPC Server gọi vào điều khiển Pipeline
int pipeline_add_camera(const std::string& url);
bool pipeline_remove_camera(int src_id);
std::vector<CameraSourceInfo> pipeline_list_cameras();
