#pragma once
#include <string>
#include <vector>

struct CameraSourceInfo {
    int src_id;
    std::string url;
};

// Implementation note.
bool pipeline_init();
void pipeline_run();
void pipeline_stop();

// Implementation note.
int pipeline_add_camera(const std::string& url);
bool pipeline_remove_camera(int src_id);
std::vector<CameraSourceInfo> pipeline_list_cameras();
