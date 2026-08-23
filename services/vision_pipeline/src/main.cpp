#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <sstream>

#include <grpcpp/grpcpp.h>
#include "camera.grpc.pb.h"
#include "pipeline.hpp"
#include "probe_processor.hpp"
#include <cstdlib>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using camera::CameraService;
using camera::CameraAddRequest;
using camera::CameraAddResponse;
using camera::CameraRemoveRequest;
using camera::CameraRemoveResponse;
using camera::CameraListRequest;
using camera::CameraListResponse;
using camera::CameraInfo;

// Implementation note.
class CameraServiceImpl final : public CameraService::Service {
    Status AddCamera(ServerContext* context, const CameraAddRequest* request,
                     CameraAddResponse* reply) override {
        std::string url = request->url();
        std::cout << "[gRPC] Received request to add camera: " << url << std::endl;

        // Implementation note.
        int src_id = pipeline_add_camera(url);

        if (src_id >= 0) {
            reply->set_src_id(src_id);
            reply->set_url(url);
            reply->set_status("added");
            std::cout << "[gRPC] Successfully added camera src_id=" << src_id << std::endl;
        } else {
            reply->set_src_id(-1);
            reply->set_url(url);
            reply->set_status("failed");
            std::cerr << "[gRPC] Failed to add camera: " << url << std::endl;
        }
        return Status::OK;
    }

    Status RemoveCamera(ServerContext* context, const CameraRemoveRequest* request,
                        CameraRemoveResponse* reply) override {
        int src_id = request->src_id();
        std::cout << "[gRPC] Received request to remove camera: " << src_id << std::endl;

        // Implementation note.
        bool ok = pipeline_remove_camera(src_id);

        reply->set_src_id(src_id);
        reply->set_status(ok ? "removed" : "not_found");
        std::cout << "[gRPC] Remove camera src_id=" << src_id << ", result=" << (ok ? "SUCCESS" : "NOT_FOUND") << std::endl;
        return Status::OK;
    }

    Status ListCameras(ServerContext* context, const CameraListRequest* request,
                       CameraListResponse* reply) override {
        std::cout << "[gRPC] Received request to list all cameras" << std::endl;

        // Implementation note.
        auto cameras = pipeline_list_cameras();

        for (const auto& cam : cameras) {
            auto* info = reply->add_cameras();
            info->set_src_id(cam.src_id);
            info->set_url(cam.url);
        }
        return Status::OK;
    }
};

void RunGrpcServer(const std::string& server_address) {
    CameraServiceImpl service;

    ServerBuilder builder;
    // Implementation note.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[gRPC] C++ Server listening on " << server_address << std::endl;

    // Implementation note.
    server->Wait();
}

int main(int argc, char** argv) {
    std::cout << "[MAIN] Starting C++ DeepStream/gRPC Service..." << std::endl;

    // Implementation note.
    const char* mqtt_broker_env = std::getenv("MQTT_BROKER");
    std::string mqtt_broker = mqtt_broker_env ? mqtt_broker_env : "127.0.0.1";

    const char* mqtt_port_env = std::getenv("MQTT_PORT");
    int mqtt_port = mqtt_port_env ? std::stoi(mqtt_port_env) : 1883;

    // Implementation note.
    if (!probe_processor_init(mqtt_broker, mqtt_port)) {
        std::cerr << "[MAIN] Probe processor initialization failed. Exiting..." << std::endl;
        return -1;
    }

    // Implementation note.
    std::string server_address("0.0.0.0:50051");
    std::thread grpc_thread(RunGrpcServer, server_address);

    // Implementation note.
    if (!pipeline_init()) {
        std::cerr << "[MAIN] Pipeline initialization failed. Exiting..." << std::endl;
        probe_processor_cleanup();
        return -1;
    }

    // Implementation note.
    const char* rtsp_urls_env = std::getenv("RTSP_URLS");
    if (rtsp_urls_env) {
        std::string urls_str(rtsp_urls_env);
        std::stringstream ss(urls_str);
        std::string url;
        while (std::getline(ss, url, ',')) {
            if (!url.empty()) {
                std::cout << "[MAIN] Automatically adding initial camera: " << url << std::endl;
                int src_id = pipeline_add_camera(url);
                if (src_id >= 0) {
                    std::cout << "[MAIN] Initial camera added with ID: " << src_id << std::endl;
                } else {
                    std::cerr << "[MAIN] Failed to add initial camera: " << url << std::endl;
                }
            }
        }
    }

    // Implementation note.
    pipeline_run();

    // Implementation note.
    pipeline_stop();
    probe_processor_cleanup();
    grpc_thread.join();

    std::cout << "[MAIN] DeepStream C++ Service stopped." << std::endl;
    return 0;
}
