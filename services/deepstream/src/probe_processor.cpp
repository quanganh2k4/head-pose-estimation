#include "probe_processor.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <chrono>
#include <mutex>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <mosquitto.h>

// DeepStream/CUDA/TensorRT headers
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// Cấu hình
static const int PROCESS_EVERY_N_FRAMES = 4;
// Ngưỡng confidence sau sigmoid (score model trả về là raw logit).
// Score head của engine này calibration thấp: mặt rõ ~0.35-0.45, không có mặt <0.01,
// nên 0.2 đủ tách biệt hai trường hợp.
static const float LANDMARK_CONF_THRESH = 0.2f;
// Số frame tối đa được phép dùng lại landmark từ cache khi inference fail/skip
static const uint64_t MAX_CACHE_STALENESS = 2 * PROCESS_EVERY_N_FRAMES;
static const uint64_t UINT64_MAX_VAL = 18446744073709551615ULL;
static const int ID_GRID_PX = 80;

// logger cho TensorRT
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} g_trt_logger;

class FaceMeshTRT {
private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    void* d_input = nullptr;
    void* d_landmarks = nullptr;
    void* d_score = nullptr;
    cudaStream_t stream = nullptr;
    std::mutex mutex;
    bool initialized = false;

public:
    FaceMeshTRT() {}
    ~FaceMeshTRT() {
        cleanup();
    }

    void cleanup() {
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        if (d_input) { cudaFree(d_input); d_input = nullptr; }
        if (d_landmarks) { cudaFree(d_landmarks); d_landmarks = nullptr; }
        if (d_score) { cudaFree(d_score); d_score = nullptr; }
        if (context) { context->destroy(); context = nullptr; }
        if (engine) { engine->destroy(); engine = nullptr; }
        if (runtime) { runtime->destroy(); runtime = nullptr; }
        initialized = false;
    }

    bool init(const std::string& engine_path) {
        cleanup();
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            std::cerr << "[TRT] Failed to open engine file: " << engine_path << std::endl;
            return false;
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "[TRT] Failed to read engine file" << std::endl;
            return false;
        }

        runtime = nvinfer1::createInferRuntime(g_trt_logger);
        if (!runtime) {
            std::cerr << "[TRT] Failed to create InferRuntime" << std::endl;
            return false;
        }

        engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
        if (!engine) {
            std::cerr << "[TRT] Failed to deserialize CUDA engine" << std::endl;
            return false;
        }

        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "[TRT] Failed to create execution context" << std::endl;
            return false;
        }

        // Allocate CUDA memory
        if (cudaMalloc(&d_input, 1 * 3 * 192 * 192 * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&d_landmarks, 1404 * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&d_score, 1 * sizeof(float)) != cudaSuccess) return false;

        if (cudaStreamCreate(&stream) != cudaSuccess) return false;

        initialized = true;
        std::cout << "[TRT] Loaded engine successfully: " << engine_path << std::endl;
        return true;
    }

    bool infer(const float* host_input, float* host_landmarks, float* host_score) {
        std::lock_guard<std::mutex> lock_g(mutex);
        if (!initialized) return false;

        if (cudaMemcpyAsync(d_input, host_input, 1 * 3 * 192 * 192 * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;

        void* buffers[3];
        int input_idx = engine->getBindingIndex("input");
        int landmarks_idx = engine->getBindingIndex("landmarks");
        int score_idx = engine->getBindingIndex("score");

        if (input_idx < 0 || landmarks_idx < 0 || score_idx < 0) {
            std::cerr << "[TRT] Failed to find binding indices" << std::endl;
            return false;
        }

        buffers[input_idx] = d_input;
        buffers[landmarks_idx] = d_landmarks;
        buffers[score_idx] = d_score;

        // executeV2 (blocking) thay vì enqueueV2 + stream riêng: enqueueV2 trong process
        // DeepStream bị xung đột stream (engine luôn nhận input rỗng), executeV2 chạy đúng
        cudaStreamSynchronize(stream);
        if (!context->executeV2(buffers)) {
            std::cerr << "[TRT] Execution failed" << std::endl;
            return false;
        }

        if (cudaMemcpyAsync(host_landmarks, d_landmarks, 1404 * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        if (cudaMemcpyAsync(host_score, d_score, 1 * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;

        cudaStreamSynchronize(stream);
        return true;
    }

    bool is_initialized() const { return initialized; }
};

static FaceMeshTRT g_face_mesh_trt;

// State của MQTT
static struct mosquitto *g_mosq = nullptr;

// Cache lưu landmark của các khuôn mặt để làm mượt giữa các frame không chạy inference
struct FaceCacheItem {
    std::vector<float> nose;
    std::vector<float> chin;
    std::vector<float> left_eye;
    std::vector<float> right_eye;
    std::vector<float> bridge;
    std::vector<float> left_cheek;
    std::vector<float> right_cheek;
    uint64_t last_seen;
};

// Key: (src_id, obj_id) -> CacheItem
static std::map<std::pair<int, int>, FaceCacheItem> g_pose_cache;
static std::mutex g_cache_mutex;
static std::map<int, uint64_t> g_frame_counters;
static std::mutex g_counters_mutex;

// Khởi tạo Landmark network & MQTT Client
bool probe_processor_init(const std::string& mqtt_broker, int mqtt_port) {
    // 1. Load TensorRT Engine
    const char* trt_path_env = std::getenv("TRT_ENGINE_PATH");
    std::string trt_path = trt_path_env ? trt_path_env : "/models/mediapipe_pose/20_new_onnx_postprocess_N-batch/face_mesh_192x192.onnx_b1_gpu0_fp16.engine";
    
    std::cout << "[Probe] Initializing FaceMeshTRT with engine path: " << trt_path << std::endl;
    if (!g_face_mesh_trt.init(trt_path)) {
        std::cerr << "[Probe] CRITICAL: Failed to initialize FaceMeshTRT engine!" << std::endl;
        return false;
    }


    // 2. Khởi tạo Mosquitto
    mosquitto_lib_init();
    g_mosq = mosquitto_new("jetson-deepstream-cpp", true, nullptr);
    if (!g_mosq) {
        std::cerr << "[MQTT] Failed to create mosquitto instance" << std::endl;
        return false;
    }

    int rc = mosquitto_connect(g_mosq, mqtt_broker.c_str(), mqtt_port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        std::cerr << "[MQTT] Connect failed with code: " << rc << std::endl;
        return false;
    }

    mosquitto_loop_start(g_mosq);
    std::cout << "[MQTT] Connected to " << mqtt_broker << ":" << mqtt_port << " and started event loop thread." << std::endl;

    return true;
}

void probe_processor_cleanup() {
    g_face_mesh_trt.cleanup();
    if (g_mosq) {
        mosquitto_disconnect(g_mosq);
        mosquitto_loop_stop(g_mosq, false);
        mosquitto_destroy(g_mosq);
        g_mosq = nullptr;
    }
    mosquitto_lib_cleanup();
    std::cout << "[Probe] Cleaned up MQTT resources." << std::endl;
}

// Kiểm tra tính hợp lý hình học của landmark trước khi chấp nhận kết quả.
// Loại các frame model "đoán bừa" (che khuất, khóa nhầm mặt bên cạnh, input hỏng gây NaN).
static bool sanity_check_landmarks(const FaceCacheItem& item, float fl, float ft, float fw, float fh) {
    const std::vector<const std::vector<float>*> pts = {
        &item.nose, &item.chin, &item.left_eye, &item.right_eye,
        &item.bridge, &item.left_cheek, &item.right_cheek
    };

    // 1. Tất cả tọa độ phải hữu hạn (NaN/Inf sẽ phá vỡ JSON và tính toán góc phía sau)
    float cx = 0.0f, cy = 0.0f;
    for (const auto* p : pts) {
        if (p->size() != 3 || !std::isfinite((*p)[0]) || !std::isfinite((*p)[1]) || !std::isfinite((*p)[2])) return false;
        cx += (*p)[0]; cy += (*p)[1];
    }
    cx /= pts.size(); cy /= pts.size();

    // 2. Trọng tâm landmark phải nằm trong bbox mở rộng (chống khóa nhầm sang mặt lân cận trong crop rộng)
    float margin_x = 0.35f * fw, margin_y = 0.45f * fh;
    if (cx < fl - margin_x || cx > fl + fw + margin_x ||
        cy < ft - margin_y || cy > ft + fh + margin_y) return false;

    // 3. Khoảng cách 2 mắt phải hợp lý so với bề rộng mặt (cận dưới thấp để không loại nhầm profile view)
    float ex = item.left_eye[0] - item.right_eye[0];
    float ey = item.left_eye[1] - item.right_eye[1];
    float eye_dist = sqrtf(ex * ex + ey * ey);
    if (eye_dist < 0.10f * fw || eye_dist > 0.90f * fw) return false;

    // 4. Khoảng cách cằm↔sống mũi phải hợp lý so với khoảng cách 2 mắt (loại kết quả
    // suy biến: các điểm trùng nhau hoặc văng quá xa). Cố ý KHÔNG so sánh theo trục Y
    // thuần túy (cằm phải "ở dưới" bridge) — khi đầu nghiêng (roll) cằm dịch ngang chứ
    // không còn thấp hơn bridge theo Y nữa, so sánh Y sẽ loại nhầm mặt nghiêng hợp lệ.
    float bx = item.chin[0] - item.bridge[0];
    float by = item.chin[1] - item.bridge[1];
    float chin_bridge_dist = sqrtf(bx * bx + by * by);
    if (chin_bridge_dist < 0.5f * eye_dist || chin_bridge_dist > 4.0f * eye_dist) return false;

    return true;
}

// Hàm trích xuất landmark bằng ONNX Model
static FaceCacheItem run_landmark_inference(NvBufSurface *src_surface, int batch_id, float fl, float ft, float fw, float fh, bool& success) {
    FaceCacheItem item;
    success = false;

    if (!src_surface || fw < 28 || fh < 28) return item;

    // Lấy kích thước gốc từ surface
    int img_w = src_surface->surfaceList[batch_id].width;
    int img_h = src_surface->surfaceList[batch_id].height;

    // Tính toán Crop với padding tương ứng bản Python
    float pad_x = 0.28f * fw;
    float pad_y_top = 0.45f * fh;
    float pad_y_bot = 0.18f * fh;

    int x1 = std::max(0, (int)(fl - pad_x));
    int y1 = std::max(0, (int)(ft - pad_y_top));
    int x2 = std::min(img_w, (int)(fl + fw + pad_x));
    int y2 = std::min(img_h, (int)(ft + fh + pad_y_bot));

    int cw = x2 - x1;
    int ch = y2 - y1;

    if (cw <= 8 || ch <= 8) return item;

    // Letterbox: giữ nguyên tỷ lệ crop khi đưa vào 192x192 (FaceMesh train trên crop vuông,
    // resize méo tỷ lệ làm landmark lệch hệ thống, đặc biệt là pitch)
    float lb_scale = 192.0f / (float)std::max(cw, ch);
    int dst_w = std::max(2, (int)roundf(cw * lb_scale));
    int dst_h = std::max(2, (int)roundf(ch * lb_scale));
    int off_x = (192 - dst_w) / 2;
    int off_y = (192 - dst_h) / 2;

    // Crop/resize bằng CPU (OpenCV) thay vì NvBufSurfTransform GPU compute mode.
    // NvBufSurfTransform GPU chạy trên stream/context nội bộ không đồng bộ đầy đủ
    // với FaceMeshTRT's stream riêng — dù đã thêm cudaDeviceSynchronize() vẫn còn
    // ~1/3 trường hợp CPU đọc phải dữ liệu rỗng/rác dù transform báo thành công
    // (xác nhận bằng crop dump). CPU path chậm hơn một chút cho vùng crop nhỏ
    // (~100x100px) nhưng hoàn toàn đồng bộ, không còn phụ thuộc đoán định GPU race.
    if (NvBufSurfaceMap(src_surface, batch_id, -1, NVBUF_MAP_READ) != 0) {
        std::cerr << "[Probe] Failed to map source surface for CPU crop." << std::endl;
        return item;
    }
    NvBufSurfaceSyncForCpu(src_surface, batch_id, -1);
    {
        // Nguồn là RGBA (memory:NVMM), cấu hình bởi caps_rgba trong pipeline.cpp
        cv::Mat full_frame(img_h, img_w, CV_8UC4,
                            src_surface->surfaceList[batch_id].mappedAddr.addr[0],
                            src_surface->surfaceList[batch_id].pitch);
        cv::Mat crop_rgba = full_frame(cv::Rect(x1, y1, cw, ch));

        cv::Mat crop_rgb_raw;
        cv::cvtColor(crop_rgba, crop_rgb_raw, cv::COLOR_RGBA2RGB);

        // Canvas đen 192x192, resize crop giữ tỷ lệ vào giữa (letterbox)
        cv::Mat crop_rgb = cv::Mat::zeros(192, 192, CV_8UC3);
        cv::Mat resized;
        cv::resize(crop_rgb_raw, resized, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);
        resized.copyTo(crop_rgb(cv::Rect(off_x, off_y, dst_w, dst_h)));

        NvBufSurfaceUnMap(src_surface, batch_id, -1);

        // Chuẩn bị dữ liệu vào cho model (swapRB = false vì ảnh crop đã là RGB)
        cv::Mat blob_img = cv::dnn::blobFromImage(crop_rgb, 1.0 / 255.0, cv::Size(192, 192), cv::Scalar(0,0,0), false, false);

        float score = 0.0f;
        std::vector<float> lm(1404);
        bool run_ok = g_face_mesh_trt.infer(blob_img.ptr<float>(), lm.data(), &score);

        // Score model trả về là raw logit → đưa qua sigmoid để có confidence [0,1]
        float conf = 1.0f / (1.0f + expf(-score));

        if (run_ok && std::isfinite(score) && conf >= LANDMARK_CONF_THRESH) {
            // Map ngược tọa độ từ không gian 192x192 (đã letterbox) về khung hình gốc.
            // Z không có gốc tuyệt đối (chỉ là độ sâu tương đối quanh tâm mặt) nên chỉ chia
            // theo lb_scale để cùng đơn vị pixel với X/Y, không cộng offset x1/y1/off.
            // Đảo dấu Z để khớp quy ước "phía trước mặt = +Z" của model 3D phía Python
            // (engine trả mũi lồi về phía camera là Z âm).
            auto get_pt = [&](int idx) -> std::vector<float> {
                float x_crop = lm[idx * 3 + 0];
                float y_crop = lm[idx * 3 + 1];
                float z_crop = lm[idx * 3 + 2];
                float x_orig = x1 + (x_crop - off_x) / lb_scale;
                float y_orig = y1 + (y_crop - off_y) / lb_scale;
                float z_orig = -z_crop / lb_scale;
                return { roundf(x_orig * 10.0f) / 10.0f, roundf(y_orig * 10.0f) / 10.0f, roundf(z_orig * 10.0f) / 10.0f };
            };

            item.nose = get_pt(1);
            item.chin = get_pt(152);

            std::vector<float> l33 = get_pt(33);
            std::vector<float> l133 = get_pt(133);
            item.left_eye = { roundf((l33[0] + l133[0]) * 5.0f) / 10.0f,
                               roundf((l33[1] + l133[1]) * 5.0f) / 10.0f,
                               roundf((l33[2] + l133[2]) * 5.0f) / 10.0f };

            std::vector<float> r263 = get_pt(263);
            std::vector<float> r362 = get_pt(362);
            item.right_eye = { roundf((r263[0] + r362[0]) * 5.0f) / 10.0f,
                                roundf((r263[1] + r362[1]) * 5.0f) / 10.0f,
                                roundf((r263[2] + r362[2]) * 5.0f) / 10.0f };

            item.bridge = get_pt(168);
            item.left_cheek = get_pt(234);
            item.right_cheek = get_pt(454);

            success = sanity_check_landmarks(item, fl, ft, fw, fh);
            if (!success) {
                std::cerr << "[Probe] Landmark sanity check failed (conf=" << conf << "), dropping frame result." << std::endl;
            }
        } else if (run_ok) {
            // Bình thường: mặt bị che khuất, xa, thiếu sáng... Nếu tần suất bất thường
            // cao trở lại và raw_score luôn đúng -8.15625 (chữ ký input rỗng) dù ánh
            // sáng/góc mặt tốt, xem lại memory facemesh-trt-quirks — trước khi chuyển
            // crop sang CPU (bản này) đó là dấu hiệu race condition GPU, không phải
            // model thật sự không nhận ra mặt.
            std::cerr << "[Probe] Landmark rejected: raw_score=" << score << " conf=" << conf
                       << " (thresh=" << LANDMARK_CONF_THRESH << ")" << std::endl;
        }
    }

    return item;
}

// Helper để sinh chuỗi JSON cho Landmark điểm
static std::string format_points_json(const FaceCacheItem& item) {
    auto pt_json = [](const std::vector<float>& p) {
        std::stringstream s;
        s << "[" << p[0] << "," << p[1] << "," << p[2] << "]";
        return s.str();
    };
    std::stringstream ss;
    ss << "{\"nose\":" << pt_json(item.nose)
       << ",\"chin\":" << pt_json(item.chin)
       << ",\"left_eye\":" << pt_json(item.left_eye)
       << ",\"right_eye\":" << pt_json(item.right_eye)
       << ",\"bridge\":" << pt_json(item.bridge)
       << ",\"left_cheek\":" << pt_json(item.left_cheek)
       << ",\"right_cheek\":" << pt_json(item.right_cheek) << "}";
    return ss.str();
}

// GStreamer Pad Probe Callback chính
GstPadProbeReturn sgie_src_pad_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    if (!buf) {
        return GST_PAD_PROBE_OK;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    // Fallback nếu ntp_timestamp per-frame không có (không nên xảy ra vì nvstreammux
    // mặc định attach-sys-ts=true, nhưng phòng trường hợp cấu hình đổi)
    auto now = std::chrono::system_clock::now();
    long long fallback_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    // Map NvBufSurface từ GStreamer Buffer
    NvBufSurface *surface = nullptr;
    GstMapInfo map_info;
    bool surface_mapped = false;

    if (gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        surface = (NvBufSurface *)map_info.data;
        surface_mapped = true;
    }

    // Duyệt qua từng Frame trong Batch
    for (NvDsFrameMetaList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        int src_id = frame_meta->source_id;
        int batch_id = frame_meta->batch_id;

        // ntp_timestamp được nvstreammux gắn NGAY LÚC NHẬN frame từ camera (ns, epoch),
        // sớm hơn nhiều so với lúc pad probe này chạy (đã qua PGIE + toàn bộ FaceMesh
        // của các object trước đó trong batch) — dùng nó thay vì giờ hiện tại để "ts"
        // phản ánh đúng thời điểm capture, không cộng dồn độ trễ xử lý AI. Mặc định
        // attach-sys-ts=true trên nvstreammux (pipeline.cpp) nghĩa là đây là giờ hệ
        // thống Jetson lúc nhận frame, KHÔNG phải giờ camera chụp (cần RTCP NTP từ
        // camera + attach-sys-ts=false mới có giờ camera thật, tùy camera có hỗ trợ).
        long long timestamp_ms = frame_meta->ntp_timestamp > 0
            ? (long long)(frame_meta->ntp_timestamp / 1000000ULL)
            : fallback_timestamp_ms;

        // Cập nhật frame counter
        uint64_t fc = 0;
        {
            std::lock_guard<std::mutex> lock(g_counters_mutex);
            g_frame_counters[src_id]++;
            fc = g_frame_counters[src_id];
        }

        bool do_detect = (fc % PROCESS_EVERY_N_FRAMES == 0);

        // Định dạng chuỗi JSON cho danh sách detections
        std::stringstream det_ss;
        bool first_det = true;

        // Duyệt qua danh sách đối tượng phát hiện được trong Frame
        for (NvDsObjectMetaList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

            // Kiểm tra xem đối tượng có phải khuôn mặt (face class ID=2 hoặc label)
            if (obj_meta->class_id == 2 || strcmp(obj_meta->obj_label, "face") == 0) {
                float fl = obj_meta->rect_params.left;
                float ft = obj_meta->rect_params.top;
                float fw = obj_meta->rect_params.width;
                float fh = obj_meta->rect_params.height;

                // Lấy ID đối tượng (nếu tracking bị mất ID sẽ tự tính bằng tọa độ lưới)
                uint64_t obj_id = obj_meta->object_id;
                if (obj_id == UINT64_MAX_VAL) {
                    obj_id = ((int)(fl + fw * 0.5f) / ID_GRID_PX) * 1000 + (int)(ft + fh * 0.5f) / ID_GRID_PX;
                }

                std::pair<int, int> cache_key = {src_id, (int)obj_id};
                FaceCacheItem pts;
                bool pts_valid = false;

                if (do_detect && surface_mapped && surface) {
                    bool inf_ok = false;
                    FaceCacheItem item = run_landmark_inference(surface, batch_id, fl, ft, fw, fh, inf_ok);
                    if (inf_ok) {
                        item.last_seen = fc;
                        std::lock_guard<std::mutex> lock(g_cache_mutex);
                        g_pose_cache[cache_key] = item;
                        pts = item;
                        pts_valid = true;
                    }
                }

                // Nếu không chạy detect ở frame này, lấy từ cache
                if (!pts_valid) {
                    std::lock_guard<std::mutex> lock(g_cache_mutex);
                    auto it = g_pose_cache.find(cache_key);
                    // Chỉ dùng lại cache còn đủ mới; landmark quá cũ làm góc "đóng băng"
                    // sai vị trí khi người đã di chuyển mà inference đang fail liên tục
                    if (it != g_pose_cache.end() && fc - it->second.last_seen <= MAX_CACHE_STALENESS) {
                        pts = it->second;
                        pts_valid = true;
                    }
                }

                // Append detection vào JSON
                if (!first_det) det_ss << ",";
                first_det = false;

                det_ss << "{\"box\":[" << (int)fl << "," << (int)ft << "," << (int)(fl + fw) << "," << (int)(ft + fh) << "]"
                       << ",\"score\":" << roundf(obj_meta->confidence * 100.0f) / 100.0f
                       << ",\"id\":" << obj_id;

                if (pts_valid) {
                    det_ss << ",\"pts\":" << format_points_json(pts);
                }
                det_ss << "}";
            }
        }


        // Serialize và gửi MQTT
        std::stringstream payload_ss;
        payload_ss << "{\"f\":" << fc
                   << ",\"ts\":" << timestamp_ms
                   << ",\"w\":1920,\"h\":1080"
                   << ",\"d\":[" << det_ss.str() << "]}";

        std::string payload = payload_ss.str();
        std::string topic = "gaze/cam" + std::to_string(src_id) + "/metadata";

        if (g_mosq) {
            mosquitto_publish(g_mosq, nullptr, topic.c_str(), payload.length(), payload.c_str(), 0, false);
        }
    }

    if (surface_mapped) {
        gst_buffer_unmap(buf, &map_info);
    }

    // Dọn dẹp cache cũ định kỳ (mỗi 300 frames)
    static uint64_t global_fc = 0;
    global_fc++;
    if (global_fc % 300 == 0) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (auto it = g_pose_cache.cbegin(); it != g_pose_cache.cend();) {
            int src_id = it->first.first;
            uint64_t current_fc = 0;
            {
                std::lock_guard<std::mutex> c_lock(g_counters_mutex);
                current_fc = g_frame_counters[src_id];
            }
            if (current_fc - it->second.last_seen > 60) {
                it = g_pose_cache.erase(it);
            } else {
                ++it;
            }
        }
    }

    return GST_PAD_PROBE_OK;
}
