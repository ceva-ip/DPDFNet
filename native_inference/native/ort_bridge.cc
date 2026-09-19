// Minimal optional C++ ABI adapter. All model arithmetic lives in the C kernels.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "dpdf_dprnn.h"
#include <memory>
#include <mutex>
#include <stdexcept>

struct Kernel {
    explicit Kernel(const OrtKernelInfo* raw) {
        Ort::ConstKernelInfo info(raw);
        freq = info.GetAttribute<int64_t>("freq");
        if (freq != 40 && freq != 48) throw std::invalid_argument("DPRNN frequency must be 40 or 48");
        auto weights = info.GetAttributes<float>("weights");
        block.reset(dpdf_create(static_cast<int>(freq), weights.data(), weights.size(),
            info.GetAttribute<float>("intra_epsilon"), info.GetAttribute<float>("inter_epsilon"), DPDF_AUTO));
        if (!block) throw std::invalid_argument("Invalid native DPRNN weights or attributes");
    }
    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try {
            Ort::KernelContext ctx(raw);
            auto x = ctx.GetInput(0), state = ctx.GetInput(1);
            const std::vector<int64_t> expected_x{1,64,1,freq}, expected_s{freq*64};
            if (x.GetTensorTypeAndShapeInfo().GetShape() != expected_x ||
                state.GetTensorTypeAndShapeInfo().GetShape() != expected_s)
                throw std::invalid_argument("Native DPRNN input shapes do not match fixed block");
            const int64_t xd[] = {1,64,1,freq}, sd[] = {freq*64};
            auto y = ctx.GetOutput(0, xd, 4), so = ctx.GetOutput(1, sd, 1);
            int rc = dpdf_process(block.get(), x.GetTensorData<float>(), state.GetTensorData<float>(),
                                  y.GetTensorMutableData<float>(), so.GetTensorMutableData<float>());
            if (rc) throw std::runtime_error("Native DPRNN process failed");
            return nullptr;
        } catch (const std::exception& e) {
            return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
        } catch (...) {
            return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown DPRNN error");
        }
    }
    int64_t freq;
    std::unique_ptr<dpdf_block, decltype(&dpdf_destroy)> block{nullptr, dpdf_destroy};
};

struct Op : Ort::CustomOpBase<Op, Kernel, true> {
    const char* GetName() const { return "DpdfDprnn"; }
    size_t GetInputTypeCount() const { return 2; }
    size_t GetOutputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** output) const noexcept {
        try {
            *output = new Kernel(info); return nullptr;
        } catch (const std::exception& e) {
            return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
        } catch (...) {
            return api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown DPRNN initialization error");
        }
    }
};

extern "C" DPDF_API OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "Native DPRNN requires ORT API 27");
    try {
        // Registration happens at session initialization, never on the audio path.
        // The immutable domain/op must outlive all sessions that refer to them.
        static std::mutex registration;
        std::lock_guard<std::mutex> lock(registration);
        static const OrtApi* initialized = nullptr;
        if (initialized && initialized != api)
            return api->CreateStatus(ORT_FAIL, "Mixing ORT runtimes in one native library is unsupported");
        if (!initialized) { Ort::InitApi(api); initialized = api; }
        static Op op;
        static Ort::CustomOpDomain domain = [&]() {
            Ort::CustomOpDomain d("ceva.dpdfnet.experimental"); d.Add(&op); return d;
        }();
        Ort::UnownedSessionOptions opts(options);
        opts.Add(domain);
        return nullptr;
    } catch (const std::exception& e) {
        return api->CreateStatus(ORT_FAIL, e.what());
    } catch (...) {
        return api->CreateStatus(ORT_FAIL, "Unknown DPRNN registration error");
    }
}
