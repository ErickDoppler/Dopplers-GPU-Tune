// Dopplers GPU Tune: finds the best VRAM clock for AI work on an NVIDIA card.
// A slider sets the memory clock offset (-500 .. +750 MHz) through the NVIDIA driver (NVML), TEST runs an LLM benchmark (token generation is memory bound)
// inside this program with the llama.cpp runtime from runtime\, samples the REAL clocks while it runs, and adds a dot to the graph:
//   x = the VRAM clock the card actually ran at,  y = tokens per second.
// The power limit makes the GPU core clock drop when the memory clock goes up; the list next to the graph shows the core clock and power of every dot,
// so the best combination for AI work can be read off directly. The test model can be downloaded from inside the program.
//   -DDRY_RUN  never writes an offset (for testing the window and the measurement without touching the card)
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>
#include <tlhelp32.h>
#include <winhttp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "llama.h"                                   // only the declarations: the DLL is loaded by hand from runtime\

namespace {
// ============================================================================ NVML (from the NVIDIA driver)
struct ClockOffset { unsigned version; int type; int pstate; int offset; int minOffset; int maxOffset; };
struct Util { unsigned gpu, mem; };
HMODULE g_nvml = nullptr; void* g_dev = nullptr; char g_gpuName[96] = "GPU";
int (*pInit)() = nullptr; int (*pHandle)(unsigned, void**) = nullptr; int (*pClock)(void*, int, unsigned*) = nullptr; int (*pPower)(void*, unsigned*) = nullptr;
int (*pUtil)(void*, Util*) = nullptr; int (*pGetOff)(void*, ClockOffset*) = nullptr; int (*pSetOff)(void*, ClockOffset*) = nullptr;
const char* (*pErr)(int) = nullptr; int (*pName)(void*, char*, unsigned) = nullptr;
template <class T> bool bind(T& f, const char* n) { f = (T)GetProcAddress(g_nvml, n); return f != nullptr; }
bool nvmlOpen(std::wstring& why) {
    g_nvml = LoadLibraryW(L"nvml.dll");
    if (!g_nvml) g_nvml = LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if (!g_nvml) { why = L"nvml.dll (NVIDIA driver) not found: this tool needs an NVIDIA GPU."; return false; }
    if (!(bind(pInit, "nvmlInit_v2") && bind(pHandle, "nvmlDeviceGetHandleByIndex_v2") && bind(pClock, "nvmlDeviceGetClockInfo") && bind(pPower, "nvmlDeviceGetPowerUsage") &&
          bind(pUtil, "nvmlDeviceGetUtilizationRates") && bind(pGetOff, "nvmlDeviceGetClockOffsets") && bind(pSetOff, "nvmlDeviceSetClockOffsets") && bind(pErr, "nvmlErrorString"))) {
        why = L"This driver is too old for clock offsets (nvmlDeviceSetClockOffsets missing). Update the NVIDIA driver."; return false; }
    bind(pName, "nvmlDeviceGetName");
    if (pInit() != 0 || pHandle(0, &g_dev) != 0) { why = L"Cannot open the GPU through NVML."; return false; }
    if (pName) pName(g_dev, g_gpuName, sizeof g_gpuName);
    return true;
}
bool getMemOffset(int pstate, int& v) {
    ClockOffset c = {}; c.version = (unsigned)(sizeof c) | (1u << 24); c.type = 2; c.pstate = pstate;      // type 2 = memory clock
    if (pGetOff(g_dev, &c) != 0) return false; v = c.offset; return true;
}
// sets the memory clock offset for the performance states a compute load runs in; the read-back proves what the driver accepted
bool setMemOffset(int v, std::wstring& err) {
#ifdef DRY_RUN
    (void)err; (void)v; return true;
#else
    for (int ps = 0; ps <= 2; ps++) {
        ClockOffset c = {}; c.version = (unsigned)(sizeof c) | (1u << 24); c.type = 2; c.pstate = ps; c.offset = v;
        const int r = pSetOff(g_dev, &c);
        if (r != 0 && ps == 0) { wchar_t b[220]; swprintf(b, 220, L"The driver refused the offset: %hs (code %d). Run this program as Administrator.", pErr(r), r); err = b; return false; }
    }
    int back = 0; if (getMemOffset(0, back) && back != v) { wchar_t b[200]; swprintf(b, 200, L"Asked for %d, the driver reports %d.", v, back); err = b; return false; }
    return true;
#endif
}

// ============================================================================ llama.cpp (loaded from runtime\)
#define LL(name) decltype(&name) p_##name = nullptr
LL(llama_backend_init); LL(llama_log_set); LL(llama_model_default_params); LL(llama_model_load_from_file); LL(llama_model_free); LL(llama_model_get_vocab);
LL(llama_context_default_params); LL(llama_init_from_model); LL(llama_free); LL(llama_tokenize); LL(llama_batch_get_one); LL(llama_decode);
LL(llama_sampler_chain_default_params); LL(llama_sampler_chain_init); LL(llama_sampler_chain_add); LL(llama_sampler_init_greedy); LL(llama_sampler_sample);
LL(llama_sampler_free); LL(llama_vocab_is_eog); LL(llama_get_memory); LL(llama_memory_clear);
HMODULE g_llama = nullptr;
void quietLog(ggml_log_level, const char*, void*) {}
std::string utf8(const std::wstring& w) { if (w.empty()) return ""; int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr); std::string s(n, 0); WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr); return s; }

std::wstring g_dir, g_modelDir, g_csv, g_ini;
bool llamaOpen(std::wstring& why) {
    if (g_llama) return true;
    const std::wstring rt = g_dir + L"\\runtime";
    if (GetFileAttributesW((rt + L"\\llama.dll").c_str()) == INVALID_FILE_ATTRIBUTES) { why = L"runtime\\llama.dll is missing (the llama.cpp runtime must sit in the runtime folder next to the program)."; return false; }
    SetDllDirectoryW(rt.c_str());
    g_llama = LoadLibraryW((rt + L"\\llama.dll").c_str());
    if (!g_llama) { why = L"Cannot load runtime\\llama.dll (error " + std::to_wstring(GetLastError()) + L")."; return false; }
    bool ok = true;
#define BINDLL(name) do { p_##name = (decltype(p_##name))GetProcAddress(g_llama, #name); if (!p_##name) ok = false; } while (0)
    BINDLL(llama_backend_init); BINDLL(llama_log_set); BINDLL(llama_model_default_params); BINDLL(llama_model_load_from_file); BINDLL(llama_model_free); BINDLL(llama_model_get_vocab);
    BINDLL(llama_context_default_params); BINDLL(llama_init_from_model); BINDLL(llama_free); BINDLL(llama_tokenize); BINDLL(llama_batch_get_one); BINDLL(llama_decode);
    BINDLL(llama_sampler_chain_default_params); BINDLL(llama_sampler_chain_init); BINDLL(llama_sampler_chain_add); BINDLL(llama_sampler_init_greedy); BINDLL(llama_sampler_sample);
    BINDLL(llama_sampler_free); BINDLL(llama_vocab_is_eog); BINDLL(llama_get_memory); BINDLL(llama_memory_clear);
    if (!ok) { why = L"runtime\\llama.dll does not match this program (missing functions)."; return false; }
    // the GPU and CPU backends live in their own DLLs next to llama.dll
    typedef void (*LoadAll)(const char*);
    LoadAll loadAll = (LoadAll)GetProcAddress(GetModuleHandleW(L"ggml.dll"), "ggml_backend_load_all_from_path");
    if (!loadAll) { why = L"runtime\\ggml.dll does not export ggml_backend_load_all_from_path."; return false; }
    loadAll(utf8(rt).c_str());
    p_llama_backend_init(); p_llama_log_set(quietLog, nullptr);
    return true;
}

// the model stays loaded between tests
struct Bench {
    std::wstring path; llama_model* model = nullptr; llama_context* ctx = nullptr; const llama_vocab* vocab = nullptr; std::vector<llama_token> prompt; llama_token filler = 0;
    void unload() { if (ctx) p_llama_free(ctx); if (model) p_llama_model_free(model); ctx = nullptr; model = nullptr; path.clear(); }
    bool load(const std::wstring& file, std::wstring& err) {
        if (model && path == file) return true;
        unload();
        llama_model_params mp = p_llama_model_default_params(); mp.n_gpu_layers = 999;                  // everything on the GPU
        model = p_llama_model_load_from_file(utf8(file).c_str(), mp);
        if (!model) { err = L"Cannot load the model (is the file complete and is the CUDA runtime in the runtime folder?)"; return false; }
        llama_context_params cp = p_llama_context_default_params(); cp.n_ctx = 2048; cp.n_batch = 512; cp.n_threads = 4; cp.n_threads_batch = 4;
        ctx = p_llama_init_from_model(model, cp);
        if (!ctx) { err = L"Cannot create the context (not enough video memory?)"; unload(); return false; }
        vocab = p_llama_model_get_vocab(model);
        // a fixed prompt of about 700 tokens (like a chat turn with its instructions), the same for every test
        std::string text; for (int i = 0; i < 24; i++) text += "You are Baker, the assistant inside a studio that makes images, music, sound effects, speech and short videos on this computer. Chat naturally and keep replies short. ";
        text += "Write a long story about a robot gardener.";
        prompt.resize(text.size() + 16);
        int n = p_llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), prompt.data(), (int32_t)prompt.size(), true, false);
        if (n <= 0) { err = L"Tokenization failed."; unload(); return false; }
        prompt.resize(std::min(n, 700));
        llama_token f = 0; int m = p_llama_tokenize(vocab, " the", 4, &f, 1, false, false); filler = m > 0 ? f : 1;
        path = file; return true;
    }
    // one pass: reads the prompt, then generates nGen tokens (greedy, an end-of-text token is replaced so that exactly nGen are produced)
    bool run(int nGen, double& genTps, double& preTps, std::wstring& err) {
        p_llama_memory_clear(p_llama_get_memory(ctx), true);
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompt.size(); i += 512) {
            const int n = (int)std::min<size_t>(512, prompt.size() - i);
            if (p_llama_decode(ctx, p_llama_batch_get_one(prompt.data() + i, n)) != 0) { err = L"Decoding the prompt failed."; return false; }
        }
        auto t1 = std::chrono::steady_clock::now();
        llama_sampler* smpl = p_llama_sampler_chain_init(p_llama_sampler_chain_default_params()); p_llama_sampler_chain_add(smpl, p_llama_sampler_init_greedy());
        for (int i = 0; i < nGen; i++) {
            llama_token tok = p_llama_sampler_sample(smpl, ctx, -1);
            if (p_llama_vocab_is_eog(vocab, tok)) tok = filler;
            if (p_llama_decode(ctx, p_llama_batch_get_one(&tok, 1)) != 0) { p_llama_sampler_free(smpl); err = L"Decoding failed."; return false; }
        }
        auto t2 = std::chrono::steady_clock::now();
        p_llama_sampler_free(smpl);
        genTps = nGen / std::chrono::duration<double>(t2 - t1).count(); preTps = prompt.size() / std::chrono::duration<double>(t1 - t0).count();
        return true;
    }
} g_bench;

// ============================================================================ the test models and their download (WinHTTP, resumable)
struct Preset { const wchar_t* label; const wchar_t* file; const wchar_t* url; unsigned long long bytes; };
const Preset kPresets[] = {
    {L"Qwen3 4B Instruct Q4_K_M  (2.5 GB, recommended)", L"Qwen3-4B-Instruct-2507-Q4_K_M.gguf", L"https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497281120ull},
    {L"Qwen3 1.7B Q4_K_M  (1.1 GB, quick download)", L"Qwen3-1.7B-Q4_K_M.gguf", L"https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf", 1107409472ull},
    {L"Qwen3 8B Q4_K_M  (5.0 GB, heavier memory load)", L"Qwen3-8B-Q4_K_M.gguf", L"https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf", 5027784512ull},
};
constexpr int kPresetCount = (int)(sizeof kPresets / sizeof kPresets[0]);
enum { WM_DONE = WM_APP + 1, WM_STATUS, WM_DLPROG, WM_DLDONE };
std::atomic<bool> g_dlCancel{false}, g_dlBusy{false}, g_busy{false};
std::atomic<unsigned long long> g_dlDone{0}, g_dlTotal{0};            // progress of the running download (read by the download dialog)
HWND g_wnd;

unsigned long long fileSizeOf(const std::wstring& p) { WIN32_FILE_ATTRIBUTE_DATA d; if (!GetFileAttributesExW(p.c_str(), GetFileExInfoStandard, &d)) return 0; return ((unsigned long long)d.nFileSizeHigh << 32) | d.nFileSizeLow; }

// downloads url to dest (through dest + ".part", resuming an unfinished one); progress goes to the window; returns an error text or empty on success
std::wstring downloadFile(const std::wstring& url, const std::wstring& dest, unsigned long long expected) {
    URL_COMPONENTSW uc = {sizeof uc}; wchar_t host[256] = {}, path[2048] = {};
    uc.lpszHostName = host; uc.dwHostNameLength = 256; uc.lpszUrlPath = path; uc.dwUrlPathLength = 2048;
    if (!WinHttpCrackUrl(url.c_str(), 0, 0, &uc)) return L"Bad download address.";
    HINTERNET ses = WinHttpOpen(L"DopplersGpuTune/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, nullptr, nullptr, 0);
    if (!ses) return L"Cannot start the network session.";
    struct Close { HINTERNET h; ~Close() { if (h) WinHttpCloseHandle(h); } };
    Close cs{ses};
    DWORD redir = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS; WinHttpSetOption(ses, WINHTTP_OPTION_REDIRECT_POLICY, &redir, sizeof redir);
    HINTERNET con = WinHttpConnect(ses, host, uc.nPort, 0); Close cc{con};
    if (!con) return L"Cannot connect to the server.";
    const std::wstring part = dest + L".part";
    unsigned long long have = fileSizeOf(part);
    if (expected && have > expected) { DeleteFileW(part.c_str()); have = 0; }
    HINTERNET req = WinHttpOpenRequest(con, L"GET", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE); Close cr{req};
    if (!req) return L"Cannot open the request.";
    std::wstring hdr = have ? L"Range: bytes=" + std::to_wstring(have) + L"-" : L"";
    if (!WinHttpSendRequest(req, hdr.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : hdr.c_str(), hdr.empty() ? 0 : (DWORD)-1L, nullptr, 0, 0, 0) || !WinHttpReceiveResponse(req, nullptr))
        return L"The server did not answer (" + std::to_wstring(GetLastError()) + L"). Check the internet connection.";
    DWORD status = 0, sz = sizeof status; WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, nullptr, &status, &sz, nullptr);
    if (status == 416 && expected && have == expected) { MoveFileExW(part.c_str(), dest.c_str(), MOVEFILE_REPLACE_EXISTING); return L""; }
    if (status != 200 && status != 206) return L"The server answered with error " + std::to_wstring(status) + L".";
    if (status == 200 && have) have = 0;                                  // the server ignored the range: start again
    wchar_t lenText[32] = {}; DWORD lsz = sizeof lenText; unsigned long long len = 0;
    if (WinHttpQueryHeaders(req, WINHTTP_QUERY_CONTENT_LENGTH, nullptr, lenText, &lsz, nullptr)) len = _wcstoui64(lenText, nullptr, 10);
    const unsigned long long total = expected ? expected : have + len;
    g_dlDone = have; g_dlTotal = total;
    HANDLE f = CreateFileW(part.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, have ? OPEN_ALWAYS : CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return L"Cannot write " + part;
    if (have) SetFilePointer(f, 0, nullptr, FILE_END); else SetEndOfFile(f);
    std::vector<char> buf(1 << 18); unsigned long long done = have; auto last = std::chrono::steady_clock::now();
    for (;;) {
        if (g_dlCancel) { CloseHandle(f); return L"Download cancelled (it can be resumed)."; }
        DWORD got = 0; if (!WinHttpReadData(req, buf.data(), (DWORD)buf.size(), &got)) { CloseHandle(f); return L"The connection broke (" + std::to_wstring(GetLastError()) + L"). Press Download to resume."; }
        if (!got) break;
        DWORD wr = 0; if (!WriteFile(f, buf.data(), got, &wr, nullptr) || wr != got) { CloseHandle(f); return L"Disk write failed (disk full?)."; }
        done += got;
        auto now = std::chrono::steady_clock::now();
        if (now - last > std::chrono::milliseconds(150)) { last = now; g_dlDone = done; g_dlTotal = total; }
    }
    CloseHandle(f);
    if (expected && fileSizeOf(part) != expected) return L"The file is incomplete (" + std::to_wstring(fileSizeOf(part) >> 20) + L" MB of " + std::to_wstring(expected >> 20) + L" MB). Press Download to resume.";
    if (!MoveFileExW(part.c_str(), dest.c_str(), MOVEFILE_REPLACE_EXISTING)) return L"Cannot rename the finished download.";
    return L"";
}

// ============================================================================ data and window
struct Pt { int off; double mem, sm, watts, tps; std::wstring model; };
std::vector<Pt> g_pts;
int g_origOff = 0; bool g_haveOrig = false, g_changed = false, g_abWarned = false;
std::wstring g_customModel;

enum { ID_SLIDER = 101, ID_TEST, ID_RESTORE, ID_CLEAR, ID_APPLY, ID_FACTOR, ID_LIST, ID_LABEL, ID_STATUS, ID_MODEL, ID_DOWNLOAD, ID_BROWSE, ID_PROGRESS, ID_FACTORTXT, ID_MODELTXT };
HWND g_graph, g_slider, g_label, g_status, g_list, g_test, g_apply, g_factor, g_combo, g_dl, g_browse, g_modelLabel;
HFONT g_font, g_bold;

void setStatus(const std::wstring& s) { SetWindowTextW(g_status, s.c_str()); }
int sliderValue() { int v = (int)SendMessageW(g_slider, TBM_GETPOS, 0, 0); return std::min(750, std::max(-500, (int)std::lround(v / 10.0) * 10)); }
int factor() { wchar_t b[16]; GetWindowTextW(g_factor, b, 16); int f = _wtoi(b); return f < 1 ? 1 : (f > 4 ? 4 : f); }
std::wstring modelFile() {                                                // the model file the combo box selects
    const int i = (int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0);
    if (i >= 0 && i < kPresetCount) return g_modelDir + L"\\" + kPresets[i].file;
    return g_customModel;
}
std::wstring baseName(const std::wstring& p) { const size_t k = p.find_last_of(L"\\/"); return k == std::wstring::npos ? p : p.substr(k + 1); }
void updateLabel() { wchar_t b[200]; const int v = sliderValue(); swprintf(b, 200, L"VRAM clock offset:  %+d MHz     (driver value %+d)", v, v * factor()); SetWindowTextW(g_label, b); }
void refreshCombo() {                                                      // item texts show which models are on the disk
    const int sel = (int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0);
    SendMessageW(g_combo, CB_RESETCONTENT, 0, 0);
    for (int i = 0; i < kPresetCount; i++) {
        const bool have = fileSizeOf(g_modelDir + L"\\" + kPresets[i].file) == kPresets[i].bytes;
        SendMessageW(g_combo, CB_ADDSTRING, 0, (LPARAM)((std::wstring(kPresets[i].label) + (have ? L"   [installed]" : L"   [not downloaded]")).c_str()));
    }
    if (!g_customModel.empty()) SendMessageW(g_combo, CB_ADDSTRING, 0, (LPARAM)(L"Own file: " + baseName(g_customModel)).c_str());
    SendMessageW(g_combo, CB_SETCURSEL, sel < 0 ? 0 : sel, 0);
}
bool modelReady() { const std::wstring f = modelFile(); const int i = (int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0); if (i >= 0 && i < kPresetCount) return fileSizeOf(f) == kPresets[i].bytes; return fileSizeOf(f) > 0; }
void updateDlButton() {
    const int i = (int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0);
    SetWindowTextW(g_dl, L"Download...");
    (void)i;
}
std::vector<const Pt*> visiblePts() { std::vector<const Pt*> v; const std::wstring m = baseName(modelFile()); for (auto& p : g_pts) if (p.model == m) v.push_back(&p); return v; }
void refreshList() {
    SendMessageW(g_list, LB_RESETCONTENT, 0, 0); int n = 0;
    for (auto* p : visiblePts()) { wchar_t b[200]; swprintf(b, 200, L"#%d  %+d  ->  %.0f MHz   %.1f tok/s   core %.0f   %.0f W", ++n, p->off, p->mem, p->tps, p->sm, p->watts); SendMessageW(g_list, LB_ADDSTRING, 0, (LPARAM)b); }
    InvalidateRect(g_graph, nullptr, FALSE);
}
void saveCsv() {
    FILE* f = _wfopen(g_csv.c_str(), L"wb"); if (!f) return;
    fprintf(f, "offset,mem_mhz,core_mhz,watts,tok_s,model\r\n");
    for (auto& p : g_pts) fprintf(f, "%d,%.0f,%.0f,%.0f,%.2f,%s\r\n", p.off, p.mem, p.sm, p.watts, p.tps, utf8(p.model).c_str());
    fclose(f);
}
void loadCsv() {
    FILE* f = _wfopen(g_csv.c_str(), L"rb"); if (!f) return;
    char line[512]; fgets(line, sizeof line, f);
    while (fgets(line, sizeof line, f)) {
        Pt p; char model[300] = {};
        if (sscanf(line, "%d,%lf,%lf,%lf,%lf,%299[^\r\n]", &p.off, &p.mem, &p.sm, &p.watts, &p.tps, model) >= 5) {
            int n = MultiByteToWideChar(CP_UTF8, 0, model, -1, nullptr, 0); p.model.assign(n > 0 ? n - 1 : 0, 0); if (n > 1) MultiByteToWideChar(CP_UTF8, 0, model, -1, &p.model[0], n);
            g_pts.push_back(p);
        }
    }
    fclose(f);
}

// ---------------------------------------------------------------------------- the graph
std::wstring nice(double v) { wchar_t b[32]; swprintf(b, 32, L"%.0f", v); return b; }
void paintGraph(HDC dc, RECT rc) {
    HBRUSH bg = CreateSolidBrush(RGB(255, 255, 255)); FillRect(dc, &rc, bg); DeleteObject(bg);
    SetBkMode(dc, TRANSPARENT); SelectObject(dc, g_font);
    const int L = 70, R = 24, T = 40, B = 52;
    RECT pl = {rc.left + L, rc.top + T, rc.right - R, rc.bottom - B};
    SetTextColor(dc, RGB(30, 30, 30)); SelectObject(dc, g_bold);
    std::wstring title = L"LLM speed against the VRAM clock the card really ran at   -   " + baseName(modelFile());
    TextOutW(dc, rc.left + L, rc.top + 8, title.c_str(), (int)title.size());
    SelectObject(dc, g_font);
    const std::vector<const Pt*> pts = visiblePts();
    double x0 = 9000, x1 = 11000, y0 = 0, y1 = 200;
    if (!pts.empty()) {
        double xa = 1e9, xb = -1e9, ya = 1e9, yb = -1e9;
        for (auto* p : pts) { xa = std::min(xa, p->mem); xb = std::max(xb, p->mem); ya = std::min(ya, p->tps); yb = std::max(yb, p->tps); }
        double xs = std::max(xb - xa, 300.0), ys = std::max(yb - ya, yb * 0.08);
        x0 = (xa + xb) / 2 - xs * 0.65; x1 = (xa + xb) / 2 + xs * 0.65; y0 = (ya + yb) / 2 - ys * 0.8; y1 = (ya + yb) / 2 + ys * 0.8;
    }
    auto X = [&](double v) { return pl.left + (int)((v - x0) / (x1 - x0) * (pl.right - pl.left)); };
    auto Y = [&](double v) { return pl.bottom - (int)((v - y0) / (y1 - y0) * (pl.bottom - pl.top)); };
    HPEN grid = CreatePen(PS_SOLID, 1, RGB(225, 228, 232)), axis = CreatePen(PS_SOLID, 1, RGB(90, 90, 90)); HGDIOBJ old = SelectObject(dc, grid);
    SetTextColor(dc, RGB(90, 90, 90));
    for (int i = 0; i <= 5; i++) {
        const int gx = pl.left + (pl.right - pl.left) * i / 5, gy = pl.bottom - (pl.bottom - pl.top) * i / 5;
        MoveToEx(dc, gx, pl.top, nullptr); LineTo(dc, gx, pl.bottom); MoveToEx(dc, pl.left, gy, nullptr); LineTo(dc, pl.right, gy);
        std::wstring sx = nice(x0 + (x1 - x0) * i / 5), sy = nice(y0 + (y1 - y0) * i / 5);
        TextOutW(dc, gx - 16, pl.bottom + 6, sx.c_str(), (int)sx.size()); TextOutW(dc, pl.left - 8 - 9 * (int)sy.size(), gy - 8, sy.c_str(), (int)sy.size());
    }
    SelectObject(dc, axis); MoveToEx(dc, pl.left, pl.top, nullptr); LineTo(dc, pl.left, pl.bottom); LineTo(dc, pl.right, pl.bottom);
    TextOutW(dc, pl.left + (pl.right - pl.left) / 2 - 120, rc.bottom - 24, L"VRAM clock under load (MHz, measured)", 37);
    TextOutW(dc, pl.left - 60, pl.top - 24, L"tok/s", 5);
    if (pts.empty()) { SetTextColor(dc, RGB(130, 130, 130)); TextOutW(dc, pl.left + 40, (pl.top + pl.bottom) / 2 - 8, L"Move the slider and press TEST: every test adds a dot.", 54); }
    std::vector<const Pt*> s = pts; std::sort(s.begin(), s.end(), [](const Pt* a, const Pt* b) { return a->mem < b->mem; });
    HPEN line = CreatePen(PS_SOLID, 2, RGB(150, 190, 235)); SelectObject(dc, line);
    for (size_t i = 0; i < s.size(); i++) { if (i == 0) MoveToEx(dc, X(s[i]->mem), Y(s[i]->tps), nullptr); else LineTo(dc, X(s[i]->mem), Y(s[i]->tps)); }
    size_t best = 0; for (size_t i = 0; i < pts.size(); i++) if (pts[i]->tps > pts[best]->tps) best = i;
    for (size_t i = 0; i < pts.size(); i++) {
        const Pt& p = *pts[i]; const bool isBest = i == best;
        HBRUSH br = CreateSolidBrush(isBest ? RGB(30, 160, 70) : RGB(30, 110, 220)); HPEN pn = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
        SelectObject(dc, br); SelectObject(dc, pn); const int r = isBest ? 8 : 6;
        Ellipse(dc, X(p.mem) - r, Y(p.tps) - r, X(p.mem) + r, Y(p.tps) + r);
        SelectObject(dc, old); DeleteObject(br); DeleteObject(pn);
        bool crowded = false;                                                  // a dot that sits on top of an earlier one gets no label (the list has every value)
        for (size_t j = 0; j < i; j++) if (std::abs(X(pts[j]->mem) - X(p.mem)) < 80 && std::abs(Y(pts[j]->tps) - Y(p.tps)) < 36) crowded = true;
        if (crowded && !isBest) continue;
        wchar_t t[64]; swprintf(t, 64, L"%+d  (%.1f)", p.off, p.tps); wchar_t t2[40]; swprintf(t2, 40, L"core %.0f", p.sm);
        SetTextColor(dc, isBest ? RGB(20, 120, 50) : RGB(60, 60, 60)); TextOutW(dc, X(p.mem) - 28, Y(p.tps) - 26, t, (int)wcslen(t));
        if (!crowded) { SetTextColor(dc, RGB(130, 130, 130)); TextOutW(dc, X(p.mem) - 22, Y(p.tps) + 10, t2, (int)wcslen(t2)); }
    }
    SelectObject(dc, old); DeleteObject(grid); DeleteObject(axis); DeleteObject(line);
}
LRESULT CALLBACK graphProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    if (m == WM_ERASEBKGND) return 1;
    if (m == WM_PAINT) {
        PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); RECT rc; GetClientRect(h, &rc);
        HDC mem = CreateCompatibleDC(dc); HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom); HGDIOBJ ob = SelectObject(mem, bmp);
        paintGraph(mem, rc); BitBlt(dc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, ob); DeleteObject(bmp); DeleteDC(mem); EndPaint(h, &ps); return 0;
    }
    return DefWindowProcW(h, m, wp, lp);
}

// ---------------------------------------------------------------------------- the test
struct Result { bool ok = false; std::wstring msg; Pt pt; };
bool afterburnerRunning() {
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0); PROCESSENTRY32W e = {sizeof e}; bool found = false;
    if (s != INVALID_HANDLE_VALUE) { for (BOOL ok = Process32FirstW(s, &e); ok; ok = Process32NextW(s, &e)) if (!_wcsicmp(e.szExeFile, L"MSIAfterburner.exe")) found = true; CloseHandle(s); }
    return found;
}
void postStatus(const std::wstring& s) { PostMessageW(g_wnd, WM_STATUS, 0, (LPARAM) new std::wstring(s)); }
Result runTest(int offset, bool apply, int fac, const std::wstring& model) {
    Result r; r.pt.off = offset; r.pt.model = baseName(model);
    std::wstring err;
    if (!llamaOpen(err)) { r.msg = err; return r; }
    if (apply) {
        if (!setMemOffset(offset * fac, err)) { r.msg = err; return r; }
        g_changed = true; Sleep(1500);                                     // let the memory clock settle
    }
    if (!g_bench.model || g_bench.path != model) { postStatus(L"Loading the model into video memory..."); if (!g_bench.load(model, err)) { r.msg = err; return r; } double a, b; g_bench.run(24, a, b, err); }
    postStatus(L"Measuring (3 runs of 256 tokens)...");
    std::atomic<bool> stop{false}; std::vector<Pt> samples;
    std::thread sampler([&] {                                              // the REAL clocks while the model generates
        while (!stop) { unsigned m = 0, c = 0, w = 0; Util u = {}; pClock(g_dev, 2, &m); pClock(g_dev, 1, &c); pPower(g_dev, &w); pUtil(g_dev, &u);
            if (u.gpu >= 50) { Pt p; p.off = 0; p.mem = m; p.sm = c; p.watts = w / 1000.0; p.tps = 0; samples.push_back(p); } Sleep(50); }
    });
    double gsum = 0; int ok = 0;
    for (int i = 0; i < 3; i++) { double g = 0, pr = 0; if (!g_bench.run(256, g, pr, err)) { r.msg = err; break; } gsum += g; ok++; }
    stop = true; sampler.join();
    if (!ok) { if (r.msg.empty()) r.msg = L"The benchmark did not run."; return r; }
    double m = 0, c = 0, w = 0; for (auto& s : samples) { m += s.mem; c += s.sm; w += s.watts; }
    const double n = (double)samples.size();
    r.pt.mem = n ? m / n : 0; r.pt.sm = n ? c / n : 0; r.pt.watts = n ? w / n : 0; r.pt.tps = gsum / ok; r.ok = true;
    return r;
}

void layout() {
    RECT rc; GetClientRect(g_wnd, &rc); const int W = rc.right, H = rc.bottom, bottom = 262;
    MoveWindow(g_graph, 10, 10, W - 20 - 430, H - bottom - 10, TRUE);
    MoveWindow(g_list, W - 420, 10, 410, H - bottom - 10, TRUE);
    const int y0 = H - bottom + 6;
    MoveWindow(g_modelLabel, 14, y0 + 4, 80, 22, TRUE);
    MoveWindow(g_combo, 96, y0, 540, 300, TRUE);
    MoveWindow(g_dl, 644, y0 - 1, 150, 28, TRUE);
    MoveWindow(g_browse, 800, y0 - 1, 100, 28, TRUE);
    MoveWindow(g_label, 14, y0 + 42, W - 28, 22, TRUE);
    MoveWindow(g_slider, 8, y0 + 66, W - 16, 48, TRUE);
    MoveWindow(g_test, 14, y0 + 124, 130, 36, TRUE);
    MoveWindow(GetDlgItem(g_wnd, ID_RESTORE), 152, y0 + 124, 230, 36, TRUE);
    MoveWindow(GetDlgItem(g_wnd, ID_CLEAR), 390, y0 + 124, 120, 36, TRUE);
    MoveWindow(g_apply, 530, y0 + 122, W - 550, 22, TRUE);
    MoveWindow(GetDlgItem(g_wnd, ID_FACTORTXT), 530, y0 + 150, 330, 20, TRUE);
    MoveWindow(g_factor, 864, y0 + 148, 36, 22, TRUE);
    MoveWindow(g_status, 14, y0 + 176, W - 28, 70, TRUE);
}
void restoreOffset(bool silent) {
    if (!g_haveOrig || !g_changed) { if (!silent) setStatus(L"Nothing to restore: this tool has not changed the offset."); return; }
    std::wstring err; if (setMemOffset(g_origOff, err)) { g_changed = false; if (!silent) { wchar_t b[120]; swprintf(b, 120, L"The offset is back to the value found at start (%+d)", g_origOff); setStatus(b); } }
    else if (!silent) setStatus(err);
}
void saveIni() {
    WritePrivateProfileStringW(L"gputune", L"model", std::to_wstring((int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0)).c_str(), g_ini.c_str());
    WritePrivateProfileStringW(L"gputune", L"custom", g_customModel.c_str(), g_ini.c_str());
}

// ============================================================================ download dialog: CUDA libraries + test models (same idea as the one in Dopplers Bakery)
// The CUDA libraries are not shipped in the release zip (about 600 MB): they come from the llama.cpp release the program was built against.
const wchar_t* kLlamaTag = L"b12282";
const unsigned long long kLlamaZipBytes = 171031017ull, kCudartZipBytes = 426314199ull;
const wchar_t* kCudaDlls[] = {L"ggml-cuda.dll", L"cudart64_13.dll", L"cublas64_13.dll", L"cublasLt64_13.dll"};
bool cudaInstalled() { for (const wchar_t* n : kCudaDlls) if (GetFileAttributesW((g_dir + L"\\runtime\\" + n).c_str()) == INVALID_FILE_ATTRIBUTES) return false; return true; }

std::mutex g_setupMx; std::wstring g_setupText, g_setupResult; std::atomic<bool> g_setupFinished{false};
void setupText(const std::wstring& s) { std::lock_guard<std::mutex> g(g_setupMx); g_setupText = s; }

std::wstring findFileRec(const std::wstring& dir, const wchar_t* name, int depth = 4) {          // first file called name under dir
    WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd); if (h == INVALID_HANDLE_VALUE) return L"";
    std::vector<std::wstring> subs; std::wstring hit;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) { if (wcscmp(fd.cFileName, L".") && wcscmp(fd.cFileName, L"..")) subs.push_back(fd.cFileName); }
        else if (_wcsicmp(fd.cFileName, name) == 0) { hit = dir + L"\\" + fd.cFileName; break; }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    if (hit.empty() && depth > 0) for (auto& s : subs) { hit = findFileRec(dir + L"\\" + s, name, depth - 1); if (!hit.empty()) break; }
    return hit;
}
void removeTree(std::wstring dir) {                                                                // a folder this program made itself
    dir.push_back(0); dir.push_back(0);
    SHFILEOPSTRUCTW op = {}; op.wFunc = FO_DELETE; op.pFrom = dir.c_str(); op.fFlags = FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI; SHFileOperationW(&op);
}
bool unzipTo(const std::wstring& zip, const std::wstring& dest) {                                  // tar.exe ships with Windows 10 1803+
    CreateDirectoryW(dest.c_str(), nullptr);
    wchar_t sysdir[MAX_PATH]; GetSystemDirectoryW(sysdir, MAX_PATH);
    std::wstring cl = std::wstring(L"\"") + sysdir + L"\\tar.exe\" -xf \"" + zip + L"\" -C \"" + dest + L"\"";
    STARTUPINFOW si = {sizeof si}; PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(nullptr, &cl[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return false;
    WaitForSingleObject(pi.hProcess, INFINITE); DWORD code = 1; GetExitCodeProcess(pi.hProcess, &code); CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return code == 0;
}
std::wstring installCuda() {                                                                       // returns an error text or empty
    const std::wstring rt = g_dir + L"\\runtime", work = rt + L"\\cuda_download";
    CreateDirectoryW(rt.c_str(), nullptr); CreateDirectoryW(work.c_str(), nullptr);
    const std::wstring base = std::wstring(L"https://github.com/ggml-org/llama.cpp/releases/download/") + kLlamaTag + L"/";
    struct Z { std::wstring remote, local, dir; unsigned long long bytes; };
    const Z zs[] = {{std::wstring(L"llama-") + kLlamaTag + L"-bin-win-cuda-13.5-x64.zip", L"llama-cuda.zip", L"x_llama", kLlamaZipBytes},
                    {L"cudart-llama-bin-win-cuda-13.5-x64.zip", L"cudart.zip", L"x_cudart", kCudartZipBytes}};
    int n = 0;
    for (const Z& z : zs) {
        setupText(L"CUDA libraries, archive " + std::to_wstring(++n) + L" of 2");
        std::wstring e = downloadFile(base + z.remote, work + L"\\" + z.local, z.bytes); if (!e.empty()) return e;
    }
    setupText(L"Unpacking the CUDA libraries...");
    for (const Z& z : zs) if (!unzipTo(work + L"\\" + z.local, work + L"\\" + z.dir)) return L"Could not unpack " + z.local + L" (needs Windows 10 version 1803 or newer).";
    const struct { const wchar_t* dll; const wchar_t* from; } items[] = {{L"ggml-cuda.dll", L"x_llama"}, {L"cudart64_13.dll", L"x_cudart"}, {L"cublas64_13.dll", L"x_cudart"}, {L"cublasLt64_13.dll", L"x_cudart"}};
    for (auto& it : items) {
        const std::wstring f = findFileRec(work + L"\\" + it.from, it.dll);
        if (f.empty()) return std::wstring(it.dll) + L" is not in the downloaded archive.";
        if (!MoveFileExW(f.c_str(), (rt + L"\\" + it.dll).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED)) return L"Cannot write runtime\\" + std::wstring(it.dll) + L" (is the program folder writable?).";
    }
    removeTree(work);
    return L"";
}

namespace {
struct Setup { HWND wnd = nullptr, list = nullptr, bar = nullptr, status = nullptr, go = nullptr, close = nullptr; bool running = false; std::thread th; } S;
enum { ID_SGO = 201, ID_SCLOSE };
unsigned long long cudaBytes() { return kLlamaZipBytes + kCudartZipBytes; }
std::wstring mb(unsigned long long b) { wchar_t t[40]; if (b >= (1ull << 30)) swprintf(t, 40, L"%.1f GB", b / 1073741824.0); else swprintf(t, 40, L"%d MB", (int)(b >> 20)); return t; }
bool presetInstalled(int i) { return fileSizeOf(g_modelDir + L"\\" + kPresets[i].file) == kPresets[i].bytes; }
void setupRows(bool keepChecks) {
    bool chk[1 + kPresetCount] = {};
    if (keepChecks) for (int i = 0; i < 1 + kPresetCount; i++) chk[i] = ListView_GetCheckState(S.list, i) != 0;
    ListView_DeleteAllItems(S.list);
    const int sel = (int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0);
    for (int i = 0; i < 1 + kPresetCount; i++) {
        const bool cuda = i == 0, inst = cuda ? cudaInstalled() : presetInstalled(i - 1);
        std::wstring name = cuda ? L"CUDA libraries (NVIDIA GPU runtime for the test)" : kPresets[i - 1].label;
        LVITEMW it = {}; it.mask = LVIF_TEXT; it.iItem = i; it.pszText = &name[0]; ListView_InsertItem(S.list, &it);
        std::wstring sz = mb(cuda ? cudaBytes() : kPresets[i - 1].bytes), st = inst ? L"installed" : L"missing";
        ListView_SetItemText(S.list, i, 1, &sz[0]); ListView_SetItemText(S.list, i, 2, &st[0]);
        const bool def = cuda ? true : (i - 1 == (sel >= 0 && sel < kPresetCount ? sel : 0));
        ListView_SetCheckState(S.list, i, !inst && (keepChecks ? chk[i] : def));
    }
}
void setupWorker(std::vector<int> pick) {
    std::wstring err;
    for (int i : pick) {
        if (i == 0) err = installCuda();
        else {
            const Preset& p = kPresets[i - 1];
            CreateDirectoryW(g_modelDir.c_str(), nullptr); setupText(std::wstring(L"Model ") + p.file);
            err = downloadFile(p.url, g_modelDir + L"\\" + p.file, p.bytes);
        }
        if (!err.empty()) break;
    }
    { std::lock_guard<std::mutex> g(g_setupMx); g_setupResult = err; }
    g_setupFinished = true;
}
LRESULT CALLBACK setupProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    switch (m) {
    case WM_COMMAND:
        if (LOWORD(wp) == ID_SGO && !S.running) {
            std::vector<int> pick; unsigned long long need = 0;
            for (int i = 0; i < 1 + kPresetCount; i++) { const bool inst = i == 0 ? cudaInstalled() : presetInstalled(i - 1); if (ListView_GetCheckState(S.list, i) && !inst) { pick.push_back(i); need += i == 0 ? cudaBytes() : kPresets[i - 1].bytes; } }
            if (pick.empty()) { SetWindowTextW(S.status, L"Nothing is checked (or it is installed already)."); return 0; }
            ULARGE_INTEGER freeB = {}, a = {}, b = {};
            if (GetDiskFreeSpaceExW(g_dir.c_str(), &freeB, &a, &b) && freeB.QuadPart < need + (1ull << 30)) { SetWindowTextW(S.status, (L"Not enough free disk space: " + mb(need) + L" needed, " + mb(freeB.QuadPart) + L" free.").c_str()); return 0; }
            S.running = true; g_dlCancel = false; g_setupFinished = false; g_dlDone = 0; g_dlTotal = 0;
            SetWindowTextW(S.go, L"Working..."); EnableWindow(S.go, FALSE); SetWindowTextW(S.close, L"Cancel");
            if (S.th.joinable()) S.th.join();
            S.th = std::thread(setupWorker, pick);
        } else if (LOWORD(wp) == ID_SCLOSE) { if (S.running) { g_dlCancel = true; SetWindowTextW(S.status, L"Cancelling..."); } else DestroyWindow(h); }
        return 0;
    case WM_TIMER:
        if (S.running) {
            const unsigned long long d = g_dlDone, t = g_dlTotal; std::wstring txt; { std::lock_guard<std::mutex> g(g_setupMx); txt = g_setupText; }
            SendMessageW(S.bar, PBM_SETPOS, t ? (WPARAM)(d * 1000 / t) : 0, 0);
            if (!g_setupFinished) { if (d || t) txt += L"   " + mb(d) + L" / " + mb(t); SetWindowTextW(S.status, txt.c_str()); }
            else {
                if (S.th.joinable()) S.th.join();
                std::wstring err; { std::lock_guard<std::mutex> g(g_setupMx); err = g_setupResult; }
                S.running = false; SendMessageW(S.bar, PBM_SETPOS, err.empty() ? 1000 : 0, 0);
                SetWindowTextW(S.status, err.empty() ? L"Done." : err.c_str()); SetWindowTextW(S.go, L"Download selected"); EnableWindow(S.go, TRUE); SetWindowTextW(S.close, L"Close");
                setupRows(false);
            }
        }
        return 0;
    case WM_CLOSE: if (S.running) { g_dlCancel = true; SetWindowTextW(S.status, L"Cancelling... close again when it stops."); return 0; } DestroyWindow(h); return 0;
    case WM_DESTROY: KillTimer(h, 1); return 0;
    }
    return DefWindowProcW(h, m, wp, lp);
}
}  // namespace

void showSetupDialog(bool first) {
    static bool reg = false; HINSTANCE inst = GetModuleHandleW(nullptr);
    if (!reg) {
        INITCOMMONCONTROLSEX ic = {sizeof ic, ICC_LISTVIEW_CLASSES}; InitCommonControlsEx(&ic);
        WNDCLASSW wc = {}; wc.lpfnWndProc = setupProc; wc.hInstance = inst; wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1); wc.lpszClassName = L"DopplersGpuTuneSetup"; wc.hIcon = (HICON)GetClassLongPtrW(g_wnd, GCLP_HICON);
        RegisterClassW(&wc); reg = true;
    }
    const int W = 700, H = 360; RECT pr; GetWindowRect(g_wnd, &pr);
    RECT r = {0, 0, W, H}; AdjustWindowRect(&r, WS_CAPTION | WS_SYSMENU, FALSE);
    S.running = false;
    S.wnd = CreateWindowExW(WS_EX_DLGMODALFRAME, L"DopplersGpuTuneSetup", first ? L"Welcome - download the files for the test" : L"Download the CUDA libraries and test models", WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
                            pr.left + ((pr.right - pr.left) - (r.right - r.left)) / 2, pr.top + 80, r.right - r.left, r.bottom - r.top, g_wnd, nullptr, inst, nullptr);
    if (!S.wnd) { setStatus(L"Cannot open the download window (error " + std::to_wstring(GetLastError()) + L")."); return; }
    auto mk = [&](const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y, int w, int hgt, int id, DWORD ex = 0) {
        HWND c = CreateWindowExW(ex, cls, text, WS_CHILD | WS_VISIBLE | style, x, y, w, hgt, S.wnd, (HMENU)(INT_PTR)id, inst, nullptr); SendMessageW(c, WM_SETFONT, (WPARAM)g_font, TRUE); return c; };
    mk(L"STATIC", first ? L"The test needs the NVIDIA CUDA libraries and a model. They are not part of the program's download: pick what to download now (the files go next to the program, into runtime\\ and models\\). You can also select your own .gguf with Browse."
                        : L"The CUDA libraries go into runtime\\, the models into models\\ next to the program.", SS_LEFT, 14, 12, W - 28, 56, 0);
    S.list = mk(WC_LISTVIEWW, L"", LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS | WS_TABSTOP, 14, 72, W - 28, 158, 10, WS_EX_CLIENTEDGE);
    ListView_SetExtendedListViewStyle(S.list, LVS_EX_CHECKBOXES | LVS_EX_FULLROWSELECT);
    const wchar_t* cols[] = {L"File", L"Size", L"State"}; const int cw[] = {440, 100, 100};
    for (int i = 0; i < 3; i++) { LVCOLUMNW c = {}; c.mask = LVCF_TEXT | LVCF_WIDTH; c.pszText = (LPWSTR)cols[i]; c.cx = cw[i]; ListView_InsertColumn(S.list, i, &c); }
    setupRows(false);
    S.status = mk(L"STATIC", L"Untick what you do not need.", SS_LEFT | SS_ENDELLIPSIS, 14, 240, W - 28, 20, 0);
    S.bar = mk(PROGRESS_CLASSW, L"", PBS_SMOOTH, 14, 266, W - 28, 20, 0); SendMessageW(S.bar, PBM_SETRANGE32, 0, 1000);
    S.go = mk(L"BUTTON", L"Download selected", BS_DEFPUSHBUTTON | WS_TABSTOP, W - 270, 310, 150, 30, ID_SGO);
    S.close = mk(L"BUTTON", first ? L"Skip" : L"Close", BS_PUSHBUTTON | WS_TABSTOP, W - 110, 310, 96, 30, ID_SCLOSE);
    SetTimer(S.wnd, 1, 200, nullptr);
    EnableWindow(g_wnd, FALSE);
    MSG msg; while (IsWindow(S.wnd) && GetMessageW(&msg, nullptr, 0, 0) > 0) { if (!IsDialogMessageW(S.wnd, &msg)) { TranslateMessage(&msg); DispatchMessageW(&msg); } }
    if (S.th.joinable()) { g_dlCancel = true; S.th.join(); }
    EnableWindow(g_wnd, TRUE); SetForegroundWindow(g_wnd);
}

LRESULT CALLBACK wndProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    switch (m) {
    case WM_HSCROLL: if ((HWND)lp == g_slider) { SendMessageW(g_slider, TBM_SETPOS, TRUE, sliderValue()); updateLabel(); } return 0;
    case WM_COMMAND:
        if (LOWORD(wp) == ID_TEST && !g_busy) {
            if (!cudaInstalled()) { setStatus(L"The CUDA libraries are not installed (without them the test would run on the CPU): press Download..."); return 0; }
            if (!modelReady()) { setStatus(L"The test model is not on the disk yet: select one and press Download... (or Browse... for your own .gguf)."); return 0; }
            const bool apply = SendMessageW(g_apply, BM_GETCHECK, 0, 0) == BST_CHECKED;
            if (apply && !g_abWarned && afterburnerRunning()) {
                g_abWarned = true;
                if (MessageBoxW(h, L"MSI Afterburner is running. Its memory offset and the one set here can add up (the driver reports 0 for Afterburner's value), which can push the memory beyond what you intended.\n\n"
                                   L"Set Afterburner's memory offset to 0 first (or close it), then press TEST again.\n\nContinue anyway?", L"Dopplers GPU Tune", MB_YESNO | MB_ICONWARNING) != IDYES) return 0;
            }
            g_busy = true; EnableWindow(g_test, FALSE); setStatus(apply ? L"Applying the offset..." : L"Starting the benchmark at the current clock...");
            const int off = sliderValue(), fac = factor(); const std::wstring model = modelFile();
            std::thread([off, apply, fac, model] { Result* r = new Result(runTest(off, apply, fac, model)); PostMessageW(g_wnd, WM_DONE, 0, (LPARAM)r); }).detach();
        } else if (LOWORD(wp) == ID_RESTORE) restoreOffset(false);
        else if (LOWORD(wp) == ID_CLEAR) { const std::wstring mname = baseName(modelFile()); g_pts.erase(std::remove_if(g_pts.begin(), g_pts.end(), [&](const Pt& p) { return p.model == mname; }), g_pts.end()); saveCsv(); refreshList(); setStatus(L"Graph cleared (this model)."); }
        else if (LOWORD(wp) == ID_FACTOR && HIWORD(wp) == EN_CHANGE) updateLabel();
        else if (LOWORD(wp) == ID_MODEL && HIWORD(wp) == CBN_SELCHANGE) { saveIni(); updateDlButton(); refreshList(); }
        else if (LOWORD(wp) == ID_BROWSE) {
            wchar_t file[MAX_PATH] = {}; OPENFILENAMEW o = {sizeof o}; o.hwndOwner = h; o.lpstrFilter = L"GGUF models\0*.gguf\0All files\0*.*\0"; o.lpstrFile = file; o.nMaxFile = MAX_PATH; o.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
            if (GetOpenFileNameW(&o)) { g_customModel = file; refreshCombo(); SendMessageW(g_combo, CB_SETCURSEL, kPresetCount, 0); saveIni(); updateDlButton(); refreshList(); }
        } else if (LOWORD(wp) == ID_DOWNLOAD) {
            showSetupDialog(false); refreshCombo(); updateDlButton(); refreshList();
            setStatus(cudaInstalled() && modelReady() ? L"Ready. Move the slider and press TEST." : L"Something is still missing: press Download... to get the CUDA libraries and a test model.");
        }
        return 0;
    case WM_STATUS: { std::wstring* s = (std::wstring*)lp; if (g_busy) setStatus(*s); delete s; return 0; }
    case WM_DONE: {
        Result* r = (Result*)lp; g_busy = false; EnableWindow(g_test, TRUE);
        if (r->ok) {
            g_pts.push_back(r->pt); saveCsv(); refreshList();
            wchar_t b[300]; swprintf(b, 300, L"Dot added:  offset %+d  ->  VRAM %.0f MHz (measured),  %.1f tok/s,  GPU core %.0f MHz,  %.0f W.", r->pt.off, r->pt.mem, r->pt.tps, r->pt.sm, r->pt.watts);
            setStatus(b);
        } else setStatus(L"Not added: " + r->msg);
        delete r; return 0;
    }
    case WM_SIZE: if (g_graph) layout(); return 0;
    case WM_GETMINMAXINFO: ((MINMAXINFO*)lp)->ptMinTrackSize = {1000, 700}; return 0;
    case WM_CLOSE: if (g_busy) return 0; restoreOffset(true); DestroyWindow(h); return 0;
    case WM_DESTROY: g_bench.unload(); PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h, m, wp, lp);
}
}  // namespace

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int show) {
    INITCOMMONCONTROLSEX ic = {sizeof ic, ICC_BAR_CLASSES | ICC_PROGRESS_CLASS}; InitCommonControlsEx(&ic);
    wchar_t p[MAX_PATH]; GetModuleFileNameW(nullptr, p, MAX_PATH); g_dir = p; g_dir.resize(g_dir.find_last_of(L'\\'));
    g_modelDir = g_dir + L"\\models"; g_csv = g_dir + L"\\gputune.csv"; g_ini = g_dir + L"\\gputune.ini";
    std::wstring why; if (!nvmlOpen(why)) { MessageBoxW(nullptr, why.c_str(), L"Dopplers GPU Tune", MB_ICONERROR); return 1; }
    g_haveOrig = getMemOffset(0, g_origOff);
    { wchar_t cu[MAX_PATH] = {}; GetPrivateProfileStringW(L"gputune", L"custom", L"", cu, MAX_PATH, g_ini.c_str()); g_customModel = cu; if (!g_customModel.empty() && fileSizeOf(g_customModel) == 0) g_customModel.clear(); }
    loadCsv();

    g_font = CreateFontW(-15, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
    g_bold = CreateFontW(-16, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
    HICON icon = LoadIconW(inst, MAKEINTRESOURCEW(1));
    WNDCLASSEXW wc = {sizeof wc}; wc.lpfnWndProc = wndProc; wc.hInstance = inst; wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1); wc.lpszClassName = L"DopplersGpuTune"; wc.hIcon = icon; wc.hIconSm = icon;
    RegisterClassExW(&wc);
    WNDCLASSW gc = {}; gc.lpfnWndProc = graphProc; gc.hInstance = inst; gc.hCursor = LoadCursorW(nullptr, IDC_ARROW); gc.lpszClassName = L"DopplersGpuTuneGraph"; RegisterClassW(&gc);
    wchar_t title[200];
#ifdef DRY_RUN
    swprintf(title, 200, L"Dopplers GPU Tune [DRY RUN: offsets are NOT written] - %hs", g_gpuName);
#else
    swprintf(title, 200, L"Dopplers GPU Tune - best VRAM clock for AI - %hs", g_gpuName);
#endif
    g_wnd = CreateWindowW(L"DopplersGpuTune", title, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, 1240, 800, nullptr, nullptr, inst, nullptr);
    auto mk = [&](const wchar_t* cls, const wchar_t* text, DWORD style, int id, DWORD ex = 0) { HWND c = CreateWindowExW(ex, cls, text, WS_CHILD | WS_VISIBLE | style, 0, 0, 10, 10, g_wnd, (HMENU)(INT_PTR)id, inst, nullptr); SendMessageW(c, WM_SETFONT, (WPARAM)g_font, TRUE); return c; };
    g_graph = CreateWindowExW(WS_EX_CLIENTEDGE, L"DopplersGpuTuneGraph", L"", WS_CHILD | WS_VISIBLE, 0, 0, 10, 10, g_wnd, nullptr, inst, nullptr);
    g_list = mk(L"LISTBOX", L"", WS_VSCROLL | LBS_NOINTEGRALHEIGHT | LBS_NOSEL, ID_LIST, WS_EX_CLIENTEDGE);
    g_modelLabel = mk(L"STATIC", L"Test model:", SS_LEFT, ID_MODELTXT);
    g_combo = mk(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, ID_MODEL);
    g_dl = mk(L"BUTTON", L"Download model", BS_PUSHBUTTON | WS_TABSTOP, ID_DOWNLOAD);
    g_browse = mk(L"BUTTON", L"Browse...", BS_PUSHBUTTON | WS_TABSTOP, ID_BROWSE);
    g_label = mk(L"STATIC", L"", SS_LEFT, ID_LABEL); SendMessageW(g_label, WM_SETFONT, (WPARAM)g_bold, TRUE);
    g_slider = mk(TRACKBAR_CLASSW, L"", TBS_HORZ | TBS_AUTOTICKS | TBS_TOOLTIPS, ID_SLIDER);
    SendMessageW(g_slider, TBM_SETRANGE, TRUE, MAKELONG(-500, 750)); SendMessageW(g_slider, TBM_SETTICFREQ, 50, 0); SendMessageW(g_slider, TBM_SETPAGESIZE, 0, 50); SendMessageW(g_slider, TBM_SETLINESIZE, 0, 10);
    SendMessageW(g_slider, TBM_SETPOS, TRUE, g_haveOrig ? std::min(750, std::max(-500, g_origOff)) : 0);
    g_test = mk(L"BUTTON", L"TEST", BS_DEFPUSHBUTTON | WS_TABSTOP, ID_TEST); SendMessageW(g_test, WM_SETFONT, (WPARAM)g_bold, TRUE);
    mk(L"BUTTON", L"Restore the original offset", BS_PUSHBUTTON | WS_TABSTOP, ID_RESTORE);
    mk(L"BUTTON", L"Clear graph", BS_PUSHBUTTON | WS_TABSTOP, ID_CLEAR);
    g_apply = mk(L"BUTTON", L"TEST sets the offset through the NVIDIA driver", BS_AUTOCHECKBOX | WS_TABSTOP, ID_APPLY); SendMessageW(g_apply, BM_SETCHECK, BST_CHECKED, 0);
    mk(L"STATIC", L"Driver units per slider step (1 = same MHz):", SS_LEFT, ID_FACTORTXT);
    g_factor = mk(L"EDIT", L"1", ES_NUMBER | ES_CENTER | WS_TABSTOP, ID_FACTOR, WS_EX_CLIENTEDGE);
    g_status = mk(L"STATIC", L"", SS_LEFT, ID_STATUS);
    refreshCombo(); { int sel = GetPrivateProfileIntW(L"gputune", L"model", 0, g_ini.c_str()); const int count = kPresetCount + (g_customModel.empty() ? 0 : 1); SendMessageW(g_combo, CB_SETCURSEL, sel >= 0 && sel < count ? sel : 0, 0); }
    updateLabel(); updateDlButton(); refreshList(); layout();
    setStatus(modelReady() ? L"Ready. Tip: set MSI Afterburner's memory offset to 0 first, then move the slider here and press TEST."
                           : L"First step: press Download... (the CUDA libraries and the 4B test model), then move the slider and press TEST.");
    ShowWindow(g_wnd, show); UpdateWindow(g_wnd);
    if (!cudaInstalled() || !modelReady()) { showSetupDialog(true); refreshCombo(); updateDlButton(); refreshList(); setStatus(cudaInstalled() && modelReady() ? L"Ready. Move the slider and press TEST." : L"Something is missing: press Download... to get the CUDA libraries and a test model."); }
    MSG msg; while (GetMessageW(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return 0;
}
