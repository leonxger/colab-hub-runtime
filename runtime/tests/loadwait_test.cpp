// Unit test of patch_loadwait.py (M15-T12) on a CPU-only build of a patched tree; runtime/tests/test_loadwait.sh builds and runs it.
#include "llama.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "gguf.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int g_fails = 0;
static int g_checks = 0;

static void check(bool ok, const std::string & what) {
    g_checks++;
    if (!ok) {
        g_fails++;
    }
    printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    fflush(stdout);
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::mutex g_log_mtx;
static std::string g_log;
static double g_prefill_line_at = 0.0;

static void log_cb(ggml_log_level, const char * text, void *) {
    std::lock_guard<std::mutex> lock(g_log_mtx);
    g_log += text;
    if (strstr(text, "MoE cache prefill after every chunk") != nullptr) {
        g_prefill_line_at = now_s();
    }
    if (getenv("LOADWAIT_VERBOSE") != nullptr) {
        fputs(text, stderr);
    }
}

static std::string log_take() {
    std::lock_guard<std::mutex> lock(g_log_mtx);
    std::string out;
    out.swap(g_log);
    return out;
}

// the file under test and its chunk bytes, for the read counter
struct watch_state {
    std::atomic<bool> on{false};
    dev_t dev = 0;
    ino_t ino = 0;
    const uint8_t * bits = nullptr;
    uint64_t chunk = 1;
    uint64_t size = 0;
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> early{0};
    std::map<std::string, uint64_t> offsets;
};
static watch_state g_watch;

static bool range_ready(uint64_t off, uint64_t len) {
    if (len == 0) {
        return true;
    }
    const uint64_t last = std::min(off + len, g_watch.size) - 1;
    for (uint64_t c = off / g_watch.chunk; c <= last / g_watch.chunk; ++c) {
        if (__atomic_load_n(g_watch.bits + c, __ATOMIC_ACQUIRE) == 0) {
            return false;
        }
    }
    return true;
}

static void note_read(uint64_t off, uint64_t len) {
    g_watch.reads++;
    if (!range_ready(off, len)) {
        g_watch.early++;
    }
}

static bool is_watched_fd(int fd) {
    struct stat st{};
    return g_watch.on && fd >= 0 && fstat(fd, &st) == 0 && st.st_ino == g_watch.ino && st.st_dev == g_watch.dev;
}

// libllama is linked statically, so its stdio and read calls resolve to these counters first
extern "C" size_t fread(void * ptr, size_t size, size_t n, FILE * fp) {
    using fn_t = size_t (*)(void *, size_t, size_t, FILE *);
    static fn_t real = (fn_t) dlsym(RTLD_NEXT, "fread");
    if (fp != nullptr && is_watched_fd(fileno(fp))) {
        const off_t pos = ftello(fp);
        if (pos >= 0) {
            note_read((uint64_t) pos, (uint64_t) size * n);
        }
    }
    return real(ptr, size, n, fp);
}

extern "C" ssize_t read(int fd, void * buf, size_t n) {
    using fn_t = ssize_t (*)(int, void *, size_t);
    static fn_t real = (fn_t) dlsym(RTLD_NEXT, "read");
    if (is_watched_fd(fd)) {
        const off_t pos = lseek(fd, 0, SEEK_CUR);
        if (pos >= 0) {
            note_read((uint64_t) pos, n);
        }
    }
    return real(fd, buf, n);
}

// a buffer type that is not host memory, so the loader takes its copy branches; it checks every copy against the chunk bytes
struct test_buffer {
    uint8_t * mem;
};

static uint64_t g_copies = 0;
static uint64_t g_early_copies = 0;

static void tb_free(ggml_backend_buffer_t buf) {
    auto * tb = (test_buffer *) buf->context;
    free(tb->mem);
    delete tb;
}
static void * tb_base(ggml_backend_buffer_t buf) { return ((test_buffer *) buf->context)->mem; }
static void tb_set(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    if (g_watch.on) {
        auto it = g_watch.offsets.find(ggml_get_name(t));
        g_copies++;
        if (it == g_watch.offsets.end() || !range_ready(it->second + offset, size)) {
            g_early_copies++;
        }
    }
    memcpy((uint8_t *) t->data + offset, data, size);
}
static void tb_get(ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    memcpy(data, (const uint8_t *) t->data + offset, size);
}
static void tb_memset(ggml_backend_buffer_t, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    memset((uint8_t *) t->data + offset, value, size);
}
static void tb_clear(ggml_backend_buffer_t buf, uint8_t value) {
    memset(((test_buffer *) buf->context)->mem, value, ggml_backend_buffer_get_size(buf));
}
static const char * tbt_name(ggml_backend_buffer_type_t) { return "LOADWAIT_TEST"; }
static size_t tbt_align(ggml_backend_buffer_type_t) { return 64; }
static bool tbt_is_host(ggml_backend_buffer_type_t) { return false; }
static ggml_backend_buffer_t tbt_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * tb = new test_buffer;
    tb->mem = (uint8_t *) aligned_alloc(64, (size + 63) / 64 * 64 + 64);
    ggml_backend_buffer_i iface{};
    iface.free_buffer = tb_free;
    iface.get_base = tb_base;
    iface.set_tensor = tb_set;
    iface.get_tensor = tb_get;
    iface.memset_tensor = tb_memset;
    iface.clear = tb_clear;
    return ggml_backend_buffer_init(buft, iface, tb, size);
}

static ggml_backend_buffer_type * test_buft() {
    static ggml_backend_buffer_type buft = [] {
        ggml_backend_buffer_type b{};
        b.iface.get_name = tbt_name;
        b.iface.alloc_buffer = tbt_alloc;
        b.iface.get_alignment = tbt_align;
        b.iface.is_host = tbt_is_host;
        b.device = nullptr;
        b.context = nullptr;
        return b;
    }();
    return &buft;
}

struct fixture {
    std::vector<uint8_t> bytes;
    uint64_t size = 0;
    uint64_t data_offset = 0;
    std::vector<std::string> names;
    std::map<std::string, std::pair<uint64_t, uint64_t>> tensors;
};

struct dims {
    int n_embd;
    int n_ff;
    int n_layer;
    int n_vocab;
};

// a llama-architecture GGUF of F32 tensors in the loader's creation order; pad grows a dummy key so the data region can be shifted
static fixture make_fixture(const dims & d, size_t pad, unsigned seed, bool with_data) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_str(g, "general.name", "loadwait-fixture");
    gguf_set_val_str(g, "general.pad", std::string(pad + 1, 'p').c_str());
    gguf_set_val_u32(g, "llama.context_length", 256);
    gguf_set_val_u32(g, "llama.embedding_length", d.n_embd);
    gguf_set_val_u32(g, "llama.block_count", d.n_layer);
    gguf_set_val_u32(g, "llama.feed_forward_length", d.n_ff);
    gguf_set_val_u32(g, "llama.attention.head_count", 4);
    gguf_set_val_u32(g, "llama.attention.head_count_kv", 4);
    gguf_set_val_f32(g, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_str(g, "tokenizer.ggml.model", "llama");
    std::vector<std::string> toks;
    std::vector<float> scores;
    std::vector<int32_t> types;
    for (int i = 0; i < d.n_vocab; ++i) {
        toks.push_back(i == 0 ? "<unk>" : i == 1 ? "<s>" : i == 2 ? "</s>" : i == 3 ? "<0x0A>" : "t" + std::to_string(i));
        scores.push_back(-(float) i);
        types.push_back(i == 0 ? 2 : i < 3 ? 3 : i == 3 ? 6 : 1);
    }
    std::vector<const char *> ptrs;
    for (const auto & t : toks) {
        ptrs.push_back(t.c_str());
    }
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 0);

    std::vector<std::pair<std::string, std::vector<int64_t>>> specs = {
        {"token_embd.weight", {d.n_embd, d.n_vocab}}, {"output_norm.weight", {d.n_embd}}, {"output.weight", {d.n_embd, d.n_vocab}}};
    for (int l = 0; l < d.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        specs.push_back({b + "attn_norm.weight", {d.n_embd}});
        specs.push_back({b + "attn_q.weight", {d.n_embd, d.n_embd}});
        specs.push_back({b + "attn_k.weight", {d.n_embd, d.n_embd}});
        specs.push_back({b + "attn_v.weight", {d.n_embd, d.n_embd}});
        specs.push_back({b + "attn_output.weight", {d.n_embd, d.n_embd}});
        specs.push_back({b + "ffn_norm.weight", {d.n_embd}});
        specs.push_back({b + "ffn_gate.weight", {d.n_embd, d.n_ff}});
        specs.push_back({b + "ffn_down.weight", {d.n_ff, d.n_embd}});
        specs.push_back({b + "ffn_up.weight", {d.n_embd, d.n_ff}});
    }
    ggml_init_params ip = {ggml_tensor_overhead() * (specs.size() + 4), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    for (const auto & s : specs) {
        ggml_tensor * t = s.second.size() == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.second[0])
                                               : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.second[0], s.second[1]);
        ggml_set_name(t, s.first.c_str());
        gguf_add_tensor(g, t);
    }
    fixture fx;
    fx.data_offset = gguf_get_meta_size(g);
    std::vector<uint8_t> meta(fx.data_offset);
    gguf_get_meta_data(g, meta.data());
    uint64_t end = fx.data_offset;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        const uint64_t off = fx.data_offset + gguf_get_tensor_offset(g, i);
        const uint64_t n = gguf_get_tensor_size(g, i);
        fx.names.push_back(name);
        fx.tensors[name] = {off, n};
        end = std::max(end, off + n);
    }
    fx.size = (end + 31) / 32 * 32;
    if (with_data) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
        fx.bytes.assign(fx.size, 0);
        memcpy(fx.bytes.data(), meta.data(), meta.size());
        for (const auto & kv : fx.tensors) {
            float * f = (float *) (fx.bytes.data() + kv.second.first);
            for (uint64_t k = 0; k < kv.second.second / 4; ++k) {
                f[k] = uni(rng);
            }
        }
    } else {
        fx.bytes = meta;
    }
    ggml_free(ctx);
    gguf_free(g);
    return fx;
}

// the landing file plus its chunk map, written the way vm/lib/hfetch.py does; land() is the fetcher's verified write
struct landing {
    std::string path;
    std::string map_path;
    int fd = -1;
    uint8_t * map = nullptr;
    size_t map_size = 0;
    uint64_t chunk = 0;
    uint64_t n = 0;
    const fixture * fx = nullptr;

    landing(const fixture & f, const std::string & data_path, const std::string & dir, uint64_t chunk_bytes, uint64_t pid) {
        fx = &f;
        path = data_path;
        chunk = chunk_bytes;
        n = (f.size + chunk - 1) / chunk;
        unlink(path.c_str());
        fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0 || ftruncate(fd, (off_t) f.size) != 0) {
            fprintf(stderr, "cannot create %s\n", path.c_str());
            exit(2);
        }
        struct stat st{};
        fstat(fd, &st);
        map_path = dir + "/" + std::to_string((unsigned long long) st.st_dev) + "-" + std::to_string((unsigned long long) st.st_ino) + ".map";
        const std::string tmp = map_path + ".tmp";
        map_size = 64 + n;
        std::vector<uint8_t> head(map_size, 0);
        memcpy(head.data(), "HUBLWAIT", 8);
        const uint64_t fields[7] = {1, f.size, chunk, n, pid, 0, 0};
        memcpy(head.data() + 8, fields, sizeof(fields));
        const int mfd = open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (mfd < 0 || write(mfd, head.data(), head.size()) != (ssize_t) head.size()) {
            fprintf(stderr, "cannot write %s\n", tmp.c_str());
            exit(2);
        }
        rename(tmp.c_str(), map_path.c_str());
        map = (uint8_t *) mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
        close(mfd);
        g_watch.dev = st.st_dev;
        g_watch.ino = st.st_ino;
        g_watch.bits = map + 64;
        g_watch.chunk = chunk;
        g_watch.size = f.size;
        g_watch.offsets.clear();
        for (const auto & kv : f.tensors) {
            g_watch.offsets[kv.first] = kv.second.first;
        }
        g_watch.reads = 0;
        g_watch.early = 0;
        g_copies = 0;
        g_early_copies = 0;
    }

    ~landing() {
        g_watch.on = false;
        munmap(map, map_size);
        close(fd);
        unlink(path.c_str());
        unlink(map_path.c_str());
    }

    void land(uint64_t c) {
        const uint64_t off = c * chunk;
        const uint64_t len = std::min(chunk, fx->size - off);
        if (off < fx->bytes.size()) {
            const uint64_t have = std::min<uint64_t>(len, fx->bytes.size() - off);
            if (pwrite(fd, fx->bytes.data() + off, have, (off_t) off) != (ssize_t) have) {
                fprintf(stderr, "pwrite failed\n");
                exit(2);
            }
        }
        __atomic_store_n(map + 64 + c, 1, __ATOMIC_RELEASE);
    }

    void land_header() {
        for (uint64_t c = 0; c * chunk < fx->data_offset; ++c) {
            land(c);
        }
    }

    void set_field(int index, uint64_t value) { __atomic_store_n((uint64_t *) (map + 8 + 8 * index), value, __ATOMIC_RELEASE); }
};

static std::vector<uint64_t> body_chunks(const landing & l) {
    std::vector<uint64_t> out;
    for (uint64_t c = 0; c < l.n; ++c) {
        if (c * l.chunk >= l.fx->data_offset) {
            out.push_back(c);
        }
    }
    return out;
}

static llama_model_params params_for(llama_load_mode mode, const llama_model_tensor_buft_override * overrides) {
    llama_model_params p = llama_model_default_params();
    p.load_mode = mode;
    p.tensor_buft_overrides = overrides;
    p.n_gpu_layers = 0;
    return p;
}

// every tensor of the loaded model, host or test buffer, equals the fixture bytes
static std::string compare_tensors(llama_model * model, const fixture & fx) {
    size_t n = 0;
    for (const auto & it : model->tensors_by_name) {
        ggml_tensor * t = it.second;
        auto f = fx.tensors.find(it.first);
        if (f == fx.tensors.end()) {
            return "tensor " + it.first + " is not in the fixture";
        }
        std::vector<uint8_t> got(ggml_nbytes(t));
        ggml_backend_tensor_get(t, got.data(), 0, got.size());
        if (got.size() != f->second.second || memcmp(got.data(), fx.bytes.data() + f->second.first, got.size()) != 0) {
            return "tensor " + it.first + " differs from the source";
        }
        n++;
    }
    return n == fx.tensors.size() ? "" : "only " + std::to_string(n) + " of " + std::to_string(fx.tensors.size()) + " tensors loaded";
}

static const dims SMALL = {256, 512, 4, 128};
static std::string g_dir;
static std::string g_shm;

// chunks land in random order while the loader runs: bytes equal, no read or copy before the bits
static void case_random(llama_load_mode mode, unsigned seed, uint64_t chunk) {
    const char * label = mode == LLAMA_LOAD_MODE_MMAP ? "mmap" : "-lm none";
    const fixture fx = make_fixture(SMALL, 0, seed, true);
    landing l(fx, g_dir + "/model.gguf", g_dir, chunk, (uint64_t) getpid());
    l.land_header();
    std::vector<uint64_t> order = body_chunks(l);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    static const llama_model_tensor_buft_override ov[] = {{"blk\\.[13]\\.|^output\\.weight", test_buft()}, {nullptr, nullptr}};
    g_watch.on = true;
    std::thread fetcher([&] {
        std::uniform_int_distribution<int> pause(0, 1500);
        std::mt19937 r2(seed + 1);
        for (uint64_t c : order) {
            std::this_thread::sleep_for(std::chrono::microseconds(pause(r2)));
            l.land(c);
        }
    });
    llama_model * model = llama_model_load_from_file(l.path.c_str(), params_for(mode, ov));
    fetcher.join();
    const std::string log = log_take();
    const std::string tag = std::string(label) + " seed " + std::to_string(seed) + " chunk " + std::to_string(chunk >> 10) + " KiB";
    check(model != nullptr, tag + ": the patched loader loads the model while its chunks arrive in random order");
    if (model != nullptr) {
        const std::string diff = compare_tensors(model, fx);
        check(diff.empty(), tag + ": every tensor's bytes equal the source " + diff);
        llama_model_free(model);
    }
    check(g_copies > 0, tag + ": the test buffer type made the loader copy (" + std::to_string(g_copies) + " copies)");
    check(g_early_copies == 0, tag + ": no copy before its chunks were verified (" + std::to_string(g_early_copies) + " early)");
    if (mode != LLAMA_LOAD_MODE_MMAP) {
        check(g_watch.reads > 0, tag + ": the read counter saw the file reads (" + std::to_string(g_watch.reads.load()) + ")");
    }
    check(g_watch.early == 0, tag + ": no file read before its chunks were verified (" + std::to_string(g_watch.early.load()) + " of " +
                                  std::to_string(g_watch.reads.load()) + " reads early)");
    check(log.find("load_wait: model.gguf") != std::string::npos, tag + ": the loader found the chunk map");
}

// a stdio buffer of llama_file must not serve bytes it read ahead of a chunk before that chunk's sha256 passed
static void case_stale_stdio() {
    const fixture fx = make_fixture(SMALL, 0, 7, true);
    const uint64_t chunk = 3 * 4096 + 1000;
    check(fx.data_offset + 512 < chunk && 2 * chunk < fx.size, "stale stdio: the chunk boundary lies inside the tensor data");
    landing l(fx, g_dir + "/model.gguf", g_dir, chunk, (uint64_t) getpid());
    for (uint64_t c = 0; c < l.n; ++c) {
        if (c != 1) {
            l.land(c);
        }
    }
    auto differ = [&](const std::vector<uint8_t> & got, uint64_t off) {
        size_t n = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            n += got[i] != fx.bytes[off + i];
        }
        return n;
    };
    llama_file file(l.path.c_str(), "rb");
    std::vector<uint8_t> before(256);
    file.seek(chunk - 512, SEEK_SET);
    file.read_raw(before.data(), before.size());
    check(differ(before, chunk - 512) == 0, "stale stdio: a read that ends 256 bytes before the held chunk returns the source");
    l.land(1);
    std::vector<uint8_t> across(1024);
    file.seek(chunk - 256, SEEK_SET);
    file.read_raw(across.data(), across.size());
    const size_t stale = differ(across, chunk - 256);
    check(stale == 0, "stale stdio: a read across the boundary after the chunk landed returns the source (" + std::to_string(stale) + " of 1024 bytes stale)");
}

// a chunk that never arrives: GUARD: loadwait after the progress timeout, or at once when the fetcher ended
static void case_guard(const char * how, llama_load_mode mode) {
    const fixture fx = make_fixture(SMALL, 0, 11, true);
    const bool dead = strcmp(how, "progress") != 0;
    uint64_t pid = (uint64_t) getpid();
    if (strcmp(how, "dead pid") == 0) {
        const pid_t child = fork();
        if (child == 0) {
            _exit(0);
        }
        waitpid(child, nullptr, 0);
        pid = (uint64_t) child;
    }
    landing l(fx, g_dir + "/model.gguf", g_dir, 64 << 10, pid);
    if (strcmp(how, "state ended") == 0) {
        l.set_field(5, 2);
    }
    const auto miss = fx.tensors.at("blk.3.ffn_up.weight");
    const uint64_t missing = (miss.first + miss.second / 2) / l.chunk;
    for (uint64_t c = 0; c < l.n; ++c) {
        if (c != missing) {
            l.land(c);
        }
    }
    static const llama_model_tensor_buft_override ov[] = {{"blk\\.[13]\\.|^output\\.weight", test_buft()}, {nullptr, nullptr}};
    setenv("LLAMA_LOAD_WAIT_TIMEOUT_S", dead ? "30" : "2", 1);
    const double t0 = now_s();
    llama_model * model = llama_model_load_from_file(l.path.c_str(), params_for(mode, ov));
    const double took = now_s() - t0;
    setenv("LLAMA_LOAD_WAIT_TIMEOUT_S", "20", 1);
    const std::string log = log_take();
    const std::string tag = std::string(mode == LLAMA_LOAD_MODE_MMAP ? "mmap" : "-lm none") + ", missing chunk, " + how;
    check(model == nullptr, tag + ": the load fails");
    if (model != nullptr) {
        llama_model_free(model);
    }
    check(log.find("GUARD: loadwait: model.gguf chunk " + std::to_string(missing)) != std::string::npos, tag + ": the log names GUARD: loadwait and the chunk");
    if (dead) {
        check(log.find("the fetcher ended") != std::string::npos && took < 1.5, tag + ": fails at once (" + std::to_string(took) + " s)");
    } else {
        check(log.find("no chunk verified for 2 s") != std::string::npos && took >= 2.0 && took < 5.0, tag + ": fails after the 2 s progress timeout (" + std::to_string(took) + " s)");
    }
}

static long shmem_kib() {
    FILE * f = fopen("/proc/meminfo", "r");
    char line[256];
    long kib = -1;
    while (f != nullptr && fgets(line, sizeof line, f) != nullptr) {
        if (strncmp(line, "Shmem:", 6) == 0) {
            kib = atol(line + 6);
        }
    }
    if (f != nullptr) {
        fclose(f);
    }
    return kib;
}

static long g_shmem_at_load = -1;

static bool first_progress(float, void *) {
    g_shmem_at_load = shmem_kib();
    return false;
}

// review B2: a sparse file under /dev/shm is mapped without populating its holes
static void case_shmem() {
    const dims big = {512, 512, 1, 65536};
    const fixture fx = make_fixture(big, 0, 3, false);
    const uint64_t chunk = 4 << 20;
    for (int on = 1; on >= 0; --on) {
        if (on) {
            setenv("LLAMA_LOAD_WAIT", g_dir.c_str(), 1);
        } else {
            unsetenv("LLAMA_LOAD_WAIT");
        }
        landing l(fx, g_shm + "/sparse.gguf", g_dir, chunk, (uint64_t) getpid());
        l.land_header();
        llama_model_params p = params_for(LLAMA_LOAD_MODE_MMAP, nullptr);
        p.progress_callback = first_progress;
        g_shmem_at_load = -1;
        const long before = shmem_kib();
        llama_model * model = llama_model_load_from_file(l.path.c_str(), p);
        const long grew = g_shmem_at_load - before;
        if (model != nullptr) {
            llama_model_free(model);
        }
        log_take();
        const long body_kib = (long) ((fx.size - fx.data_offset) >> 10);
        if (on) {
            check(g_shmem_at_load >= 0 && grew < (long) (chunk >> 10),
                  "sparse /dev/shm file, load_wait on: Shmem grew " + std::to_string(grew) + " KiB at mapping, less than one chunk (" + std::to_string(chunk >> 10) + " KiB)");
        } else {
            check(g_shmem_at_load >= 0 && grew > body_kib * 9 / 10,
                  "sparse /dev/shm file, load_wait off (control): MAP_POPULATE allocated the holes, Shmem grew " + std::to_string(grew) + " of " + std::to_string(body_kib) + " KiB");
        }
    }
    setenv("LLAMA_LOAD_WAIT", g_dir.c_str(), 1);
}

#ifdef LOADWAIT_MOE
// b11963: the MoE cache prefill copies host experts, so context creation waits for every shard chunk first
static void case_moe_prefill() {
    const fixture fx = make_fixture(SMALL, 0, 5, true);
    landing l(fx, g_dir + "/model.gguf", g_dir, 64 << 10, (uint64_t) getpid());
    const auto last_copied = fx.tensors.at("blk.2.ffn_up.weight");
    const uint64_t held_from = (last_copied.first + last_copied.second + l.chunk - 1) / l.chunk;
    for (uint64_t c = 0; c < held_from; ++c) {
        l.land(c);
    }
    static const llama_model_tensor_buft_override ov[] = {{"blk\\.[012]\\.|^output\\.weight", test_buft()}, {nullptr, nullptr}};
    g_watch.on = true;
    llama_model * model = llama_model_load_from_file(l.path.c_str(), params_for(LLAMA_LOAD_MODE_MMAP, ov));
    check(model != nullptr, "MoE prefill: the model loads while the chunks of the host-mapped blk.3 tensors are missing");
    check(held_from < l.n, "MoE prefill: " + std::to_string(l.n - held_from) + " chunks are held back");
    if (model == nullptr) {
        return;
    }
    const std::string profile = g_dir + "/profile.txt";
    FILE * pf = fopen(profile.c_str(), "w");
    fputs("0 0\n", pf);
    fclose(pf);
    setenv("LLAMA_MOE_CACHE_PROFILE", profile.c_str(), 1);
    double last_bit = 0.0;
    std::thread fetcher([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        for (uint64_t c = held_from; c < l.n; ++c) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            l.land(c);
        }
        last_bit = now_s();
    });
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.moe_cache_size = 1 << 20;
    g_prefill_line_at = 0.0;
    const double t0 = now_s();
    llama_context * ctx = llama_init_from_model(model, cp);
    const double returned = now_s();
    fetcher.join();
    const std::string log = log_take();
    unsetenv("LLAMA_MOE_CACHE_PROFILE");
    check(g_prefill_line_at > 0.0 && g_prefill_line_at >= last_bit, "MoE prefill: the wait before the prefill ends after the last shard bit (" +
          std::to_string(g_prefill_line_at - t0) + " s against " + std::to_string(last_bit - t0) + " s)");
    const size_t wait_at = log.find("MoE cache prefill after every chunk");
    const size_t cache_at = log.find("MoE cache requires a GPU backend");
    check(wait_at != std::string::npos && cache_at != std::string::npos && wait_at < cache_at,
          "MoE prefill: the cache (and its first upload) comes only after the wait; on the CPU build it then stops at 'requires a GPU backend'");
    check(ctx == nullptr && returned >= last_bit, "MoE prefill: context creation returned after the last bit");
    if (ctx != nullptr) {
        llama_free(ctx);
    }
    const std::string diff = compare_tensors(model, fx);
    check(diff.empty(), "MoE prefill: every tensor equals the source after the wait " + diff);
    llama_model_free(model);
}
#endif

int main(int argc, char ** argv) {
    const char * only = argc > 1 ? argv[1] : nullptr;
    char dir_tmpl[] = "/tmp/loadwait-test.XXXXXX";
    char shm_tmpl[] = "/dev/shm/loadwait-test.XXXXXX";
    g_dir = mkdtemp(dir_tmpl);
    g_shm = mkdtemp(shm_tmpl);
    setenv("LLAMA_LOAD_WAIT", g_dir.c_str(), 1);
    setenv("LLAMA_LOAD_WAIT_TIMEOUT_S", "20", 1);
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    const unsigned base = getenv("LOADWAIT_SEED") ? (unsigned) atoi(getenv("LOADWAIT_SEED")) : (unsigned) time(nullptr);
    printf("seed base %u\n", base);
    auto want = [&](const char * name) { return only == nullptr || strcmp(only, name) == 0; };
    if (want("random")) {
        for (unsigned k = 0; k < 3; ++k) {
            case_random(LLAMA_LOAD_MODE_MMAP, base + k, k == 1 ? 96 << 10 : 64 << 10);
            case_random(LLAMA_LOAD_MODE_NONE, base + 10 + k, k == 1 ? 96 << 10 : 64 << 10);
        }
    }
    if (want("stale")) {
        case_stale_stdio();
    }
    if (want("guard")) {
        case_guard("progress", LLAMA_LOAD_MODE_MMAP);
        case_guard("progress", LLAMA_LOAD_MODE_NONE);
        case_guard("dead pid", LLAMA_LOAD_MODE_MMAP);
        case_guard("state ended", LLAMA_LOAD_MODE_NONE);
    }
    if (want("shmem")) {
        case_shmem();
    }
#ifdef LOADWAIT_MOE
    if (want("moe")) {
        case_moe_prefill();
    }
#endif
    llama_backend_free();
    rmdir(g_dir.c_str());
    rmdir(g_shm.c_str());
    printf("%s loadwait_test: %d of %d checks failed\n", g_fails ? "FAIL" : "PASS", g_fails, g_checks);
    return g_fails ? 1 : 0;
}
