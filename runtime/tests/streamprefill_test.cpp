// Unit test of patch_streamprefill.py (M15-T13) on a CPU-only build of the patched DeepSeek tree with a fake GPU device; runtime/tests/test_loadwait.sh builds and runs it.
#define main loadwait_main
#include "loadwait_test.cpp"
#undef main

#include "llama-moe-cache.h"

#include <climits>
#include <fstream>
#include <set>
#include <sstream>
#include <tuple>

// a GPU that only stores bytes: the loader copies the dense tensors into it, the MoE cache allocates its banks in it
struct bank_write {
    std::string bank;
    size_t dst;
    size_t size;
    int64_t src_off;
    bool ready;
};
static std::mutex g_wr_mtx;
static std::vector<bank_write> g_bank_writes;
static const uint8_t * g_map_base = nullptr;

static void fb_set(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    if (strncmp(ggml_get_name(t), "moe_cache.", 10) == 0) {
        const uint8_t * src = (const uint8_t *) data;
        bank_write w{ggml_get_name(t), offset, size, -1, false};
        if (g_map_base != nullptr && src >= g_map_base && src + size <= g_map_base + g_watch.size) {
            w.src_off = src - g_map_base;
            w.ready = range_ready((uint64_t) w.src_off, size);
        }
        std::lock_guard<std::mutex> lock(g_wr_mtx);
        g_bank_writes.push_back(w);
    }
    memcpy((uint8_t *) t->data + offset, data, size);
}

static ggml_backend_buffer_t fbt_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * tb = new test_buffer;
    tb->mem = (uint8_t *) aligned_alloc(128, (size + 127) / 128 * 128 + 128);
    ggml_backend_buffer_i iface{};
    iface.free_buffer = tb_free;
    iface.get_base = tb_base;
    iface.set_tensor = fb_set;
    iface.get_tensor = tb_get;
    iface.memset_tensor = tb_memset;
    iface.clear = tb_clear;
    return ggml_backend_buffer_init(buft, iface, tb, size);
}

static const char * fbt_name(ggml_backend_buffer_type_t) { return "FAKEGPU"; }
static size_t fbt_align(ggml_backend_buffer_type_t) { return 128; }
static const char * fd_name(ggml_backend_dev_t) { return "FAKEGPU"; }
static const char * fd_desc(ggml_backend_dev_t) { return "streamprefill test GPU"; }
static void fd_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = (size_t) 1 << 30; }
static enum ggml_backend_dev_type fd_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static ggml_backend_t fd_init(ggml_backend_dev_t, const char *) { return nullptr; }
static bool fd_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool fd_offload_op(ggml_backend_dev_t, const ggml_tensor *) { return false; }
static const char * fr_name(ggml_backend_reg_t) { return "FAKEGPU"; }
static size_t fr_count(ggml_backend_reg_t) { return 1; }
static const char * fbk_name(ggml_backend_t) { return "FAKEGPU"; }

static ggml_backend_buffer_type g_fake_buft;
static ggml_backend_device g_fake_dev;
static ggml_backend_reg g_fake_reg;
static ggml_backend g_fake_backend;
static ggml_guid g_fake_guid = {0x7a};

static ggml_backend_buffer_type_t fd_buft(ggml_backend_dev_t) { return &g_fake_buft; }
static bool fd_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) { return b == &g_fake_buft; }
static ggml_backend_dev_t fr_get(ggml_backend_reg_t, size_t) { return &g_fake_dev; }
static void fd_props(ggml_backend_dev_t dev, ggml_backend_dev_props * p) {
    p->name = fd_name(dev);
    p->description = fd_desc(dev);
    fd_memory(dev, &p->memory_free, &p->memory_total);
    p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    p->device_id = nullptr;
    p->caps = {false, false, false, false, true};
}

static void fake_gpu_init() {
    g_fake_buft.iface.get_name = fbt_name;
    g_fake_buft.iface.alloc_buffer = fbt_alloc;
    g_fake_buft.iface.get_alignment = fbt_align;
    g_fake_buft.iface.is_host = tbt_is_host;
    g_fake_buft.device = &g_fake_dev;
    g_fake_dev.iface.get_name = fd_name;
    g_fake_dev.iface.get_description = fd_desc;
    g_fake_dev.iface.get_memory = fd_memory;
    g_fake_dev.iface.get_type = fd_type;
    g_fake_dev.iface.get_props = fd_props;
    g_fake_dev.iface.init_backend = fd_init;
    g_fake_dev.iface.get_buffer_type = fd_buft;
    g_fake_dev.iface.supports_op = fd_supports_op;
    g_fake_dev.iface.supports_buft = fd_supports_buft;
    g_fake_dev.iface.offload_op = fd_offload_op;
    g_fake_dev.reg = &g_fake_reg;
    g_fake_reg.api_version = GGML_BACKEND_API_VERSION;
    g_fake_reg.iface.get_name = fr_name;
    g_fake_reg.iface.get_device_count = fr_count;
    g_fake_reg.iface.get_device = fr_get;
    g_fake_backend.guid = &g_fake_guid;
    g_fake_backend.iface.get_name = fbk_name;
    g_fake_backend.device = &g_fake_dev;
}

// three MoE layers of 384 experts (the count punch.py assumes): layers 0 and 2 F32, layer 1 F16, so the cache makes two groups
static const int N_EMBD = 32, N_FF = 24, N_LAYER = 3, N_VOCAB = 128, N_EXPERT = 384, N_USED = 2;
static const size_t CACHE_BYTES = 3000000;
static const int DYN = 40;
static const char * CPU_RE = "blk\\.[0-9]+\\.ffn_(up|gate|down)_exps\\.weight";
static const char * EXPS[] = {"gate", "up", "down"};

static bool is_expert(const std::string & name) { return name.find("_exps.weight") != std::string::npos; }

static fixture make_moe_fixture(unsigned seed) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_str(g, "general.name", "streamprefill-fixture");
    gguf_set_val_u32(g, "llama.context_length", 256);
    gguf_set_val_u32(g, "llama.embedding_length", N_EMBD);
    gguf_set_val_u32(g, "llama.block_count", N_LAYER);
    gguf_set_val_u32(g, "llama.feed_forward_length", N_FF);
    gguf_set_val_u32(g, "llama.attention.head_count", 4);
    gguf_set_val_u32(g, "llama.attention.head_count_kv", 4);
    gguf_set_val_u32(g, "llama.expert_count", N_EXPERT);
    gguf_set_val_u32(g, "llama.expert_used_count", N_USED);
    gguf_set_val_f32(g, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_str(g, "tokenizer.ggml.model", "llama");
    std::vector<std::string> toks;
    std::vector<float> scores;
    std::vector<int32_t> types;
    for (int i = 0; i < N_VOCAB; ++i) {
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

    struct spec {
        std::string name;
        std::vector<int64_t> ne;
        ggml_type type;
    };
    std::vector<spec> specs = {{"token_embd.weight", {N_EMBD, N_VOCAB}, GGML_TYPE_F32}, {"output_norm.weight", {N_EMBD}, GGML_TYPE_F32},
                               {"output.weight", {N_EMBD, N_VOCAB}, GGML_TYPE_F32}};
    for (int l = 0; l < N_LAYER; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const ggml_type et = l == 1 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        specs.push_back({b + "attn_norm.weight", {N_EMBD}, GGML_TYPE_F32});
        specs.push_back({b + "attn_q.weight", {N_EMBD, N_EMBD}, GGML_TYPE_F32});
        specs.push_back({b + "attn_k.weight", {N_EMBD, N_EMBD}, GGML_TYPE_F32});
        specs.push_back({b + "attn_v.weight", {N_EMBD, N_EMBD}, GGML_TYPE_F32});
        specs.push_back({b + "attn_output.weight", {N_EMBD, N_EMBD}, GGML_TYPE_F32});
        specs.push_back({b + "ffn_norm.weight", {N_EMBD}, GGML_TYPE_F32});
        specs.push_back({b + "ffn_gate_inp.weight", {N_EMBD, N_EXPERT}, GGML_TYPE_F32});
        specs.push_back({b + "ffn_gate_exps.weight", {N_EMBD, N_FF, N_EXPERT}, et});
        specs.push_back({b + "ffn_down_exps.weight", {N_FF, N_EMBD, N_EXPERT}, et});
        specs.push_back({b + "ffn_up_exps.weight", {N_EMBD, N_FF, N_EXPERT}, et});
    }
    ggml_init_params ip = {ggml_tensor_overhead() * (specs.size() + 4), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    for (const auto & s : specs) {
        ggml_tensor * t = ggml_new_tensor(ctx, s.type, (int) s.ne.size(), s.ne.data());
        ggml_set_name(t, s.name.c_str());
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
    std::mt19937 rng(seed);
    fx.bytes.assign(fx.size, 0);
    memcpy(fx.bytes.data(), meta.data(), meta.size());
    for (uint64_t i = fx.data_offset; i < fx.size; ++i) {
        fx.bytes[i] = (uint8_t) (rng() >> 24);
    }
    ggml_free(ctx);
    gguf_free(g);
    return fx;
}

// a profile in random order over experts 0 to 199 of each layer, with repeats and bad ids, cut by a word as fscanf stops there
static std::string make_profile(unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::pair<int, int>> pairs;
    for (int l = 0; l < N_LAYER; ++l) {
        for (int e = 0; e < 200; ++e) {
            pairs.push_back({l, e});
        }
    }
    std::shuffle(pairs.begin(), pairs.end(), rng);
    std::ostringstream out;
    for (size_t i = 0; i < pairs.size(); ++i) {
        out << pairs[i].first << " " << pairs[i].second << "\n";
        if (i % 37 == 5) {
            out << pairs[i / 2].first << " " << pairs[i / 2].second << "\n";
        }
        if (i == 100) {
            out << "7 3\n0 999\n-1 2\n";
        }
    }
    out << "end\n0 300\n";
    return out.str();
}

static std::string slurp(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool exists(const std::string & path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}

struct prefill_run {
    bool ok = false;
    std::string error;
    std::string resident;
    std::vector<std::vector<int32_t>> slots;
    std::vector<std::vector<uint8_t>> banks;
    std::vector<int64_t> n_slots;
    std::vector<bank_write> writes;
    std::vector<std::string> uploaded;
    std::string log;
    double returned = 0.0;
    double last_bit = 0.0;
    size_t held = 0;
    size_t slot_errors = 0;
};

enum arrival { ARRIVE_ALL_FIRST, ARRIVE_DENSE_FIRST, ARRIVE_INTERLEAVED, ARRIVE_REVERSE };

static const char * arrival_name(arrival a) {
    return a == ARRIVE_ALL_FIRST ? "every chunk before the load" : a == ARRIVE_DENSE_FIRST ? "dense chunks first, experts in random order"
         : a == ARRIVE_INTERLEAVED ? "dense and expert chunks interleaved at random" : "experts in reverse file order";
}

static std::set<std::pair<int, int>> parse_resident(const std::string & text) {
    std::set<std::pair<int, int>> out;
    std::istringstream in(text);
    int il = 0, e = 0;
    while (in >> il >> e) {
        out.insert({il, e});
    }
    return out;
}

static std::string expert_name(int il, const char * kind) { return "blk." + std::to_string(il) + ".ffn_" + kind + "_exps.weight"; }

// chunk classes for the arrival schedule: 0 touches a dense tensor, 1 a resident expert, 2 only cold experts
static std::vector<int> chunk_classes(const fixture & fx, const landing & l, const std::set<std::pair<int, int>> & res) {
    std::vector<int> cls(l.n, 2);
    auto mark = [&](uint64_t off, uint64_t len, int c) {
        for (uint64_t k = off / l.chunk; k <= (off + len - 1) / l.chunk; ++k) {
            cls[k] = std::min(cls[k], c);
        }
    };
    for (const auto & kv : fx.tensors) {
        if (!is_expert(kv.first)) {
            mark(kv.second.first, kv.second.second, 0);
        }
    }
    for (const auto & r : res) {
        for (const char * kind : EXPS) {
            const auto t = fx.tensors.at(expert_name(r.first, kind));
            const uint64_t es = t.second / N_EXPERT;
            mark(t.first + r.second * es, es, 1);
        }
    }
    return cls;
}

static prefill_run run_prefill(const fixture & fx, arrival how, unsigned seed, uint64_t chunk,
                               const std::set<std::pair<int, int>> & res_ref, const std::string & upload_log) {
    prefill_run run;
    landing l(fx, g_dir + "/moe.gguf", g_dir, chunk, (uint64_t) getpid());
    l.land_header();
    std::vector<uint64_t> order[3];
    const std::vector<int> cls = chunk_classes(fx, l, res_ref);
    for (uint64_t c : body_chunks(l)) {
        order[cls[c]].push_back(c);
    }
    std::mt19937 rng(seed);
    for (auto & o : order) {
        std::shuffle(o.begin(), o.end(), rng);
    }
    if (how == ARRIVE_INTERLEAVED) {
        order[0].insert(order[0].end(), order[1].begin(), order[1].end());
        order[1].clear();
        std::shuffle(order[0].begin(), order[0].end(), rng);
    } else if (how == ARRIVE_REVERSE) {
        std::sort(order[1].rbegin(), order[1].rend());
    }
    run.held = order[1].size() + order[2].size();
    if (how == ARRIVE_ALL_FIRST) {
        for (const auto & o : order) {
            for (uint64_t c : o) {
                l.land(c);
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_wr_mtx);
        g_bank_writes.clear();
    }
    g_map_base = nullptr;
    g_watch.on = true;
    std::thread fetcher;
    if (how != ARRIVE_ALL_FIRST) {
        fetcher = std::thread([&, seed] {
            std::mt19937 r2(seed + 1);
            std::uniform_int_distribution<int> pause(0, 2000);
            const size_t total = order[0].size() + order[1].size() + order[2].size();
            size_t n = 0;
            for (int k = 0; k < 3; ++k) {
                if (k == 2) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
                for (uint64_t c : order[k]) {
                    std::this_thread::sleep_for(std::chrono::microseconds(pause(r2)));
                    if (++n == total) {
                        run.last_bit = now_s();
                    }
                    l.land(c);
                }
            }
        });
    }
    static const llama_model_tensor_buft_override ov[] = {{CPU_RE, ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    static ggml_backend_dev_t devs[] = {&g_fake_dev, nullptr};
    llama_model_params p = params_for(LLAMA_LOAD_MODE_MMAP, ov);
    p.devices = devs;
    p.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(l.path.c_str(), p);
    const std::string res_path = g_dir + "/resident.txt";
    unlink(res_path.c_str());
    if (model == nullptr) {
        run.error = "the model did not load";
    } else {
        const auto ref = fx.tensors.at("blk.0.ffn_up_exps.weight");
        for (const auto & it : model->tensors_by_name) {
            if (it.first == "blk.0.ffn_up_exps.weight") {
                g_map_base = (const uint8_t *) it.second->data - ref.first;
            }
        }
        try {
            std::vector<ggml_backend_t> backends = {&g_fake_backend};
            std::vector<ggml_backend_buffer_type_t> bufts = {&g_fake_buft};
            llama_moe_cache cache(*model, false, backends, bufts, CACHE_BYTES, 0.0f);
            cache.prefill((g_dir + "/profile.txt").c_str(), res_path.c_str());
            run.returned = now_s();
            for (int il = 0; il < N_LAYER; ++il) {
                const ggml_tensor * sm = cache.get_slot_map(il, 1, 1);
                run.slots.emplace_back(sm ? (const int32_t *) sm->data : nullptr, sm ? (const int32_t *) sm->data + N_EXPERT : nullptr);
                for (const char * kind : EXPS) {
                    const ggml_tensor * w = nullptr;
                    for (const auto & it : model->tensors_by_name) {
                        if (it.first == expert_name(il, kind)) {
                            w = it.second;
                        }
                    }
                    const ggml_tensor * view = w ? cache.get_experts(w) : nullptr;
                    std::vector<uint8_t> bytes(view ? ggml_nbytes(view) : 0);
                    if (view) {
                        ggml_backend_tensor_get(view, bytes.data(), 0, bytes.size());
                    }
                    run.n_slots.push_back(view ? view->ne[2] : -1);
                    // each slot of a cached expert holds that expert's source bytes
                    if (view && sm) {
                        const auto t = fx.tensors.at(expert_name(il, kind));
                        const size_t es = t.second / N_EXPERT;
                        for (int e = 0; e < N_EXPERT; ++e) {
                            const int32_t s = ((const int32_t *) sm->data)[e];
                            if (s < view->ne[2] && memcmp(bytes.data() + (size_t) s * es, fx.bytes.data() + t.first + (size_t) e * es, es) != 0) {
                                run.slot_errors++;
                            }
                        }
                    }
                    run.banks.push_back(std::move(bytes));
                }
            }
            run.ok = true;
        } catch (const std::exception & ex) {
            run.returned = now_s();
            run.error = ex.what();
        }
    }
    if (fetcher.joinable()) {
        fetcher.join();
    }
    g_watch.on = false;
    run.resident = slurp(res_path);
    {
        std::lock_guard<std::mutex> lock(g_wr_mtx);
        run.writes = g_bank_writes;
    }
    std::istringstream up(exists(upload_log) ? slurp(upload_log) : "");
    for (std::string line; std::getline(up, line);) {
        run.uploaded.push_back(line);
    }
    run.log = log_take();
    if (model != nullptr) {
        llama_model_free(model);
    }
    return run;
}

// the uploaded lines the patched server must write: each dense device tensor once, each pinned expert's range once per expert tensor
static std::vector<std::string> expected_uploads(const fixture & fx, const std::string & path, const std::set<std::pair<int, int>> & res) {
    std::vector<std::string> out;
    for (const auto & kv : fx.tensors) {
        if (!is_expert(kv.first) && kv.first != "token_embd.weight") {
            out.push_back("uploaded " + path + " " + std::to_string(kv.second.first) + " " + std::to_string(kv.second.second));
        }
    }
    for (const auto & r : res) {
        for (const char * kind : EXPS) {
            const auto t = fx.tensors.at(expert_name(r.first, kind));
            const uint64_t es = t.second / N_EXPERT;
            out.push_back("uploaded " + path + " " + std::to_string(t.first + r.second * es) + " " + std::to_string(es));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static void case_stream(const fixture & fx, unsigned base) {
    setenv("LLAMA_MOE_CACHE_PIN_PREFILL", "1", 1);
    setenv("LLAMA_MOE_CACHE_DYN", std::to_string(DYN).c_str(), 1);
    unsetenv("LLAMA_MOE_STREAM_PREFILL");
    unsetenv("LLAMA_UPLOAD_LOG");
    const std::string log1 = g_dir + "/uploaded-1.txt";
    const prefill_run ref = run_prefill(fx, ARRIVE_ALL_FIRST, base, 64 << 10, {}, log1);
    const std::set<std::pair<int, int>> res = parse_resident(ref.resident);
    check(ref.ok, "key off: today's prefill runs on the fake GPU (" + ref.error + ")");
    check(res.size() > 100 && ref.writes.size() == res.size() * 3, "key off: " + std::to_string(res.size()) + " experts pinned, " +
          std::to_string(ref.writes.size()) + " bank writes (three expert tensors each)");
    check(ref.slot_errors == 0, "key off: every slot of a pinned expert holds its source bytes (" + std::to_string(ref.slot_errors) + " differ)");
    check(!exists(log1) && ref.uploaded.empty(), "key off: no LLAMA_UPLOAD_LOG, no uploaded file");
    check(ref.log.find("MoE cache streaming prefill") == std::string::npos, "key off: no streaming prefill line in the log");
    char real[PATH_MAX];

    const arrival modes[] = {ARRIVE_DENSE_FIRST, ARRIVE_INTERLEAVED, ARRIVE_REVERSE};
    for (int k = 0; k < 3; ++k) {
        const unsigned seed = base + 100 + k;
        const uint64_t chunk = k == 1 ? 96 << 10 : 64 << 10;
        const std::string upload_log = g_dir + "/uploaded-" + std::to_string(k + 2) + ".txt";
        unlink(upload_log.c_str());
        setenv("LLAMA_MOE_STREAM_PREFILL", "1", 1);
        setenv("LLAMA_UPLOAD_LOG", upload_log.c_str(), 1);
        const prefill_run run = run_prefill(fx, modes[k], seed, chunk, res, upload_log);
        unsetenv("LLAMA_MOE_STREAM_PREFILL");
        unsetenv("LLAMA_UPLOAD_LOG");
        const std::string tag = "key on, " + std::string(arrival_name(modes[k])) + ", seed " + std::to_string(seed) + ", chunk " + std::to_string(chunk >> 10) + " KiB";
        check(run.ok, tag + ": the streaming prefill completes (" + run.error + ")");
        check(run.held > 0, tag + ": " + std::to_string(run.held) + " expert chunks were not yet verified when the load began");
        check(run.resident == ref.resident && !run.resident.empty(), tag + ": resident.txt is byte-equal to the key-off prefill's");
        check(run.slots == ref.slots, tag + ": every layer's slot map equals the key-off prefill's");
        check(run.n_slots == ref.n_slots && run.banks == ref.banks, tag + ": every bank's bytes equal the key-off prefill's");
        check(run.slot_errors == 0, tag + ": every slot of a pinned expert holds its source bytes (" + std::to_string(run.slot_errors) + " differ)");
        size_t early = 0, unknown = 0;
        for (const auto & w : run.writes) {
            early += w.src_off >= 0 && !w.ready;
            unknown += w.src_off < 0;
        }
        check(run.writes.size() == ref.writes.size() && early == 0 && unknown == 0,
              tag + ": every bank write came after its source chunks were verified (" + std::to_string(early) + " early of " + std::to_string(run.writes.size()) + ")");
        check(run.last_bit > 0.0 && run.returned >= run.last_bit, tag + ": the prefill returned after the last shard bit (" +
              std::to_string(run.returned - run.last_bit) + " s after it)");
        check(run.log.find("load_wait: MoE cache streaming prefill after every chunk") != std::string::npos, tag + ": the log names the final wait");
        std::vector<std::string> got = run.uploaded;
        std::sort(got.begin(), got.end());
        const std::string real_path = realpath(g_dir.c_str(), real) ? std::string(real) + "/moe.gguf" : g_dir + "/moe.gguf";
        const std::vector<std::string> want = expected_uploads(fx, real_path, res);
        check(got == want, tag + ": the uploaded lines are exactly the dense device tensors and the pinned experts' ranges (" + std::to_string(got.size()) +
              " lines, " + std::to_string(want.size()) + " expected)");
        check(run.log.find("GUARD") == std::string::npos, tag + ": no GUARD line");
        if (k == 0 && getenv("STREAMPREFILL_KEEP") != nullptr) {
            const std::string keep = getenv("STREAMPREFILL_KEEP");
            FILE * f = fopen((keep + "/uploaded.txt").c_str(), "w");
            for (const auto & line : run.uploaded) {
                fprintf(f, "%s\n", line.c_str());
            }
            fclose(f);
            f = fopen((keep + "/resident.txt").c_str(), "w");
            fputs(ref.resident.c_str(), f);
            fclose(f);
        }
    }
    unsetenv("LLAMA_MOE_CACHE_PIN_PREFILL");
    unsetenv("LLAMA_MOE_CACHE_DYN");
}

// two mapped files resolve by address to their own real path and offset; a released mapping resolves to nothing
static void case_registry() {
    const fixture a = make_fixture(SMALL, 0, 21, true);
    const fixture b = make_fixture(SMALL, 64, 22, true);
    const std::string pa = g_dir + "/reg-a.gguf", pb = g_dir + "/reg-b.gguf", log = g_dir + "/uploaded-reg.txt";
    for (const auto & fp : {std::make_pair(&a, pa), std::make_pair(&b, pb)}) {
        FILE * f = fopen(fp.second.c_str(), "wb");
        fwrite(fp.first->bytes.data(), 1, fp.first->bytes.size(), f);
        fclose(f);
    }
    unsetenv("LLAMA_LOAD_WAIT");
    setenv("LLAMA_UPLOAD_LOG", log.c_str(), 1);
    unlink(log.c_str());
    char real[PATH_MAX];
    const std::string ra = realpath(pa.c_str(), real) ? real : "", rb = realpath(pb.c_str(), real) ? real : "";
    {
        llama_file fa(pa.c_str(), "rb");
        llama_mmap ma(&fa, 0);
        const void * b_addr = nullptr;
        {
            llama_file fb(pb.c_str(), "rb");
            llama_mmap mb(&fb, 0);
            b_addr = (const uint8_t *) mb.addr() + 4096;
            const llama_load_range r1 = llama_load_range_of((const uint8_t *) ma.addr() + 1000, 64);
            const llama_load_range r2 = llama_load_range_of(b_addr, 128);
            check(r1.region && r1.offset == 1000 && r1.len == 64 && llama_load_range_ready(r1), "registry: an address in the first mapping is offset 1000 of its file");
            check(r2.region && r2.offset == 4096 && r2.len == 128, "registry: an address in the second mapping is offset 4096 of its file");
            llama_upload_log(r1);
            llama_upload_log(r2);
        }
        const llama_load_range gone = llama_load_range_of(b_addr, 128);
        check(!gone.region, "registry: a released mapping resolves to nothing");
    }
    const std::string text = slurp(log);
    check(text == "uploaded " + ra + " 1000 64\nuploaded " + rb + " 4096 128\n", "registry: LLAMA_UPLOAD_LOG holds one line per range with the real path [" + text + "]");
    unsetenv("LLAMA_UPLOAD_LOG");
    setenv("LLAMA_LOAD_WAIT", g_dir.c_str(), 1);
    unlink(pa.c_str());
    unlink(pb.c_str());
    unlink(log.c_str());
}

// with the key on the context no longer waits for every shard chunk before the cache; with it off it still does (T12)
static void case_context(const fixture & fx) {
    for (int on = 1; on >= 0; --on) {
        landing l(fx, g_dir + "/moe.gguf", g_dir, 64 << 10, (uint64_t) getpid());
        l.land_header();
        const auto first = fx.tensors.at("blk.0.ffn_gate_exps.weight");
        const uint64_t held_from = first.first / l.chunk + 1;
        for (uint64_t c = 0; c < held_from; ++c) {
            l.land(c);
        }
        llama_model * model = llama_model_load_from_file(l.path.c_str(), params_for(LLAMA_LOAD_MODE_MMAP, nullptr));
        const std::string tag = on ? "context, key on" : "context, key off (control)";
        check(model != nullptr, tag + ": the CPU model loads with its expert chunks held");
        if (model == nullptr) {
            continue;
        }
        setenv("LLAMA_MOE_CACHE_PROFILE", (g_dir + "/profile.txt").c_str(), 1);
        if (on) {
            setenv("LLAMA_MOE_STREAM_PREFILL", "1", 1);
        }
        double last_bit = 0.0;
        std::thread fetcher([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
            for (uint64_t c = held_from; c < l.n; ++c) {
                if (c + 1 == l.n) {
                    last_bit = now_s();
                }
                l.land(c);
            }
        });
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.moe_cache_size = 1 << 20;
        llama_context * ctx = llama_init_from_model(model, cp);
        const double returned = now_s();
        fetcher.join();
        unsetenv("LLAMA_MOE_STREAM_PREFILL");
        unsetenv("LLAMA_MOE_CACHE_PROFILE");
        const std::string log = log_take();
        if (ctx != nullptr) {
            llama_free(ctx);
        }
        const bool waited = log.find("MoE cache prefill after every chunk") != std::string::npos;
        const bool reached = log.find("MoE cache requires a GPU backend") != std::string::npos;
        if (on) {
            check(!waited && reached && returned < last_bit, tag + ": the cache is built before the held chunks land (" + std::to_string(last_bit - returned) + " s before the last bit)");
        } else {
            check(waited && reached && returned >= last_bit, tag + ": the context waits for every shard chunk first");
        }
        llama_model_free(model);
    }
}

int main(int argc, char ** argv) {
    const char * only = argc > 1 ? argv[1] : nullptr;
    char dir_tmpl[] = "/tmp/streamprefill-test.XXXXXX";
    char shm_tmpl[] = "/dev/shm/streamprefill-test.XXXXXX";
    g_dir = mkdtemp(dir_tmpl);
    g_shm = mkdtemp(shm_tmpl);
    setenv("LLAMA_LOAD_WAIT", g_dir.c_str(), 1);
    setenv("LLAMA_LOAD_WAIT_TIMEOUT_S", "20", 1);
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    fake_gpu_init();
    const unsigned base = getenv("STREAMPREFILL_SEED") ? (unsigned) atoi(getenv("STREAMPREFILL_SEED")) : (unsigned) time(nullptr);
    printf("seed base %u\n", base);
    const fixture fx = make_moe_fixture(base);
    const std::string profile = make_profile(base + 7);
    {
        FILE * f = fopen((g_dir + "/profile.txt").c_str(), "w");
        fputs(profile.c_str(), f);
        fclose(f);
    }
    auto want = [&](const char * name) { return only == nullptr || strcmp(only, name) == 0; };
    if (want("stream")) {
        case_stream(fx, base);
    }
    if (want("registry")) {
        case_registry();
    }
    if (want("context")) {
        case_context(fx);
    }
    if (const char * keep = getenv("STREAMPREFILL_KEEP")) {
        FILE * f = fopen((std::string(keep) + "/fixture.gguf").c_str(), "wb");
        fwrite(fx.bytes.data(), 1, fx.data_offset, f);
        fclose(f);
        f = fopen((std::string(keep) + "/profile.txt").c_str(), "w");
        fputs(profile.c_str(), f);
        fclose(f);
        f = fopen((std::string(keep) + "/params.txt").c_str(), "w");
        fprintf(f, "%zu %d %s=CPU\n", CACHE_BYTES, DYN, CPU_RE);
        fclose(f);
    }
    llama_backend_free();
    unlink((g_dir + "/profile.txt").c_str());
    rmdir(g_dir.c_str());
    rmdir(g_shm.c_str());
    printf("%s streamprefill_test: %d of %d checks failed\n", g_fails ? "FAIL" : "PASS", g_fails, g_checks);
    return g_fails ? 1 : 0;
}
