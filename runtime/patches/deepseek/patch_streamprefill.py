"""Streaming MoE prefill (M15-T13): with LLAMA_MOE_STREAM_PREFILL=1 and a profile the prefill fixes the pinned experts and their slots first, then uploads each one once its chunks are verified; LLAMA_UPLOAD_LOG gets one line per device copy of a mapped range for the punch follower. usage: python3 patch_streamprefill.py LLAMA_SRC_DIR"""
import os, sys

src = sys.argv[1]


def edit(rel, old, new):
    p = os.path.join(src, rel)
    s = open(p).read()
    if new in s:
        print('already patched', rel, new.strip()[:50])
        return
    assert s.count(old) == 1, (rel, old)
    open(p, 'w').write(s.replace(old, new))
    print('patched', rel, new.strip()[:50])


H = 'src/llama-mmap.h'
edit(H, '    void wait_range(size_t offset, size_t len) const;\nprivate:\n',
     '    void wait_range(size_t offset, size_t len) const;\n\n'
     '    // STREAMPREFILL_PATCH: records [addr, addr + size) as this file mapped, for llama_load_range_of\n'
     '    void map_region(const void * addr, size_t size) const;\nprivate:\n')
edit(H, 'void llama_load_wait_all(const char * why);\n', '''void llama_load_wait_all(const char * why);

// STREAMPREFILL_PATCH: a range of a mapped model file found by its address; the streaming prefill uploads it once ready, LLAMA_UPLOAD_LOG gets "uploaded <real path> <offset> <size>" per device copy (vm/deepseek/punch.py --follow)
struct llama_mapped_region;
struct llama_load_range {
    std::shared_ptr<const llama_mapped_region> region;
    size_t offset = 0;
    size_t len = 0;
};
struct llama_load_idle {
    double t_progress = -1.0;
    double t_check = -1.0;
    uint64_t seen = 0;
};
bool llama_moe_stream_prefill();
llama_load_range llama_load_range_of(const void * addr, size_t len);
bool llama_load_range_ready(const llama_load_range & r);
void llama_load_range_idle(const llama_load_range & r, llama_load_idle & idle);
void llama_upload_log(const llama_load_range & r);
void llama_upload_log_addr(const void * addr, size_t len);
void llama_unmap_region(const void * addr);
''')

C = 'src/llama-mmap.cpp'
edit(C, '''void llama_file::wait_range(size_t offset, size_t len) const {
    if (lw) {
        llama_lw_wait_range(*lw, offset, len);
    }
}
''', r'''void llama_file::wait_range(size_t offset, size_t len) const {
    if (lw) {
        llama_lw_wait_range(*lw, offset, len);
    }
}

// STREAMPREFILL_PATCH: mapped model files, newest first, since a later mapping can take a hole that unmap_fragment left in an older one
struct llama_mapped_region {
    const uint8_t * addr = nullptr;
    size_t size = 0;
    std::string path;
    std::shared_ptr<llama_load_wait_map> lw;
};

static std::mutex & llama_regions_mtx() {
    static std::mutex m;
    return m;
}

static std::vector<std::shared_ptr<const llama_mapped_region>> & llama_regions() {
    static std::vector<std::shared_ptr<const llama_mapped_region>> v;
    return v;
}

bool llama_moe_stream_prefill() {
    const char * on = getenv("LLAMA_MOE_STREAM_PREFILL");
    return on != nullptr && std::strcmp(on, "1") == 0;
}

static void llama_region_add(const void * addr, size_t size, const std::string & path, const std::shared_ptr<llama_load_wait_map> & lw) {
    auto r = std::make_shared<llama_mapped_region>();
    r->addr = (const uint8_t *) addr;
    r->size = size;
    r->path = path;
    r->lw = lw;
    std::lock_guard<std::mutex> lock(llama_regions_mtx());
    llama_regions().insert(llama_regions().begin(), r);
}

void llama_unmap_region(const void * addr) {
    std::lock_guard<std::mutex> lock(llama_regions_mtx());
    auto & v = llama_regions();
    v.erase(std::remove_if(v.begin(), v.end(), [&](const std::shared_ptr<const llama_mapped_region> & r) { return r->addr == addr; }), v.end());
}

llama_load_range llama_load_range_of(const void * addr, size_t len) {
    llama_load_range r;
    const uint8_t * p = (const uint8_t *) addr;
    std::lock_guard<std::mutex> lock(llama_regions_mtx());
    for (const auto & reg : llama_regions()) {
        if (p >= reg->addr && p + len <= reg->addr + reg->size) {
            r.region = reg;
            r.offset = (size_t) (p - reg->addr);
            r.len = len;
            break;
        }
    }
    return r;
}

#if defined(__linux__)
bool llama_load_range_ready(const llama_load_range & r) {
    if (!r.region || !r.region->lw) {
        return true;
    }
    const llama_load_wait_map & m = *r.region->lw;
    if (m.missing) {
        return false;
    }
    if (r.len == 0 || r.offset >= m.file_size) {
        return true;
    }
    const uint64_t last = std::min<uint64_t>((uint64_t) r.offset + r.len, m.file_size) - 1;
    for (uint64_t c = r.offset / m.chunk; c <= last / m.chunk; ++c) {
        if (!m.has(c)) {
            return false;
        }
    }
    return true;
}

// one idle step of a caller that polls many ranges: 1 ms, and GUARD: loadwait as in llama_lw_wait_chunk when no chunk was verified for the timeout or the fetcher ended
void llama_load_range_idle(const llama_load_range & r, llama_load_idle & idle) {
    if (!r.region || !r.region->lw) {
        return;
    }
    const llama_load_wait_map & m = *r.region->lw;
    if (m.missing) {
        llama_lw_fail(format("GUARD: loadwait: %s has no chunk map in %s", m.name.c_str(), llama_lw_dir()));
    }
    const uint64_t last = std::min<uint64_t>((uint64_t) r.offset + std::max<size_t>(r.len, 1), m.file_size) - 1;
    uint64_t c = r.offset / m.chunk;
    while (c < last / m.chunk && m.has(c)) {
        ++c;
    }
    const char * t = getenv("LLAMA_LOAD_WAIT_TIMEOUT_S");
    const double timeout = t != nullptr && atof(t) > 0.0 ? atof(t) : 60.0;
    if (idle.t_progress < 0.0) {
        idle.t_progress = idle.t_check = llama_lw_now();
        idle.seen = llama_lw_progress();
    }
    if ((m.field(48) != 0 || !llama_lw_alive(m.field(40))) && !llama_load_range_ready(r)) {
        llama_lw_fail(format("GUARD: loadwait: %s chunk %llu of %llu was not verified and the fetcher ended", m.name.c_str(),
                (unsigned long long) c, (unsigned long long) m.n_chunks));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const double now = llama_lw_now();
    if (now - idle.t_check >= 0.1) {
        idle.t_check = now;
        const uint64_t n = llama_lw_progress();
        if (n != idle.seen) {
            idle.seen = n;
            idle.t_progress = now;
        } else if (now - idle.t_progress >= timeout) {
            llama_lw_fail(format("GUARD: loadwait: %s chunk %llu of %llu not verified, no chunk verified for %.0f s", m.name.c_str(),
                    (unsigned long long) c, (unsigned long long) m.n_chunks, now - idle.t_progress));
        }
    }
}

// one write(2) per line on an O_APPEND file, so the follower never reads half a line; reopened when LLAMA_UPLOAD_LOG names another file
void llama_upload_log(const llama_load_range & r) {
    const char * path = getenv("LLAMA_UPLOAD_LOG");
    if (path == nullptr || *path == '\0' || !r.region) {
        return;
    }
    static std::mutex mtx;
    static std::string open_path;
    static int fd = -1;
    std::lock_guard<std::mutex> lock(mtx);
    if (open_path != path) {
        if (fd >= 0) {
            close(fd);
        }
        open_path = path;
        fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (fd < 0) {
            LLAMA_LOG_WARN("%s: cannot open %s (%s), its ranges are punched at the end\n", __func__, path, strerror(errno));
        }
    }
    const std::string line = format("uploaded %s %zu %zu\n", r.region->path.c_str(), r.offset, r.len);
    if (fd >= 0 && write(fd, line.data(), line.size()) != (ssize_t) line.size()) {
        LLAMA_LOG_WARN("%s: %s not written (%s), it is punched at the end\n", __func__, line.c_str(), strerror(errno));
    }
}
#else
bool llama_load_range_ready(const llama_load_range &) { return true; }
void llama_load_range_idle(const llama_load_range &, llama_load_idle &) {}
void llama_upload_log(const llama_load_range &) {}
#endif

void llama_upload_log_addr(const void * addr, size_t len) {
    const char * path = getenv("LLAMA_UPLOAD_LOG");
    if (path != nullptr && *path != '\0') {
        llama_upload_log(llama_load_range_of(addr, len));
    }
}
''')
edit(C, 'llama_file::llama_file(FILE * file) : pimpl(std::make_unique<impl>(file)) {}\n', '''llama_file::llama_file(FILE * file) : pimpl(std::make_unique<impl>(file)) {}

void llama_file::map_region(const void * addr, size_t size) const {
#if defined(__linux__)
    if (!llama_load_wait_active() && getenv("LLAMA_UPLOAD_LOG") == nullptr) {
        return;
    }
    char * real = realpath(pimpl->fname.c_str(), nullptr);
    if (real != nullptr) {
        llama_region_add(addr, size, real, lw); // STREAMPREFILL_PATCH
        free(real);
    }
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(size);
#endif
}
''')
edit(C, '''        const ranges & lazy_ranges) : pimpl(std::make_unique<impl>(file, prefetch, numa, lazy_ranges)) {}
llama_mmap::~llama_mmap() = default;
''', '''        const ranges & lazy_ranges) : pimpl(std::make_unique<impl>(file, prefetch, numa, lazy_ranges)) {
    file->map_region(addr(), size()); // STREAMPREFILL_PATCH
}
llama_mmap::~llama_mmap() {
    llama_unmap_region(addr()); // STREAMPREFILL_PATCH
}
''')

L = 'src/llama-model-loader.cpp'
edit(L, '''                if (llama_load_wait_active()) {
                    llama_load_wait_populate(data, n_size);
                }
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
''', '''                if (llama_load_wait_active()) {
                    llama_load_wait_populate(data, n_size);
                }
                ggml_backend_tensor_set(cur, data, 0, n_size);
                if (!ggml_backend_buffer_is_host(cur->buffer)) {
                    llama_upload_log_addr(data, n_size); // STREAMPREFILL_PATCH: the device holds it, the punch follower may free the RAM copy
                }
            }
''')

X = 'src/llama-context.cpp'
edit(X, '            if (llama_load_wait_active() && !model.no_alloc()) {\n                llama_load_wait_all("MoE cache prefill");',
     '            const char * stream_prof = getenv("LLAMA_MOE_CACHE_PROFILE"); // STREAMPREFILL_PATCH: a streaming prefill waits per expert and at its end\n'
     '            if (llama_load_wait_active() && !model.no_alloc() && !(llama_moe_stream_prefill() && stream_prof != nullptr && *stream_prof != \'\\0\')) {\n'
     '                llama_load_wait_all("MoE cache prefill");')

M = 'src/llama-moe-cache.cpp'
edit(M, '#include "llama-model.h"\n', '#include "llama-model.h"\n#include "llama-mmap.h" // STREAMPREFILL_PATCH\n')
edit(M, '        const int32_t dyn = getenv("LLAMA_MOE_CACHE_DYN") ? atoi(getenv("LLAMA_MOE_CACHE_DYN")) : 320;\n',
     '''        const int32_t dyn = getenv("LLAMA_MOE_CACHE_DYN") ? atoi(getenv("LLAMA_MOE_CACHE_DYN")) : 320;
        const bool stream = llama_moe_stream_prefill(); // STREAMPREFILL_PATCH
        struct upload {
            ggml_tensor * bank;
            const uint8_t * src;
            size_t dst;
            size_t size;
        };
        std::vector<upload> ups;
''')
edit(M, '''                ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + e*es, s*es, es);
                bytes += es;
''', '''                if (stream) {
                    ups.push_back({g.banks[ip], (const uint8_t *) w->data + e*es, (size_t) s*es, es}); // STREAMPREFILL_PATCH: the slot is fixed now, the bytes go up once verified
                } else {
                    ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + e*es, s*es, es);
                }
                bytes += es;
''')
edit(M, '        LLAMA_LOG_INFO("%s: prefilled %zu experts (%zu from the profile), %.2f GiB, %zu resident\\n"',
     '''        // STREAMPREFILL_PATCH: pinned set and slots are final, so upload in file order among the verified ranges, never blocked on one late expert
        if (stream) {
            std::sort(ups.begin(), ups.end(), [](const upload & a, const upload & b) { return a.src < b.src; });
            std::vector<llama_load_range> ranges;
            std::vector<size_t> pending;
            for (size_t i = 0; i < ups.size(); ++i) {
                ranges.push_back(llama_load_range_of(ups[i].src, ups[i].size));
                pending.push_back(i);
            }
            llama_load_idle idle;
            while (!pending.empty()) {
                size_t kept = 0;
                for (size_t k = 0; k < pending.size(); ++k) {
                    const size_t i = pending[k];
                    if (!llama_load_range_ready(ranges[i])) {
                        pending[kept++] = i;
                        continue;
                    }
                    if (llama_load_wait_active()) {
                        llama_load_wait_populate(ups[i].src, ups[i].size);
                    }
                    ggml_backend_tensor_set(ups[i].bank, ups[i].src, ups[i].dst, ups[i].size);
                    if (pin) {
                        llama_upload_log(ranges[i]);
                    }
                }
                if (kept == pending.size()) {
                    llama_load_range_idle(ranges[pending.front()], idle);
                } else {
                    idle = llama_load_idle();
                }
                pending.resize(kept);
            }
            if (llama_load_wait_active()) {
                llama_load_wait_all("MoE cache streaming prefill"); // the warm-up and every later miss copy verified host experts
            }
        }
        LLAMA_LOG_INFO("%s: prefilled %zu experts (%zu from the profile), %.2f GiB, %zu resident\\n"''')
