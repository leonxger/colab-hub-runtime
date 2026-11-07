"""Load while downloading (M15-T12): with LLAMA_LOAD_WAIT set, every read of model file bytes waits until the fetcher verified the chunks under it. usage: python3 patch_loadwait.py LLAMA_SRC_DIR"""
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
edit(H, 'struct llama_mlock;\n', 'struct llama_mlock;\nstruct llama_load_wait_map; // LOADWAIT_PATCH\n')
edit(H, '    bool has_direct_io() const;\nprivate:\n    struct impl;\n    std::unique_ptr<impl> pimpl;\n};\n',
     '''    bool has_direct_io() const;

    // LOADWAIT_PATCH: blocks until the fetcher verified every chunk of [offset, offset + len)
    void wait_range(size_t offset, size_t len) const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
    std::shared_ptr<llama_load_wait_map> lw;
};
''')
edit(H, 'size_t llama_path_max();\n', '''size_t llama_path_max();

// LOADWAIT_PATCH: LLAMA_LOAD_WAIT names the fetcher's directory of chunk maps (vm/lib/hfetch.py); a file with a map is read only where its chunks passed their sha256
bool llama_load_wait_active();
void llama_load_wait_populate(const void * addr, size_t len);
void llama_load_wait_all(const char * why);
''')

C = 'src/llama-mmap.cpp'
edit(C, '// llama_file\n\nstruct llama_file::impl {\n', r'''// LOADWAIT_PATCH: one map per model file, <LLAMA_LOAD_WAIT>/<st_dev>-<st_ino>.map: magic, version, file size, chunk bytes, chunk count, fetcher pid, state (0 fetching, 1 done, 2 ended), flags (1 post phase), then one byte per chunk set to 1 after its sha256 passed
struct llama_load_wait_map {
    std::string name;
    std::string path;
    unsigned long long map_ino = 0;
    const uint8_t * base = nullptr;
    size_t map_size = 0;
    uint64_t file_size = 0;
    uint64_t chunk = 0;
    uint64_t n_chunks = 0;
    bool post = false;
    bool missing = false;

    ~llama_load_wait_map() {
#if defined(__linux__)
        if (base) {
            munmap((void *) base, map_size);
        }
#endif
    }
    uint64_t field(size_t off) const { return __atomic_load_n((const uint64_t *) (base + off), __ATOMIC_ACQUIRE); }
    bool has(uint64_t c) const { return __atomic_load_n(base + 64 + c, __ATOMIC_ACQUIRE) != 0; }
    uint64_t count() const {
        uint64_t n = 0;
        for (uint64_t c = 0; c < n_chunks; ++c) {
            n += has(c);
        }
        return n;
    }
};

static const char * llama_lw_dir() {
    const char * dir = getenv("LLAMA_LOAD_WAIT");
    return dir != nullptr ? dir : "";
}

bool llama_load_wait_active() {
    return *llama_lw_dir() != '\0';
}

#if defined(__linux__)
static double llama_lw_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] static void llama_lw_fail(const std::string & msg) {
    LLAMA_LOG_ERROR("%s\n", msg.c_str());
    throw std::runtime_error(msg);
}

struct llama_lw_state {
    std::mutex mtx;
    std::vector<std::shared_ptr<llama_load_wait_map>> opened;
    std::vector<std::shared_ptr<llama_load_wait_map>> seen;
    double scanned = -1.0;
};

static llama_lw_state & llama_lw() {
    static llama_lw_state s;
    return s;
}

static bool llama_lw_current(const llama_load_wait_map & m) {
    struct stat st{};
    return stat(m.path.c_str(), &st) == 0 && (unsigned long long) st.st_ino == m.map_ino;
}

static std::shared_ptr<llama_load_wait_map> llama_lw_map_open(const std::string & path, const std::string & name) {
    auto m = std::make_shared<llama_load_wait_map>();
    m->name = name;
    m->path = path;
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        m->missing = true;
        return m;
    }
    struct stat st{};
    void * p = MAP_FAILED;
    if (fstat(fd, &st) == 0 && st.st_size >= 64) {
        p = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    }
    close(fd);
    if (p == MAP_FAILED) {
        llama_lw_fail(format("GUARD: loadwait: chunk map %s is unreadable", path.c_str()));
    }
    m->base = (const uint8_t *) p;
    m->map_size = st.st_size;
    m->map_ino = st.st_ino;
    m->file_size = m->field(16);
    m->chunk = m->field(24);
    m->n_chunks = m->field(32);
    m->post = (m->field(56) & 1) != 0;
    if (memcmp(m->base, "HUBLWAIT", 8) != 0 || m->field(8) != 1 || m->chunk == 0 ||
            m->n_chunks != (m->file_size + m->chunk - 1) / m->chunk || m->map_size < 64 + m->n_chunks) {
        llama_lw_fail(format("GUARD: loadwait: chunk map %s has a bad header", path.c_str()));
    }
    return m;
}

static std::shared_ptr<llama_load_wait_map> llama_load_wait_open(int fd, const char * fname) {
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        llama_lw_fail(format("GUARD: loadwait: cannot stat %s", fname));
    }
    std::string name = fname;
    if (name.find_last_of('/') != std::string::npos) {
        name = name.substr(name.find_last_of('/') + 1);
    }
    const std::string path = format("%s/%llu-%llu.map", llama_lw_dir(), (unsigned long long) st.st_dev, (unsigned long long) st.st_ino);
    auto & s = llama_lw();
    std::lock_guard<std::mutex> lock(s.mtx);
    for (const auto & m : s.opened) {
        if (m->path == path && llama_lw_current(*m)) {
            return m;
        }
    }
    auto m = llama_lw_map_open(path, name);
    if (m->missing) {
        LLAMA_LOG_WARN("load_wait: %s has no chunk map, a read of it fails\n", name.c_str());
        return m;
    }
    if (m->file_size != (uint64_t) st.st_size) {
        llama_lw_fail(format("GUARD: loadwait: chunk map of %s is for %llu bytes, the file has %llu", name.c_str(),
                (unsigned long long) m->file_size, (unsigned long long) st.st_size));
    }
    s.opened.push_back(m);
    LLAMA_LOG_INFO("load_wait: %s: %llu chunks of %llu MiB, %llu verified at open\n", name.c_str(),
            (unsigned long long) m->n_chunks, (unsigned long long) (m->chunk >> 20), (unsigned long long) m->count());
    return m;
}

// every map of the directory counts as fetch progress, also those of files this process never opens
static uint64_t llama_lw_progress() {
    auto & s = llama_lw();
    std::lock_guard<std::mutex> lock(s.mtx);
    const double now = llama_lw_now();
    if (now - s.scanned >= 1.0) {
        s.scanned = now;
        if (DIR * d = llama_load_wait_active() ? opendir(llama_lw_dir()) : nullptr) {
            while (const struct dirent * e = readdir(d)) {
                const std::string entry = e->d_name;
                if (entry.size() < 5 || entry.compare(entry.size() - 4, 4, ".map") != 0) {
                    continue;
                }
                const std::string path = std::string(llama_lw_dir()) + "/" + entry;
                bool known = false;
                for (const auto & m : s.seen) {
                    known = known || (m->path == path && llama_lw_current(*m));
                }
                if (!known) {
                    try {
                        auto m = llama_lw_map_open(path, entry);
                        if (!m->missing) {
                            s.seen.push_back(m);
                        }
                    } catch (const std::exception &) {
                    }
                }
            }
            closedir(d);
        }
    }
    uint64_t n = 0;
    for (const auto & m : s.seen) {
        n += m->count();
    }
    for (const auto & m : s.opened) {
        n += m->count();
    }
    return n;
}

static bool llama_lw_alive(uint64_t pid) {
    return pid > 0 && (kill((pid_t) pid, 0) == 0 || errno == EPERM);
}

// GUARD: loadwait on fetch progress, never on a fixed wait per tensor: no chunk verified anywhere for LLAMA_LOAD_WAIT_TIMEOUT_S (60), or at once when the fetcher ended
static void llama_lw_wait_chunk(const llama_load_wait_map & m, uint64_t c) {
    if (m.has(c)) {
        return;
    }
    const char * t = getenv("LLAMA_LOAD_WAIT_TIMEOUT_S");
    const double timeout = t != nullptr && atof(t) > 0.0 ? atof(t) : 60.0;
    const double t0 = llama_lw_now();
    double t_progress = t0;
    double t_check = t0;
    uint64_t seen = llama_lw_progress();
    while (!m.has(c)) {
        if (m.field(48) != 0 || !llama_lw_alive(m.field(40))) {
            if (m.has(c)) {
                break;
            }
            llama_lw_fail(format("GUARD: loadwait: %s chunk %llu of %llu was not verified and the fetcher ended", m.name.c_str(),
                    (unsigned long long) c, (unsigned long long) m.n_chunks));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const double now = llama_lw_now();
        if (now - t_check >= 0.1) {
            t_check = now;
            const uint64_t n = llama_lw_progress();
            if (n != seen) {
                seen = n;
                t_progress = now;
            } else if (now - t_progress >= timeout) {
                llama_lw_fail(format("GUARD: loadwait: %s chunk %llu of %llu not verified, no chunk verified for %.0f s", m.name.c_str(),
                        (unsigned long long) c, (unsigned long long) m.n_chunks, now - t_progress));
            }
        }
    }
}

static void llama_lw_wait_range(const llama_load_wait_map & m, size_t offset, size_t len) {
    if (m.missing) {
        llama_lw_fail(format("GUARD: loadwait: %s has no chunk map in %s", m.name.c_str(), llama_lw_dir()));
    }
    if (len == 0 || offset >= m.file_size) {
        return;
    }
    const uint64_t last = std::min<uint64_t>((uint64_t) offset + len, m.file_size) - 1;
    for (uint64_t c = offset / m.chunk; c <= last / m.chunk; ++c) {
        llama_lw_wait_chunk(m, c);
    }
}

void llama_load_wait_all(const char * why) {
    std::vector<std::shared_ptr<llama_load_wait_map>> maps;
    {
        auto & s = llama_lw();
        std::lock_guard<std::mutex> lock(s.mtx);
        for (const auto & m : s.opened) {
            if (!m->missing && !m->post && llama_lw_current(*m)) {
                maps.push_back(m);
            }
        }
    }
    const double t0 = llama_lw_now();
    for (const auto & m : maps) {
        llama_lw_wait_range(*m, 0, m->file_size);
    }
    LLAMA_LOG_INFO("load_wait: %s after every chunk of %zu files was verified, waited %.2f s\n", why, maps.size(), llama_lw_now() - t0);
}

void llama_load_wait_populate(const void * addr, size_t len) {
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t beg = (uintptr_t) addr & ~(page - 1);
    const uintptr_t end = ((uintptr_t) addr + len + page - 1) & ~(page - 1);
    if (len > 0) {
        madvise((void *) beg, end - beg, MADV_POPULATE_READ);
    }
}
#else
static std::shared_ptr<llama_load_wait_map> llama_load_wait_open(int, const char *) { return nullptr; }
static void llama_lw_wait_range(const llama_load_wait_map &, size_t, size_t) {}
void llama_load_wait_all(const char *) {}
void llama_load_wait_populate(const void *, size_t) {}
#endif

void llama_file::wait_range(size_t offset, size_t len) const {
    if (lw) {
        llama_lw_wait_range(*lw, offset, len);
    }
}

// llama_file

struct llama_file::impl {
''')
edit(C, '#include <algorithm>\n', '''#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <dirent.h>
#include <signal.h>
#endif
''')
edit(C, '''llama_file::llama_file(const char * fname, const char * mode, const bool use_direct_io) :
    pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}
''', '''llama_file::llama_file(const char * fname, const char * mode, const bool use_direct_io) :
    pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {
    if (llama_load_wait_active() && std::strcmp(mode, "rb") == 0) {
        lw = llama_load_wait_open(file_id(), fname); // LOADWAIT_PATCH
        if (lw && pimpl->fp != nullptr) {
            setvbuf(pimpl->fp, nullptr, _IONBF, 0); // a stdio buffer would keep bytes read ahead of their chunk's sha256
        }
    }
}
''')
edit(C, 'void llama_file::read_raw(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }\n',
     'void llama_file::read_raw(void * ptr, size_t len) { wait_range(lw ? tell() : 0, len); pimpl->read_raw(ptr, len); } // LOADWAIT_PATCH\n')
edit(C, '#ifdef _WIN32\nvoid llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }\n#else\nvoid llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }\n#endif\n',
     '#ifdef _WIN32\nvoid llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }\n#else\n'
     'void llama_file::read_raw_unsafe(void * ptr, size_t len) { wait_range(lw ? tell() : 0, len); pimpl->read_raw_unsafe(ptr, len); } // LOADWAIT_PATCH\n#endif\n')
edit(C, 'uint32_t llama_file::read_u32() { return pimpl->read_u32(); }\n',
     'uint32_t llama_file::read_u32() { wait_range(lw ? tell() : 0, sizeof(uint32_t)); return pimpl->read_u32(); } // LOADWAIT_PATCH\n')

L = 'src/llama-model-loader.cpp'
edit(L, '            const size_t prefetch_size = prefetch && use_mmap ? -1 : 0;\n',
     '            const size_t prefetch_size = prefetch && use_mmap && !llama_load_wait_active() ? -1 : 0; // LOADWAIT_PATCH: no MAP_POPULATE over holes\n')
edit(L, '''    if (use_mmap) {
        data = (const uint8_t *) mappings.at(w.idx)->addr() + w.offs + offs;
    } else {''', '''    if (use_mmap) {
        data = (const uint8_t *) mappings.at(w.idx)->addr() + w.offs + offs;
        files.at(w.idx)->wait_range(w.offs + offs, size); // LOADWAIT_PATCH
    } else {''')
edit(L, '    if (!use_mmap) {\n        std::stable_sort(tensors.begin(), tensors.end(),',
     '    // LOADWAIT_PATCH: the async upload needs no staging buffer, so its tensors load in file order like the download\n'
     '    if (!use_mmap && !(upload_backend && llama_load_wait_active())) {\n        std::stable_sort(tensors.begin(), tensors.end(),')
edit(L, '''            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {''', '''            if (check_tensors) {
                files.at(weight->idx)->wait_range(weight->offs, n_size); // LOADWAIT_PATCH
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {''')
edit(L, '''            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            const auto & file = files.at(weight->idx);''', '''            } else {
                files.at(weight->idx)->wait_range(weight->offs, n_size); // LOADWAIT_PATCH
                if (llama_load_wait_active()) {
                    llama_load_wait_populate(data, n_size);
                }
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            const auto & file = files.at(weight->idx);''')

X = 'src/llama-context.cpp'
edit(X, '            moe_cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size, cparams.moe_cache_cpu_share);\n',
     '''            if (llama_load_wait_active() && !model.no_alloc()) {
                llama_load_wait_all("MoE cache prefill"); // LOADWAIT_PATCH: the prefill and every cache miss copy host experts, so every shard chunk is verified first
            }
            moe_cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size, cparams.moe_cache_cpu_share);
''')
