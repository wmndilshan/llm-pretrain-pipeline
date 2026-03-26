/*
 * Fast BPE Tokenizer Engine in C++
 * Compatible with HuggingFace byte-level BPE tokenizers
 *
 * Compile (MinGW): g++ -O3 -shared -o tokenizer.dll tokenizer.cpp -static -static-libgcc -static-libstdc++
 * Compile (MSVC): cl /O2 /LD /EHsc tokenizer.cpp /Fe:tokenizer.dll
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
#include <limits>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Hash for pair of ints
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<long long>()(((long long)p.first << 32) | (unsigned int)p.second);
    }
};

class BPETokenizer {
private:
    std::unordered_map<std::string, int> token_to_id;     // token -> id
    std::unordered_map<int, std::string> id_to_token;     // id -> token
    std::map<std::pair<int, int>, int> merge_priority;    // (id1, id2) -> priority (lower = higher priority)
    std::unordered_map<std::pair<int, int>, int, PairHash> merges;  // (id1, id2) -> merged_id

    int vocab_size_;
    int pad_id_, unk_id_, bos_id_, eos_id_;
    int next_merge_priority_;

    // GPT-2 style byte-to-unicode mapping
    std::unordered_map<unsigned char, std::string> byte_encoder;
    std::unordered_map<std::string, unsigned char> byte_decoder;

    // Encode a Unicode codepoint to UTF-8 string
    std::string codepoint_to_utf8(int cp) {
        std::string result;
        if (cp < 0x80) {
            result += (char)cp;
        } else if (cp < 0x800) {
            result += (char)(0xC0 | (cp >> 6));
            result += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            result += (char)(0xE0 | (cp >> 12));
            result += (char)(0x80 | ((cp >> 6) & 0x3F));
            result += (char)(0x80 | (cp & 0x3F));
        }
        return result;
    }

    void init_byte_encoding() {
        // GPT-2 uses a special byte-to-unicode mapping to avoid control characters
        // Bytes 33-126, 161-172, 174-255 map to themselves as unicode
        // Other bytes get mapped to 256+ range (encoded as UTF-8)
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_encoder[(unsigned char)b] = std::string(1, (char)b);
            } else {
                // Map to unicode range 256+n, encoded as UTF-8
                int codepoint = 256 + n;
                byte_encoder[(unsigned char)b] = codepoint_to_utf8(codepoint);
                n++;
            }
        }
        // Build reverse mapping
        for (auto& kv : byte_encoder) {
            byte_decoder[kv.second] = kv.first;
        }
    }

public:
    BPETokenizer() : vocab_size_(0), pad_id_(0), unk_id_(1), bos_id_(2), eos_id_(3), next_merge_priority_(0) {
        init_byte_encoding();
        // Initialize with special tokens
        add_token("<PAD>", pad_id_);
        add_token("<UNK>", unk_id_);
        add_token("<BOS>", bos_id_);
        add_token("<EOS>", eos_id_);
    }

    void add_token(const std::string& token, int id) {
        token_to_id[token] = id;
        id_to_token[id] = token;
        if (id >= vocab_size_) {
            vocab_size_ = id + 1;
        }
    }

    void add_merge(int id1, int id2, int merged_id) {
        auto pair = std::make_pair(id1, id2);
        merges[pair] = merged_id;
        merge_priority[pair] = next_merge_priority_++;
    }

    // Add merge by priority (for loading from HuggingFace format)
    void add_merge_with_priority(int id1, int id2, int merged_id, int priority) {
        auto pair = std::make_pair(id1, id2);
        merges[pair] = merged_id;
        merge_priority[pair] = priority;
    }

    // Convert text to byte-level tokens
    std::vector<std::string> text_to_byte_tokens(const std::string& text) {
        std::vector<std::string> tokens;
        for (unsigned char c : text) {
            tokens.push_back(byte_encoder[c]);
        }
        return tokens;
    }

    // Convert byte-level tokens back to text
    std::string byte_tokens_to_text(const std::vector<std::string>& tokens) {
        std::string result;
        for (const auto& tok : tokens) {
            // Try to decode the entire token first
            auto it = byte_decoder.find(tok);
            if (it != byte_decoder.end()) {
                result += (char)it->second;
                continue;
            }

            // Handle multi-character tokens by trying each UTF-8 sequence
            size_t i = 0;
            while (i < tok.size()) {
                bool found = false;

                // Try 2-byte UTF-8 first
                if (i + 1 < tok.size()) {
                    std::string two_byte = tok.substr(i, 2);
                    auto it2 = byte_decoder.find(two_byte);
                    if (it2 != byte_decoder.end()) {
                        result += (char)it2->second;
                        i += 2;
                        found = true;
                    }
                }

                // Try 1-byte
                if (!found) {
                    std::string one_byte = tok.substr(i, 1);
                    auto it1 = byte_decoder.find(one_byte);
                    if (it1 != byte_decoder.end()) {
                        result += (char)it1->second;
                    } else {
                        result += tok[i];  // fallback
                    }
                    i += 1;
                }
            }
        }
        return result;
    }

    // Tokenize single text - returns vector of token IDs
    std::vector<int> encode(const std::string& text, bool add_special = true) {
        std::vector<int> ids;

        if (add_special) {
            ids.push_back(bos_id_);
        }

        if (text.empty()) {
            if (add_special) {
                ids.push_back(eos_id_);
            }
            return ids;
        }

        // Convert text bytes to GPT-2 style unicode tokens and look up IDs
        std::vector<int> token_ids;
        for (unsigned char c : text) {
            std::string byte_tok = byte_encoder[c];
            auto it = token_to_id.find(byte_tok);
            if (it != token_to_id.end()) {
                token_ids.push_back(it->second);
            } else {
                // Token not in vocab - this shouldn't happen for byte-level BPE
                // Try to find individual bytes
                token_ids.push_back(unk_id_);
            }
        }

        // Apply BPE merges iteratively (greedy - always pick highest priority merge)
        bool changed = true;
        while (changed && token_ids.size() > 1) {
            changed = false;

            // Find the merge with highest priority (lowest priority number)
            int best_idx = -1;
            int best_priority = std::numeric_limits<int>::max();

            for (size_t i = 0; i < token_ids.size() - 1; i++) {
                auto pair = std::make_pair(token_ids[i], token_ids[i + 1]);
                auto it = merge_priority.find(pair);
                if (it != merge_priority.end() && it->second < best_priority) {
                    best_idx = static_cast<int>(i);
                    best_priority = it->second;
                }
            }

            if (best_idx >= 0) {
                // Apply the merge
                auto pair = std::make_pair(token_ids[best_idx], token_ids[best_idx + 1]);
                token_ids[best_idx] = merges[pair];
                token_ids.erase(token_ids.begin() + best_idx + 1);
                changed = true;
            }
        }

        // Add to output
        for (int id : token_ids) {
            ids.push_back(id);
        }

        if (add_special) {
            ids.push_back(eos_id_);
        }

        return ids;
    }

    // Decode token IDs back to text
    std::string decode(const std::vector<int>& ids, bool skip_special = true) {
        std::vector<std::string> tokens;

        for (int id : ids) {
            if (skip_special && (id == pad_id_ || id == bos_id_ || id == eos_id_ || id == unk_id_)) {
                continue;
            }
            auto it = id_to_token.find(id);
            if (it != id_to_token.end()) {
                tokens.push_back(it->second);
            }
        }

        return byte_tokens_to_text(tokens);
    }

    int get_vocab_size() const { return vocab_size_; }
    int get_pad_id() const { return pad_id_; }
    int get_unk_id() const { return unk_id_; }
    int get_bos_id() const { return bos_id_; }
    int get_eos_id() const { return eos_id_; }

    int get_token_id(const std::string& token) const {
        auto it = token_to_id.find(token);
        return (it != token_to_id.end()) ? it->second : unk_id_;
    }

    std::string get_token(int id) const {
        auto it = id_to_token.find(id);
        return (it != id_to_token.end()) ? it->second : "";
    }
};

// C API for Python bindings
extern "C" {

EXPORT void* tokenizer_create() {
    return new BPETokenizer();
}

EXPORT void tokenizer_destroy(void* tok) {
    delete static_cast<BPETokenizer*>(tok);
}

EXPORT void tokenizer_add_token(void* tok, const char* token, int id) {
    static_cast<BPETokenizer*>(tok)->add_token(token, id);
}

EXPORT void tokenizer_add_merge(void* tok, int id1, int id2, int merged_id) {
    static_cast<BPETokenizer*>(tok)->add_merge(id1, id2, merged_id);
}

EXPORT void tokenizer_add_merge_with_priority(void* tok, int id1, int id2, int merged_id, int priority) {
    static_cast<BPETokenizer*>(tok)->add_merge_with_priority(id1, id2, merged_id, priority);
}

// Encode single text - returns length, fills output buffer
EXPORT int tokenizer_encode(void* tok, const char* text, int* output, int max_len, int add_special) {
    auto* tokenizer = static_cast<BPETokenizer*>(tok);
    std::vector<int> ids = tokenizer->encode(text, add_special != 0);

    int len = std::min(static_cast<int>(ids.size()), max_len);
    for (int i = 0; i < len; i++) {
        output[i] = ids[i];
    }
    return len;
}

// Batch encode (no OpenMP for now - simpler build)
EXPORT void tokenizer_encode_batch(
    void* tok,
    const char** texts,
    int num_texts,
    int* output,      // flat array [num_texts * max_len]
    int* lengths,     // array [num_texts]
    int max_len,
    int add_special,
    int num_threads   // unused for now
) {
    auto* tokenizer = static_cast<BPETokenizer*>(tok);
    (void)num_threads;  // unused

    for (int i = 0; i < num_texts; i++) {
        std::vector<int> ids = tokenizer->encode(texts[i], add_special != 0);

        int len = std::min(static_cast<int>(ids.size()), max_len);
        lengths[i] = len;

        int offset = i * max_len;
        for (int j = 0; j < len; j++) {
            output[offset + j] = ids[j];
        }
        // Pad the rest
        for (int j = len; j < max_len; j++) {
            output[offset + j] = tokenizer->get_pad_id();
        }
    }
}

EXPORT int tokenizer_get_vocab_size(void* tok) {
    return static_cast<BPETokenizer*>(tok)->get_vocab_size();
}

EXPORT int tokenizer_get_pad_id(void* tok) {
    return static_cast<BPETokenizer*>(tok)->get_pad_id();
}

EXPORT int tokenizer_get_token_id(void* tok, const char* token) {
    return static_cast<BPETokenizer*>(tok)->get_token_id(token);
}

// Decode tokens back to text
EXPORT int tokenizer_decode(void* tok, const int* ids, int len, char* output, int max_output_len) {
    auto* tokenizer = static_cast<BPETokenizer*>(tok);
    std::vector<int> id_vec(ids, ids + len);
    std::string text = tokenizer->decode(id_vec, true);

    int copy_len = std::min(static_cast<int>(text.size()), max_output_len - 1);
    if (copy_len > 0) {
        memcpy(output, text.c_str(), copy_len);
    }
    output[copy_len] = '\0';
    return copy_len;
}

// Load vocabulary from file (format: token\tid per line)
EXPORT int tokenizer_load_vocab(void* tok, const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;

    auto* tokenizer = static_cast<BPETokenizer*>(tok);
    std::string line;
    while (std::getline(file, line)) {
        size_t tab = line.find('\t');
        if (tab != std::string::npos) {
            std::string token = line.substr(0, tab);
            int id = std::stoi(line.substr(tab + 1));
            tokenizer->add_token(token, id);
        }
    }
    return 1;
}

// Load merges from file (format: id1\tid2\tmerged_id per line)
EXPORT int tokenizer_load_merges(void* tok, const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;

    auto* tokenizer = static_cast<BPETokenizer*>(tok);
    std::string line;
    int priority = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int id1, id2, merged;
        if (iss >> id1 >> id2 >> merged) {
            tokenizer->add_merge_with_priority(id1, id2, merged, priority++);
        }
    }
    return 1;
}

} // extern "C"
