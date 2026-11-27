/*
 * aminfer.h — Public Plain-C ABI for libaminfer (ADR-0001, ADR-0022 C17/C21).
 *
 * Exposes in-process engine lifecycle, co-tenant wired-memory admission (R9),
 * cooperative atomic cancellation, native Qwen/DeepSeek BPE tokenization (W5),
 * synchronous & streaming generation (<think> & <tool_call> deltas), read-only
 * zero-copy fp16/fp32 logits & hidden-state views, embeddings & reranking (C21),
 * and zero-copy CVPixelBufferRef / IOSurfaceRef / IOSurfaceID vision ingress (G14).
 *
 * Zero Objective-C types and zero Python/CPython headers appear in this ABI.
 *
 * Dual-licensed under MIT OR Apache-2.0.
 */

#ifndef AMINFER_SDK_H
#define AMINFER_SDK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Public C-ABI version number. */
#define AMI_SDK_ABI_VERSION 1u

/* Maximum inline stop token IDs in ami_sdk_gen_params. */
#define AMI_SDK_MAX_STOP_TOKENS 8u

/* Sentinel value for unset / auto-resolved token IDs. */
#define AMI_SDK_TOKEN_NONE 0xFFFFFFFFu

/* Explicit return status codes (namespaced cleanly from internal bridge types). */
typedef int32_t ami_sdk_status;

enum {
    AMI_SDK_OK                    =  0,
    AMI_SDK_ERR_INVALID_ARGUMENT  = -1,
    AMI_SDK_ERR_ABI_MISMATCH      = -2,
    AMI_SDK_ERR_OUT_OF_MEMORY     = -3,
    AMI_SDK_ERR_IO                = -4,
    AMI_SDK_ERR_FORMAT            = -5,
    AMI_SDK_ERR_BUDGET_EXCEEDED   = -6,
    AMI_SDK_ERR_LIVE_RAM_EXCEEDED = -7,
    AMI_SDK_ERR_METAL             = -8,
    AMI_SDK_ERR_MODEL             = -9,
    AMI_SDK_ERR_TOKENIZER         = -10,
    AMI_SDK_ERR_BUFFER_TOO_SMALL  = -11,
    AMI_SDK_ERR_SEQUENCE_FULL     = -12,
    AMI_SDK_ERR_VISION            = -13,
    AMI_SDK_ERR_CANCELLED         = -14,
    AMI_SDK_ERR_UNSUPPORTED       = -15
};

/* Scalar element data types for zero-copy tensor views. */
typedef enum ami_sdk_dtype {
    AMI_SDK_DTYPE_F16 = 0,
    AMI_SDK_DTYPE_F32 = 1
} ami_sdk_dtype;

/* Pre-tokenizer dialect auto-detected from tokenizer.json (W5 / W24). */
typedef enum ami_sdk_dialect {
    AMI_SDK_DIALECT_QWEN     = 0,
    AMI_SDK_DIALECT_DEEPSEEK = 1,
    AMI_SDK_DIALECT_INKLING2 = 2,
    AMI_SDK_DIALECT_HY4      = 3
} ami_sdk_dialect;

/* Reasoning effort level for W24 / ADR-0024 template and thinking control. */
typedef enum ami_sdk_reasoning_effort {
    AMI_SDK_EFFORT_DEFAULT      = 0,
    AMI_SDK_EFFORT_NONE         = 1,
    AMI_SDK_EFFORT_NO_THINK     = 2,
    AMI_SDK_EFFORT_MINIMAL      = 3,
    AMI_SDK_EFFORT_LOW          = 4,
    AMI_SDK_EFFORT_MEDIUM       = 5,
    AMI_SDK_EFFORT_HIGH         = 6,
    AMI_SDK_EFFORT_XHIGH        = 7,
    AMI_SDK_EFFORT_MAX          = 8,
    AMI_SDK_EFFORT_CUSTOM_FLOAT = 9
} ami_sdk_reasoning_effort;

/* Streaming callback delta channel classification. */
typedef enum ami_sdk_delta_kind {
    AMI_SDK_DELTA_TEXT          = 0,
    AMI_SDK_DELTA_REASONING     = 1,
    AMI_SDK_DELTA_TOOL_CALL     = 2,
    AMI_SDK_DELTA_TOOL_CALL_END = 3
} ami_sdk_delta_kind;

/* Generation termination reason. */
typedef enum ami_sdk_finish_reason {
    AMI_SDK_FINISH_LENGTH        = 0,
    AMI_SDK_FINISH_STOP          = 1,
    AMI_SDK_FINISH_TOOL_CALLS    = 2,
    AMI_SDK_FINISH_CANCELLED     = 3,
    AMI_SDK_FINISH_CALLBACK_STOP = 4
} ami_sdk_finish_reason;

/* Pooling mode for ami_sdk_embed_* (C21). */
typedef enum ami_sdk_pooling_mode {
    AMI_SDK_POOL_LAST_TOKEN = 0,
    AMI_SDK_POOL_MEAN       = 1,
    AMI_SDK_POOL_CLS        = 2
} ami_sdk_pooling_mode;

/* Scoring mode for ami_sdk_rerank_* cross-encoder logit scoring (C21). */
typedef enum ami_sdk_rerank_mode {
    AMI_SDK_RERANK_LOGIT_DIFF = 0, /* logit[pos] - logit[neg] */
    AMI_SDK_RERANK_PROB_YES   = 1, /* exp(logit[pos]) / (exp(logit[pos]) + exp(logit[neg])) */
    AMI_SDK_RERANK_RAW_LOGIT  = 2  /* logit[pos] */
} ami_sdk_rerank_mode;

/* Zero-copy IOSurface pixel format for vision ingress (G14). */
typedef enum ami_sdk_vision_format {
    AMI_SDK_VISION_FMT_BGRA8 = 1,
    AMI_SDK_VISION_FMT_NV12  = 2,
    AMI_SDK_VISION_FMT_F16   = 3
} ami_sdk_vision_format;

/* Speculative draft mode for ami_sdk_engine_configure_speculative (M8.2 D14). */
typedef enum ami_sdk_draft_mode {
    AMI_SDK_DRAFT_NONE   = 0,
    AMI_SDK_DRAFT_AUTO   = 1,
    AMI_SDK_DRAFT_L1     = 2,
    AMI_SDK_DRAFT_DFLARE = 3,
    AMI_SDK_DRAFT_DFLY2  = 4
} ami_sdk_draft_mode;

/* Speculative decoding configuration attached to an engine handle (M8.2). */
typedef struct ami_sdk_spec_options {
    uint32_t struct_size;
    uint32_t draft_mode;              /* ami_sdk_draft_mode */
    const char *draft_amf_path;       /* NULL => auto-discover sibling dflare.amf / dfly2.amf */
    const char *draft_config_path;    /* NULL => auto-discover sibling JSON or embedded __meta */
    uint32_t max_tree_nodes;          /* 0 => 16 default */
    uint32_t enable_pruning;          /* 0 => disabled, 1 => D7 expert-budget pruning */
    float confidence_threshold;       /* <= 0.0f => config default (0.5f for DFly2) */
    uint32_t reserved;
} ami_sdk_spec_options;

/* Speculative decoding telemetry snapshot from an engine handle (M8.2). */
typedef struct ami_sdk_spec_stats {
    uint32_t struct_size;
    uint32_t active_draft_mode;       /* ami_sdk_draft_mode */
    uint64_t rounds;
    uint64_t emitted_tokens;
    uint64_t drafted_nodes;
    uint64_t dflare_rounds;
    uint64_t dflare_core3_batches;
    uint64_t dfly2_rounds;
    uint64_t dfly2_early_stops;
    double mean_accept_len;
} ami_sdk_spec_stats;

/* Opaque SDK handles. */
typedef struct ami_sdk_engine ami_sdk_engine;
typedef struct ami_sdk_tokenizer ami_sdk_tokenizer;
typedef struct ami_sdk_cancel_token ami_sdk_cancel_token;
typedef struct ami_sdk_vision_input ami_sdk_vision_input;

/* Host Mach VM / wired memory snapshot for co-tenant & zero-leak verification. */
typedef struct ami_sdk_host_vm_info {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t wired_bytes;
    uint64_t free_bytes;
    uint64_t live_reclaimable_bytes;
    uint64_t swapouts;
    uint64_t pageouts;
    uint64_t phys_footprint_bytes;
} ami_sdk_host_vm_info;

/* Engine startup and co-tenant memory budget options. */
typedef struct ami_sdk_engine_options {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *amf_path;
    const char *config_json_path;     /* NULL => auto-discover <amf_dir>/config.json */
    const char *tokenizer_json_path;  /* NULL => auto-discover <amf_dir>/tokenizer.json if present */
    uint64_t wired_budget_bytes;      /* 0 => automatic host ceiling (ADR-0012) */
    uint64_t headroom_bytes;          /* 0 => 3 GiB default OS headroom */
    uint64_t live_safety_margin_bytes;/* 0 => 512 MiB default R9 live margin */
    uint64_t live_reclaimable_override_bytes;
    uint32_t has_live_reclaimable_override;
    uint32_t strict_live_ram;         /* 1 => reject if required > live zero-swap budget (R9) */
    uint32_t max_seq_len;             /* 0 => 2048 default */
    uint32_t verify_crcs;             /* 1 => verify .amf CRC32 checksums */
    uint32_t startup_canary;          /* 1 => run G16 Metal numerical canary on load */
    uint32_t reserved;
} ami_sdk_engine_options;

/* Runtime model & wired-memory telemetry for an active engine handle. */
typedef struct ami_sdk_engine_stats {
    uint32_t struct_size;
    uint32_t is_graph_model;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t max_seq_len;
    uint32_t seq_len;
    uint32_t reserved;
    uint64_t total_wired_bytes;       /* model + SDK preallocated wired bytes */
    uint64_t sdk_wired_bytes;         /* SDK zero-copy view & scratch wired bytes */
    uint64_t wired_budget_bytes;
    uint64_t live_reclaimable_bytes;
    uint64_t live_budget_bytes;
} ami_sdk_engine_stats;

/* Per-request sampling and streaming configuration. */
typedef struct ami_sdk_gen_params {
    uint32_t struct_size;
    uint32_t max_tokens;
    float temperature;                /* <= 0.0f => greedy argmax */
    float top_p;                      /* (0, 1]; 1.0f => disabled */
    uint32_t top_k;                   /* 0 => disabled */
    float min_p;                      /* [0, 1); 0.0f => disabled */
    float repetition_penalty;         /* 1.0f => disabled */
    float presence_penalty;           /* 0.0f => disabled */
    float frequency_penalty;          /* 0.0f => disabled */
    uint32_t has_seed;                /* 0 => hardware TRNG (D11), 1 => explicit seed */
    uint64_t seed;
    uint32_t stop_token_ids[AMI_SDK_MAX_STOP_TOKENS];
    uint32_t stop_token_count;
    uint32_t ignore_eos;
    uint32_t start_in_think;          /* 1 => initial tokens start inside <think> */
    uint32_t reset_kv_cache;          /* 1 => reset KV cache before prefill, 0 => append */
    uint32_t skip_special_tokens;     /* 1 => omit special tokens from UTF-8 deltas */
    uint32_t reserved;
    uint32_t reasoning_effort;        /* ami_sdk_reasoning_effort */
    float reasoning_effort_value;     /* [0.0f, 0.99f] when AMI_SDK_EFFORT_CUSTOM_FLOAT */
    const ami_sdk_cancel_token *cancel_token;
} ami_sdk_gen_params;

/* Event payload delivered to ami_sdk_stream_cb during generation. */
typedef struct ami_sdk_stream_event {
    uint32_t struct_size;
    uint32_t token_id;
    uint32_t step_index;
    uint32_t delta_kind;              /* ami_sdk_delta_kind */
    const char *utf8_ptr;             /* valid only for the duration of the callback */
    size_t utf8_len;
    uint32_t is_final_flush;          /* 1 on end-of-stream partial-tag/UTF-8 flush */
    uint32_t reserved;
} ami_sdk_stream_event;

/* Return 0 to continue decoding, or non-zero to stop cooperatively. */
typedef int32_t (*ami_sdk_stream_cb)(const ami_sdk_stream_event *event, void *user_data);

/* Summary timings and accounting from a generation call. */
typedef struct ami_sdk_gen_result {
    uint32_t struct_size;
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint32_t finish_reason;           /* ami_sdk_finish_reason */
    uint64_t seed_used;
    double prefill_ns;
    double decode_ns;
} ami_sdk_gen_result;

/* Read-only zero-copy tensor view over preallocated wired memory. */
typedef struct ami_sdk_tensor_view {
    uint32_t struct_size;
    uint32_t dtype;                   /* ami_sdk_dtype */
    const void *data;                 /* 64-byte/page-aligned wired memory; never free */
    size_t byte_len;
    uint32_t ndim;
    uint32_t reserved;
    size_t shape[2];
} ami_sdk_tensor_view;

/* Embedding extraction options (C21). */
typedef struct ami_sdk_embed_params {
    uint32_t struct_size;
    uint32_t pooling;                 /* ami_sdk_pooling_mode */
    uint32_t dimensions;              /* 0 => full hidden_size; 1..H => Matryoshka slice */
    uint32_t normalize;               /* 1 => L2 normalize after slicing */
    const ami_sdk_cancel_token *cancel_token;
} ami_sdk_embed_params;

/* Cross-encoder reranking options (C21). */
typedef struct ami_sdk_rerank_params {
    uint32_t struct_size;
    uint32_t score_mode;              /* ami_sdk_rerank_mode */
    uint32_t positive_token_id;       /* AMI_SDK_TOKEN_NONE => auto ("yes"/"Yes"/1) */
    uint32_t negative_token_id;       /* AMI_SDK_TOKEN_NONE => auto ("no"/"No"/0) */
    const ami_sdk_cancel_token *cancel_token;
} ami_sdk_rerank_params;

/* Physical-page and Metal zero-copy metadata for an ami_sdk_vision_input (G14). */
typedef struct ami_sdk_vision_info {
    uint32_t struct_size;
    uint32_t format;                  /* ami_sdk_vision_format */
    uint32_t cv_pixel_format;
    uint32_t width;
    uint32_t height;
    uint32_t plane_count;
    uint32_t iosurface_id;
    uint32_t zero_copy_verified;
    uint32_t is_page_aligned;
    uint32_t reserved;
    uint64_t base_address;
    uint64_t surface_alloc_size;
} ami_sdk_vision_info;

/* Multimodal patchify + 2x2 merge + 3D MRoPE prefill options (G14). */
typedef struct ami_sdk_vision_params {
    uint32_t struct_size;
    uint32_t patch_size;              /* e.g. 14 or 16 */
    uint32_t t_patch;                 /* 1 (image) or 2 (video tube) */
    uint32_t spatial_merge;           /* e.g. 2 (2x2 merge) or 1 */
    float mean_rgb[3];
    float inv_std_rgb[3];
    uint32_t compute_logits;          /* 1 => compute final-token logits */
    uint32_t reserved;
    const ami_sdk_cancel_token *cancel_token;
} ami_sdk_vision_params;

/* Multimodal vision prefill telemetry & geometry output (G14). */
typedef struct ami_sdk_vision_result {
    uint32_t struct_size;
    uint32_t first_token_id;
    uint32_t prefix_tokens;
    uint32_t vision_tokens;
    uint32_t suffix_tokens;
    uint32_t total_prompt_tokens;
    uint32_t grid_t;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t all_frames_zero_copy;
    uint32_t all_frames_page_aligned;
    uint32_t reserved;
    double vision_encode_us;
} ami_sdk_vision_result;

/* -------------------------------------------------------------------------
 * ABI Version, Status Strings, Thread-Local Diagnostics & Host Telemetry
 * ------------------------------------------------------------------------- */

uint32_t ami_sdk_abi_version(void);
const char *ami_sdk_version_string(void);
const char *ami_sdk_status_string(ami_sdk_status status);
const char *ami_sdk_last_error(void);
void ami_sdk_clear_last_error(void);
ami_sdk_status ami_sdk_host_vm_snapshot(ami_sdk_host_vm_info *out_info);

/* Default struct initializers (set struct_size, abi_version, and safe defaults). */
void ami_sdk_engine_options_init(ami_sdk_engine_options *opts);
void ami_sdk_gen_params_init(ami_sdk_gen_params *params);
void ami_sdk_embed_params_init(ami_sdk_embed_params *params);
void ami_sdk_rerank_params_init(ami_sdk_rerank_params *params);
void ami_sdk_vision_params_init(ami_sdk_vision_params *params);
void ami_sdk_spec_options_init(ami_sdk_spec_options *opts);

/* Configure or disable speculative decoding (L1 / DFlare / DFly2 / Auto) on an engine. */
ami_sdk_status ami_sdk_engine_configure_speculative(ami_sdk_engine *engine,
                                                    const ami_sdk_spec_options *opts);
ami_sdk_status ami_sdk_engine_query_spec_stats(const ami_sdk_engine *engine,
                                               ami_sdk_spec_stats *out_stats);

/* -------------------------------------------------------------------------
 * Cooperative Cancellation Token (atomic flag, zero POSIX signal changes)
 * ------------------------------------------------------------------------- */

ami_sdk_status ami_sdk_cancel_token_create(ami_sdk_cancel_token **out_token);
void ami_sdk_cancel_token_destroy(ami_sdk_cancel_token *token);
void ami_sdk_cancel_token_cancel(ami_sdk_cancel_token *token);
void ami_sdk_cancel_token_reset(ami_sdk_cancel_token *token);
uint32_t ami_sdk_cancel_token_is_cancelled(const ami_sdk_cancel_token *token);

/* -------------------------------------------------------------------------
 * Native Byte-Level BPE Tokenizer (W5: Qwen & DeepSeek dialects)
 * ------------------------------------------------------------------------- */

ami_sdk_status ami_sdk_tokenizer_from_file(const char *path,
                                           ami_sdk_tokenizer **out_tokenizer);
ami_sdk_status ami_sdk_tokenizer_from_json(const char *json_ptr, size_t json_len,
                                           ami_sdk_tokenizer **out_tokenizer);
void ami_sdk_tokenizer_destroy(ami_sdk_tokenizer *tokenizer);

uint32_t ami_sdk_tokenizer_vocab_size(const ami_sdk_tokenizer *tokenizer);
uint32_t ami_sdk_tokenizer_dialect(const ami_sdk_tokenizer *tokenizer);
ami_sdk_status ami_sdk_tokenizer_special_id(const ami_sdk_tokenizer *tokenizer,
                                            const char *token_str, size_t token_len,
                                            uint32_t *out_id);

/*
 * Encodes UTF-8 `text_ptr[0..text_len)` into `out_ids[0..out_cap)`.
 * Pass `out_ids == NULL` and `out_cap == 0` to query exact token count in `*out_len`.
 * If `out_ids != NULL` and `out_cap < needed`, sets `*out_len = needed` and returns
 * `AMI_SDK_ERR_BUFFER_TOO_SMALL`.
 */
ami_sdk_status ami_sdk_tokenize(const ami_sdk_tokenizer *tokenizer,
                                const char *text_ptr, size_t text_len,
                                uint32_t allow_special_tokens,
                                uint32_t *out_ids, size_t out_cap,
                                size_t *out_len);

ami_sdk_status ami_sdk_count_tokens(const ami_sdk_tokenizer *tokenizer,
                                    const char *text_ptr, size_t text_len,
                                    uint32_t allow_special_tokens,
                                    size_t *out_count);

/*
 * Decodes `ids_ptr[0..ids_len)` into `out_utf8[0..out_cap)`.
 * Pass `out_utf8 == NULL` and `out_cap == 0` to query exact byte length in `*out_len`.
 */
ami_sdk_status ami_sdk_detokenize(const ami_sdk_tokenizer *tokenizer,
                                  const uint32_t *ids_ptr, size_t ids_len,
                                  uint32_t skip_special_tokens,
                                  char *out_utf8, size_t out_cap,
                                  size_t *out_len);

/* -------------------------------------------------------------------------
 * Engine Lifecycle, Co-Tenant Memory Governance & Low-Level Stepping
 * ------------------------------------------------------------------------- */

ami_sdk_status ami_sdk_engine_create(const ami_sdk_engine_options *opts,
                                     ami_sdk_engine **out_engine);
void ami_sdk_engine_destroy(ami_sdk_engine *engine);

ami_sdk_status ami_sdk_engine_reset_kv(ami_sdk_engine *engine);
ami_sdk_status ami_sdk_engine_query_stats(const ami_sdk_engine *engine,
                                          ami_sdk_engine_stats *out_stats);

/*
 * Sets or queries the C27 opt-in approximate cold-tail skipping threshold
 * (`0.0f` default = strictly lossless; values `< 0.0f`, `>= 1.0f`, or NaN
 * are rejected with `AMI_SDK_ERR_INVALID_ARGUMENT`).
 */
ami_sdk_status ami_sdk_engine_set_cold_expert_min_weight(ami_sdk_engine *engine,
                                                         float min_weight);
ami_sdk_status ami_sdk_engine_get_cold_expert_min_weight(const ami_sdk_engine *engine,
                                                         float *out_min_weight);

/* Borrowed pointer to the engine's loaded tokenizer, or NULL if none was loaded. */
const ami_sdk_tokenizer *ami_sdk_engine_tokenizer(const ami_sdk_engine *engine);

/* -------------------------------------------------------------------------
 * Universal Hot-Swap LoRA & DoRA Adapter Management (M9.5 W30)
 * ------------------------------------------------------------------------- */

typedef struct ami_sdk_lora_info {
    uint32_t struct_size;
    uint32_t active_adapters;
    uint32_t active_slot;       /* UINT32_MAX when base model is active */
    uint32_t max_adapters;
    uint64_t pool_capacity_bytes;
    uint64_t pool_used_bytes;
} ami_sdk_lora_info;

/*
 * Preallocates the 16 KiB-aligned wired LoRA pool on an engine (both Qwen3
 * and GraphModel / Inkling 2 / Hy4). Pass `pool_bytes == 0` for the default
 * pool size (`4 MiB`).
 */
ami_sdk_status ami_sdk_engine_init_lora_pool(ami_sdk_engine *engine,
                                             size_t pool_bytes);

/*
 * Hot-loads an `.amlora` bundle from `amlora_path` under adapter `name`.
 * Lazily initializes the wired LoRA pool on first load if not yet initialized,
 * and calibrates exact PEFT DoRA row scales against base weights.
 * Optional `out_slot_id` receives the assigned slot index (`0..15`).
 */
ami_sdk_status ami_sdk_engine_load_lora(ami_sdk_engine *engine,
                                        const char *name,
                                        const char *amlora_path,
                                        uint32_t *out_slot_id);

/*
 * Hot-unloads adapter `name`, immediately returning its wired 16 KiB pages
 * to the pool. Resets active adapter to base model if `name` was active.
 */
ami_sdk_status ami_sdk_engine_unload_lora(ami_sdk_engine *engine,
                                          const char *name);

/*
 * Activates loaded adapter `name`, or resets to bit-exact base model when
 * `name == NULL` or `name[0] == '\0'`.
 */
ami_sdk_status ami_sdk_engine_set_active_lora(ami_sdk_engine *engine,
                                              const char *name);

/* Queries current LoRA pool usage and active adapter state. */
ami_sdk_status ami_sdk_engine_query_lora_info(const ami_sdk_engine *engine,
                                              ami_sdk_lora_info *out_info);

ami_sdk_status ami_sdk_prefill_tokens(ami_sdk_engine *engine,
                                      const uint32_t *prompt_ids, size_t prompt_len,
                                      uint32_t compute_logits,
                                      uint32_t reset_kv_cache,
                                      const ami_sdk_cancel_token *cancel_token);

ami_sdk_status ami_sdk_step_decode(ami_sdk_engine *engine,
                                   uint32_t token_id,
                                   uint32_t *out_next_token_id);

/* -------------------------------------------------------------------------
 * Synchronous & Streaming Generation (<think> & <tool_call> deltas)
 * ------------------------------------------------------------------------- */

ami_sdk_status ami_sdk_generate_tokens(ami_sdk_engine *engine,
                                       const uint32_t *prompt_ids, size_t prompt_len,
                                       const ami_sdk_gen_params *params,
                                       ami_sdk_stream_cb stream_cb, void *user_data,
                                       uint32_t *out_token_ids, size_t out_token_cap,
                                       ami_sdk_gen_result *out_result);

ami_sdk_status ami_sdk_generate_text(ami_sdk_engine *engine,
                                     const char *prompt_utf8, size_t prompt_len,
                                     const ami_sdk_gen_params *params,
                                     ami_sdk_stream_cb stream_cb, void *user_data,
                                     uint32_t *out_token_ids, size_t out_token_cap,
                                     ami_sdk_gen_result *out_result);

/* -------------------------------------------------------------------------
 * Zero-Copy Logits & Final Hidden-State Views (valid until next step/destroy)
 * ------------------------------------------------------------------------- */

ami_sdk_status ami_sdk_get_logits_view(ami_sdk_engine *engine,
                                       uint32_t dtype,
                                       ami_sdk_tensor_view *out_view);

ami_sdk_status ami_sdk_get_hidden_state_view(ami_sdk_engine *engine,
                                             uint32_t dtype,
                                             uint32_t normalized,
                                             ami_sdk_tensor_view *out_view);

/* -------------------------------------------------------------------------
 * Native Embeddings & Cross-Encoder Reranking (C21, lm_head elided on embed)
 * ------------------------------------------------------------------------- */

ami_sdk_status ami_sdk_embed_tokens(ami_sdk_engine *engine,
                                    const uint32_t *token_ids, size_t token_len,
                                    const ami_sdk_embed_params *params,
                                    float *out_embedding, size_t out_cap,
                                    size_t *out_dims);

ami_sdk_status ami_sdk_embed_text(ami_sdk_engine *engine,
                                  const char *text_ptr, size_t text_len,
                                  const ami_sdk_embed_params *params,
                                  float *out_embedding, size_t out_cap,
                                  size_t *out_dims);

ami_sdk_status ami_sdk_rerank_tokens(ami_sdk_engine *engine,
                                     const uint32_t *query_ids, size_t query_len,
                                     const uint32_t *const *doc_id_ptrs,
                                     const size_t *doc_id_lens,
                                     size_t num_docs,
                                     const ami_sdk_rerank_params *params,
                                     float *out_scores);

ami_sdk_status ami_sdk_rerank_texts(ami_sdk_engine *engine,
                                    const char *query_ptr, size_t query_len,
                                    const char *const *doc_ptrs,
                                    const size_t *doc_lens,
                                    size_t num_docs,
                                    const ami_sdk_rerank_params *params,
                                    float *out_scores);

/* -------------------------------------------------------------------------
 * Zero-Copy Multimodal Vision Ingress (G14: CVPixelBufferRef / IOSurfaceRef)
 * ------------------------------------------------------------------------- */

ami_sdk_status ami_sdk_vision_from_cvpixelbuffer(ami_sdk_engine *engine,
                                                 void *cv_pixel_buffer_ref,
                                                 ami_sdk_vision_input **out_vision);

ami_sdk_status ami_sdk_vision_from_iosurface(ami_sdk_engine *engine,
                                             void *iosurface_ref,
                                             ami_sdk_vision_input **out_vision);

ami_sdk_status ami_sdk_vision_from_iosurface_id(ami_sdk_engine *engine,
                                                uint32_t iosurface_id,
                                                ami_sdk_vision_input **out_vision);

ami_sdk_status ami_sdk_vision_create_iosurface(ami_sdk_engine *engine,
                                               uint32_t format,
                                               uint32_t width, uint32_t height,
                                               ami_sdk_vision_input **out_vision);

ami_sdk_status ami_sdk_vision_load_image(ami_sdk_engine *engine,
                                         const char *image_path,
                                         ami_sdk_vision_input **out_vision);

void ami_sdk_vision_destroy(ami_sdk_vision_input *vision);

ami_sdk_status ami_sdk_vision_query(const ami_sdk_vision_input *vision,
                                    ami_sdk_vision_info *out_info);

/* Borrowed underlying CVPixelBufferRef / IOSurfaceRef pointers (valid while vision lives). */
void *ami_sdk_vision_cvpixelbuffer(const ami_sdk_vision_input *vision);
void *ami_sdk_vision_iosurface(const ami_sdk_vision_input *vision);

ami_sdk_status ami_sdk_vision_lock_plane(ami_sdk_vision_input *vision,
                                         uint32_t plane, uint32_t read_only,
                                         void **out_base_ptr,
                                         size_t *out_bytes_per_row,
                                         uint32_t *out_height);

ami_sdk_status ami_sdk_vision_unlock(ami_sdk_vision_input *vision,
                                     uint32_t read_only);

ami_sdk_status ami_sdk_prefill_vision(ami_sdk_engine *engine,
                                      const uint32_t *prefix_ids, size_t prefix_len,
                                      const ami_sdk_vision_input *const *frames,
                                      size_t frame_count,
                                      const uint32_t *suffix_ids, size_t suffix_len,
                                      const ami_sdk_vision_params *params,
                                      ami_sdk_vision_result *out_result);

/* -------------------------------------------------------------------------
 * Native Multi-Threaded Weighted OLS RCO Affine Quantizer (gsq-convert)
 * ------------------------------------------------------------------------- */

/* src_dtype: 0 = F32, 1 = F16, 2 = BF16.
 * bits: 2, 3, 4, 5, 6, or 8. group_size: 32, 64, or 128 (k % group_size == 0).
 * col_weights: optional float[k] per-column activation variance E[x_c^2] (NULL for unweighted).
 * num_threads: 0 = auto (clamped to min(cpu_count, n_rows)).
 * dst_qweight: caller buffer of size n_rows * ((k * bits) / 8) bytes (LSB-first MLX affine packing).
 * dst_params: caller buffer of size n_rows * (k / group_size) * 4 bytes (packed little-endian [f16 scale, f16 bias]).
 * out_mean_mse: optional double* receiving mean per-group (weighted or unweighted) MSE. */
ami_sdk_status ami_sdk_quantize_rows_affine_raw(
    const void *src_bytes,
    uint32_t src_dtype,
    uint32_t n_rows,
    uint32_t k,
    uint32_t bits,
    uint32_t group_size,
    uint32_t use_rco,
    uint32_t rco_iters,
    const float *col_weights,
    uint32_t num_threads,
    uint8_t *dst_qweight,
    uint8_t *dst_params,
    double *out_mean_mse);

/* Batch-dequantize row-major NVFP4_G16 (`k % 16 == 0`) rows into float32
 * `[n_rows, k]` (`data_bytes`: `n_rows * (k / 2)`, `scale_bytes`: `n_rows * (k / 16)`
 * OCP FP8 E4M3FN bytes, multiplied by `outer_scale`). */
ami_sdk_status ami_sdk_dequant_nvfp4_rows_f32(
    const uint8_t *data_bytes,
    const uint8_t *scale_bytes,
    float outer_scale,
    uint32_t n_rows,
    uint32_t k,
    float *dst_f32);

#ifdef __cplusplus
}
#endif

#endif /* AMINFER_SDK_H */
