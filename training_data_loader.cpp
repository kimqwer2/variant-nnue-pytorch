#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

using namespace bin;
using namespace chess;

static constexpr int MAX_PIECES = PIECE_COUNT;
static constexpr int MAX_HAND_PIECES = POCKETS ? 2 * static_cast<int>(File::FILE_NB) : 0;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return flip_horizontally(flip_vertically(sq));
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (sq == Square::NB)
        // map missing king to zero
        return Square::MIN;
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return flip_vertically(sq);
    }
}


static Square mirror_square_with_files(Square sq)
{
    if (sq == Square::NB)
        return sq;

    const int sq_idx = static_cast<int>(sq);
    const int rank = sq_idx / FILES;
    const int file = sq_idx % FILES;
    return static_cast<Square>(rank * FILES + ((FILES - 1) - file));
}

static int map_king(Square sq)
{
    // palace squares for Xiangi/Janggi
    // map accessible king squares skipping the gaps
    if (Square::KNB == Square(9) && Square::KNB != Square::NB)
        return (int(sq) - 6 * (int(sq) / int(File::FILE_NB)) - 3) % int(Square::KNB);

    return int(sq) % int(Square::KNB);
}

static float calculate_janggi_winner(const Position& pos, float deom)
{
    auto piece_points = [](PieceType pt) -> float {
        // Janggi mapping requested by user:
        // Rook=13, JanggiCannon=7, Horse=5, JanggiElephant=3, Wazir=3, Soldier=2.
        // The current shared enum names are chess-like in this codebase, so we map
        // by available symbols and preserve the intended Janggi point scale.
        if (pt == PieceType::Rook) return 13.0f;
        if (pt == PieceType::Queen) return 7.0f;   // JanggiCannon-equivalent slot
        if (pt == PieceType::Knight) return 5.0f;  // Horse
        if (pt == PieceType::Bishop) return 3.0f;  // JanggiElephant
        if (pt == PieceType::Pawn) return 2.0f;    // Soldier

        const int pti = static_cast<int>(pt);
        if (pti == 5) return 3.0f; // Wazir-like extra piece slot in PIECE_TYPES=7 variants

        return 0.0f;
    };

    float white = deom;
    float black = 0.0f;

    for (Square sq = Square::MIN; sq <= Square::MAX; ++sq)
    {
        const auto p = pos.pieceAt(sq);
        if (p == Piece::None)
            continue;

        const float pts = piece_points(type_of(p));
        if (color_of(p) == Color::White)
            white += pts;
        else
            black += pts;
    }

    if (white > black) return 1.0f;
    if (black > white) return 0.0f;
    return 0.5f;
}

struct HalfKP {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = static_cast<int>(PieceType::MaxPiece) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color, bool horizontal_mirroring)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);
        if (horizontal_mirroring)
            ksq = mirror_square_with_files(ksq);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            const auto src_sq = horizontal_mirroring ? mirror_square_with_files(sq) : sq;
            auto p = pos.pieceAt(src_sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color, bool horizontal_mirroring)
    {
        auto [start_j, offset] = HalfKP::fill_features_sparse(e, features, values, color, horizontal_mirroring);
        int j = start_j;
        auto& pos = e.pos;
        {
            // king square factor
            auto ksq = pos.kingSquare(color);
            if (horizontal_mirroring)
                ksq = mirror_square_with_files(ksq);
            features[j] = offset + static_cast<int>(orient(color, ksq));
            values[j] = static_cast<float>(start_j);
            ++j;
        }
        offset += K_INPUTS;

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            const auto src_sq = horizontal_mirroring ? mirror_square_with_files(sq) : sq;
            auto p = pos.pieceAt(src_sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color, bool horizontal_mirroring)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);
        if (horizontal_mirroring)
            ksq = mirror_square_with_files(ksq);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            const auto src_sq = horizontal_mirroring ? mirror_square_with_files(sq) : sq;
            auto p = pos.pieceAt(src_sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color, bool horizontal_mirroring)
    {
        const auto [start_j, offset] = HalfKA::fill_features_sparse(e, features, values, color, horizontal_mirroring);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            const auto src_sq = horizontal_mirroring ? mirror_square_with_files(sq) : sq;
            auto p = pos.pieceAt(src_sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2 {
    static constexpr int NUM_KSQ = static_cast<int>(Square::KNB);
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2 - (NUM_KSQ > 1);
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - (NUM_KSQ > 1));
    static constexpr int INPUTS = NUM_PLANES * NUM_KSQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        if (NUM_PT % 2 && p_idx == NUM_PT)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static int feature_index(Color color, Square ksq, int handCount, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return handCount + p_idx * MAX_HAND_PIECES + NUM_SQ * NUM_PT + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color, bool horizontal_mirroring)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);
        if (horizontal_mirroring)
            ksq = mirror_square_with_files(ksq);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            const auto src_sq = horizontal_mirroring ? mirror_square_with_files(sq) : sq;
            auto p = pos.pieceAt(src_sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    features[j] = feature_index(color, orient_flip(color, ksq), i, make_piece(pt, c));
                    ++j;
                }

        return { j, INPUTS };
    }
};

struct HalfKAv2Factorized {
    // Factorized features
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int PIECE_INPUTS = HalfKAv2::NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - 2 * (HalfKAv2::NUM_KSQ > 1));
    static constexpr int INPUTS = HalfKAv2::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color, bool horizontal_mirroring)
    {
        const auto [start_j, offset] = HalfKAv2::fill_features_sparse(e, features, values, color, horizontal_mirroring);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            const auto src_sq = horizontal_mirroring ? mirror_square_with_files(sq) : sq;
            auto p = pos.pieceAt(src_sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    auto p_idx = static_cast<int>(pt) * 2 + (c != color);
                    features[j] = offset + i + p_idx * MAX_HAND_PIECES + HalfKAv2::NUM_SQ * NUM_PT;
                    ++j;
                }

        return { j, INPUTS };
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color, bool horizontal_mirroring)
    {
        return T::fill_features_sparse(e, features, values, color, horizontal_mirroring);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries, bool horizontal_mirroring, bool resolve_draws, float deom)
        : m_horizontal_mirroring(horizontal_mirroring), m_resolve_draws(resolve_draws), m_deom(deom)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices = new int[size];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;
    bool m_horizontal_mirroring;
    bool m_resolve_draws;
    float m_deom;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

private:
    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        static thread_local std::mt19937 mirror_gen(std::random_device{}());
        static thread_local std::bernoulli_distribution mirror_coin(0.5);
        const bool do_mirror = m_horizontal_mirroring && mirror_coin(mirror_gen);
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);

        if (m_resolve_draws && e.result == 0)
        {
            const float white_wdl = calculate_janggi_winner(e.pos, m_deom);
            // Outcome is from side-to-move perspective in training tensors.
            outcome[i] = is_white[i] > 0.5f ? white_wdl : (1.0f - white_wdl);
        }
        else
        {
            outcome[i] = (e.result + 1.0f) / 2.0f;
        }

        score[i] = e.score;
        psqt_indices[i] = (e.pos.pieceCount() - 1) * 8 / MAX_PIECES;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e, do_mirror);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e, bool do_mirror)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::White, do_mirror)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::Black, do_mirror)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(int concurrency, const char** filenames, int num_files, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate)
    {
        m_streams.reserve(num_files);
        m_stream_mutexes.reserve(num_files);
        for (int i = 0; i < num_files; ++i)
        {
            auto stream = training_data::open_sfen_input_file_parallel(concurrency, filenames[i], cyclic, skipPredicate);
            if (stream)
            {
                m_streams.emplace_back(std::move(stream));
                m_stream_mutexes.emplace_back(std::make_unique<std::mutex>());
            }
        }
    }

    virtual StorageT* next() = 0;

protected:
    std::vector<std::unique_ptr<training_data::BasicSfenInputStream>> m_streams;
    std::vector<std::unique_ptr<std::mutex>> m_stream_mutexes;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const char** filenames, int num_files, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filenames, num_files, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const char** filenames, int num_files, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate, bool horizontal_mirroring, bool resolve_draws, float deom) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filenames,
            num_files,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size),
        m_horizontal_mirroring(horizontal_mirroring),
        m_resolve_draws(resolve_draws),
        m_deom(deom)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                if (BaseType::m_streams.empty())
                    break;

                static thread_local std::mt19937 pick_stream_gen(std::random_device{}());
                std::uniform_int_distribution<std::size_t> pick_stream(0, BaseType::m_streams.size() - 1);

                // True random interleaving: choose a random stream per entry.
                for (int eidx = 0; eidx < m_batch_size; ++eidx)
                {
                    const std::size_t stream_idx = pick_stream(pick_stream_gen);
                    auto& stream = BaseType::m_streams[stream_idx];
                    auto& stream_mutex = *BaseType::m_stream_mutexes[stream_idx];

                    std::optional<TrainingDataEntry> value;
                    {
                        std::unique_lock lock(stream_mutex);
                        value = stream->next();
                    }

                    if (value.has_value())
                    {
                        entries.emplace_back(*value);
                    }
                }

                if (entries.empty())
                    break;

                auto batch = new StorageT(FeatureSet{}, entries, m_horizontal_mirroring, m_resolve_draws, m_deom);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    bool m_horizontal_mirroring;
    bool m_resolve_draws;
    float m_deom;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};


std::function<bool(const TrainingDataEntry&)> make_skip_predicate(bool filtered, int random_fen_skipping)
{
    if (filtered || random_fen_skipping)
    {
        return [
            random_fen_skipping,
            prob = double(random_fen_skipping) / (random_fen_skipping + 1),
            filtered
            ](const TrainingDataEntry& e){

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return false;
            };

            static thread_local std::mt19937 gen(std::random_device{}());
            return (random_fen_skipping && do_skip()) || (filtered && do_filter());
        };
    }

    return nullptr;
}

extern "C" {

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char** filenames, int num_files, int batch_size, bool cyclic, bool filtered, int random_fen_skipping, bool horizontal_mirroring, bool resolve_draws, float deom)
    {
        auto skipPredicate = make_skip_predicate(filtered, random_fen_skipping);

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filenames, num_files, batch_size, cyclic, skipPredicate, horizontal_mirroring, resolve_draws, deom);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filenames, num_files, batch_size, cyclic, skipPredicate, horizontal_mirroring, resolve_draws, deom);
        }
        else if (feature_set == "HalfKA")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filenames, num_files, batch_size, cyclic, skipPredicate, horizontal_mirroring, resolve_draws, deom);
        }
        else if (feature_set == "HalfKA^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filenames, num_files, batch_size, cyclic, skipPredicate, horizontal_mirroring, resolve_draws, deom);
        }
        else if (feature_set == "HalfKAv2")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2>, SparseBatch>(concurrency, filenames, num_files, batch_size, cyclic, skipPredicate, horizontal_mirroring, resolve_draws, deom);
        }
        else if (feature_set == "HalfKAv2^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2Factorized>, SparseBatch>(concurrency, filenames, num_files, batch_size, cyclic, skipPredicate, horizontal_mirroring, resolve_draws, deom);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    const char* files[] = {"10m_d3_q_2.bin"};
    auto stream = create_sparse_batch_stream("HalfKP", 4, files, 1, 8192, true, false, 0, false, false, 1.5f);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/
