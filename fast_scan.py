"""Fast minimax / Absurdle-greedy scan using a precomputed compare() cache.

Usage:
    python fast_scan.py build                                # build/verify compare_cache.npz
    python fast_scan.py a  [N] [--limit SEC]                 # Absurdle greedy adversary, first N answers (default all)
    python fast_scan.py m  [N] [--limit SEC] [--workers W]   # true minimax adversary, resumable, W processes

While running, the top-level loop prints progress and an ETA for the current depth k.
--limit stops after SEC seconds.

Minimax mode saves its state under scan_state/ and resumes from it on the next run:
  - progress_m_<N>.json : current k and the first guesses already proven to fail at this k
  - memo_*.pkl          : solved / unsolvable bounds of sub-states (shared by all N and all workers)
Progress is written every ~30 s and memo every ~5 min, so a stop (Ctrl+C, --limit, crash)
loses at most that much work. Delete scan_state/ to start from scratch.
"""
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

from functions import load_wordlist, compare, ternary, generate

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "compare_cache.npz")

words = load_wordlist(os.path.join(HERE, "valid-wordle-words.txt"))
answers = load_wordlist(os.path.join(HERE, "valid-wordle-answers.txt"))
colorings = generate()
G = len(words)


# ---------------------------------------------------------------- cache
def build_patterns():
    """pat[g, a] == ternary(compare(words[g], answers[a])) as uint8."""
    W = np.array([[ord(c) - 65 for c in w] for w in words], np.int8)
    A = np.array([[ord(c) - 65 for c in w] for w in answers], np.int8)
    pat = np.zeros((G, len(A)), np.uint8)
    pw = np.array([81, 27, 9, 3, 1], np.uint8)
    onehot = np.eye(26, dtype=np.int8)[A]                   # (N, 5, 26)
    for g in range(G):
        green = A == W[g]                                    # (N, 5)
        cnt = (onehot * (~green)[:, :, None]).sum(1)         # unmatched letters per answer
        res = green.astype(np.uint8) * 2
        for i in range(5):
            y = (~green[:, i]) & (cnt[:, W[g, i]] > 0)
            res[y, i] = 1
            cnt[y, W[g, i]] -= 1
        pat[g] = res @ pw
    return pat


def verify(pat, samples=20000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        g, a = rng.integers(G), rng.integers(len(answers))
        assert pat[g, a] == ternary(compare(words[g], answers[a])), (words[g], answers[a])


def load_patterns():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if list(z["words"]) == words and list(z["answers"]) == answers:
            return z["pat"]
        print("word lists changed, rebuilding cache", flush=True)
    t = time.perf_counter()
    pat = build_patterns()
    verify(pat)
    np.savez(CACHE, pat=pat, words=np.array(words), answers=np.array(answers))
    print(f"built {CACHE} in {time.perf_counter() - t:.1f}s", flush=True)
    return pat


pat = load_patterns()

# Absurdle tie-break: bigger basket, then fewer greens, fewer yellows, smaller id.
rank = np.empty(243, np.int64)
rank[sorted(range(243), key=lambda i: (colorings[i].count(2), colorings[i].count(1), i))] = np.arange(243)
TIE = 242 - rank
ROWS = (np.arange(G, dtype=np.int32) * 243)[:, None]
CAP = [0, 1] + [243 ** j for j in range(1, 10)]  # max set size solvable in k guesses (loose)


def counts_of(P):
    """(G, 243) basket sizes for every guess."""
    return np.bincount((ROWS + P).ravel(), minlength=G * 243).reshape(G, 243)


def any_all_singletons(P):
    """Is there a guess that puts every word in its own basket?"""
    s = np.sort(P, axis=1)
    return bool(np.all(s[:, 1:] != s[:, :-1], axis=1).any())


# ---------------------------------------------------------------- progress
class Progress:
    def __init__(self, limit):
        self.limit, self.t0, self.last = limit, time.perf_counter(), 0.0

    def tick(self, k, i, total):
        now = time.perf_counter()
        el = now - self.t0
        if now - self.last >= 5 or i == total:
            self.last = now
            rate = el / i if i else float("nan")
            print(f"  k={k}: {i}/{total} top-level candidates, {el:.0f}s elapsed, "
                  f"worst-case ETA for this k ~ {rate * (total - i):.0f}s", flush=True)
        if self.limit and el > self.limit:
            raise TimeoutError


# ---------------------------------------------------------------- Absurdle greedy
a_lo, a_hi = {}, {}


def absurdle_le(S, k, prog=None):
    """True iff greedy Absurdle can be beaten within k guesses from answer set S."""
    n = len(S)
    if n <= 1:
        return k >= n
    if k <= 1 or n > CAP[k]:
        return False
    key = S.tobytes()
    if a_hi.get(key, 99) <= k:
        return True
    if a_lo.get(key, 0) >= k:
        return False
    P = pat[:, S].astype(np.int32)
    cnt = counts_of(P)
    chosen = np.where(cnt > 0, cnt * 243 + TIE, -1).argmax(1)
    size = cnt[np.arange(G), chosen]
    g_ok = np.nonzero(size < n)[0]
    res = False
    if (size[g_ok] == 1).any():
        res = True
    elif k > 2:
        g_ok = g_ok[size[g_ok] <= CAP[k - 1]]
        masks = P[g_ok] == chosen[g_ok, None]
        _, first = np.unique(np.packbits(masks, axis=1), axis=0, return_index=True)
        order = first[np.argsort(size[g_ok][first], kind="stable")]
        for i, j in enumerate(order):
            if prog:
                prog.tick(k, i, len(order))
            if absurdle_le(S[masks[j]], k - 1):
                res = True
                break
    (a_hi if res else a_lo)[key] = k if res else max(a_lo.get(key, 0), k)
    return res


# ---------------------------------------------------------------- true minimax
m_lo, m_hi = {}, {}


def solve(S, k, prog=None):
    """True iff every answer in S can be found within k guesses vs. any adversary."""
    n = len(S)
    if n <= 1:
        return k >= n
    if k <= 1 or n > CAP[k]:
        return False
    key = hashlib.blake2b(S.tobytes(), digest_size=16).digest()   # 16-byte key keeps memo small
    if m_hi.get(key, 99) <= k:
        return True
    if m_lo.get(key, 0) >= k:
        return False
    P = pat[:, S].astype(np.int32)
    if k == 2 or n <= 64:
        res = any_all_singletons(P)
        if res or k == 2:
            (m_hi if res else m_lo)[key] = k
            return res
    cnt = counts_of(P)
    mx = cnt.max(1)
    g_ok = np.nonzero((mx < n) & (mx <= CAP[k - 1]))[0]
    _, first = np.unique(P[g_ok], axis=0, return_index=True)
    cand = g_ok[first][np.argsort(mx[g_ok][first], kind="stable")]
    res = False
    for i, g in enumerate(cand):
        if prog:
            prog.tick(k, i, len(cand))
        row, c = P[g], cnt[g]
        good = True
        for col in np.argsort(-c)[: np.count_nonzero(c)]:   # biggest basket first
            if not solve(S[row == col], k - 1):
                good = False
                break
        if good:
            res = True
            break
    if res:
        m_hi[key] = k
    else:
        m_lo[key] = max(m_lo.get(key, 0), k)
    return res


def search(fn, S, limit):
    prog = Progress(limit)
    k = 1
    while True:
        tk = time.perf_counter()
        ok = fn(S, k, prog)
        print(f"k={k}: {'solvable' if ok else 'not solvable'} ({time.perf_counter() - tk:.1f}s)", flush=True)
        if ok:
            return k
        k += 1


# ---------------------------------------------------------------- resumable parallel minimax
STATE_DIR = os.path.join(HERE, "scan_state")
LISTS_HASH = hashlib.sha1(("\n".join(words) + "|" + "\n".join(answers)).encode()).hexdigest()
MEMO_EVERY = float(os.environ.get("FAST_SCAN_MEMO_EVERY", 300))  # seconds between worker memo dumps
PROGRESS_EVERY = 30   # seconds between progress file writes
MEMO_FMT = 2          # bump when the memo key format changes (2 = blake2b-16 of the index array)


def atomic_write(path, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def merge_memo_files():
    """Merge every memo_*.pkl into memo_merged.pkl (bounds stay valid for any N / k)."""
    for p in glob.glob(os.path.join(STATE_DIR, "*.tmp")):    # leftovers of interrupted writes
        os.remove(p)
    paths = glob.glob(os.path.join(STATE_DIR, "memo_*.pkl"))
    lo, hi = {}, {}
    for p in paths:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception as e:                       # half-written / corrupt: skip it
            print(f"  skip unreadable {os.path.basename(p)}: {e}", flush=True)
            continue
        if d.get("hash") != LISTS_HASH or d.get("fmt") != MEMO_FMT:
            continue
        for key, v in d["lo"].items():
            if v > lo.get(key, 0):
                lo[key] = v
        for key, v in d["hi"].items():
            if v < hi.get(key, 99):
                hi[key] = v
    merged = os.path.join(STATE_DIR, "memo_merged.pkl")
    atomic_write(merged, pickle.dumps({"hash": LISTS_HASH, "fmt": MEMO_FMT, "lo": lo, "hi": hi}, pickle.HIGHEST_PROTOCOL))
    for p in paths:
        if os.path.abspath(p) != os.path.abspath(merged):
            os.remove(p)
    return merged, len(lo) + len(hi)


_last_dump = 0.0


def dump_memo(force=False):
    global _last_dump
    now = time.perf_counter()
    if force or now - _last_dump >= MEMO_EVERY:
        _last_dump = now
        path = os.path.join(STATE_DIR, f"memo_w{os.getpid()}.pkl")
        atomic_write(path, pickle.dumps({"hash": LISTS_HASH, "fmt": MEMO_FMT, "lo": m_lo, "hi": m_hi}, pickle.HIGHEST_PROTOCOL))


_barrier = None


def flush_task(_):
    """Each worker takes exactly one of these (barrier) and dumps its memo."""
    dump_memo(force=True)
    _barrier.wait(timeout=120)
    return os.getpid()


def flush_all(pool, workers):
    try:
        pool.map(flush_task, range(workers), chunksize=1)
    except Exception as e:
        print(f"  memo flush incomplete: {e}", flush=True)


def init_worker(merged_path, barrier=None):
    global _last_dump, _barrier
    _barrier = barrier
    if merged_path and os.path.exists(merged_path):
        with open(merged_path, "rb") as f:
            d = pickle.load(f)
        m_lo.update(d["lo"])
        m_hi.update(d["hi"])
    _last_dump = time.perf_counter()


def check_first_guess(task):
    """Worker: can first guess g on answers[:n] be finished within k guesses in total?"""
    n, k, g = task
    S = np.arange(n, dtype=np.int32)
    row = pat[g, S]
    c = np.bincount(row, minlength=243)
    ok = True
    for col in np.argsort(-c)[: np.count_nonzero(c)]:   # biggest basket first
        if not solve(S[row == col], k - 1):
            ok = False
            break
    dump_memo()
    return g, ok


def root_candidates(S, k):
    """Same root logic as solve(): returns (decided_result or None, candidate first guesses in order)."""
    n = len(S)
    if n <= 1:
        return k >= n, []
    if k <= 1 or n > CAP[k]:
        return False, []
    P = pat[:, S].astype(np.int32)
    if any_all_singletons(P):
        return True, []
    if k == 2:
        return False, []
    cnt = counts_of(P)
    mx = cnt.max(1)
    g_ok = np.nonzero((mx < n) & (mx <= CAP[k - 1]))[0]
    _, first = np.unique(P[g_ok], axis=0, return_index=True)
    cand = g_ok[first][np.argsort(mx[g_ok][first], kind="stable")]
    return None, [int(g) for g in cand]


def minimax_resumable(n, workers, limit):
    os.makedirs(STATE_DIR, exist_ok=True)
    prog_path = os.path.join(STATE_DIR, f"progress_m_{n}.json")
    state = {"hash": LISTS_HASH, "n": n, "k": 1, "done": [], "result": None}
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            old = json.load(f)
        if old.get("hash") == LISTS_HASH and old.get("n") == n:
            state = old
            print(f"resuming: k={state['k']}, {len(state['done'])} first guesses already ruled out", flush=True)
        else:
            print("progress file is for other word lists, starting over", flush=True)
    if state["result"] is not None:
        print(f"already finished: m n={n}: {state['result']} guesses (best first guess {state.get('best_guess')})")
        return state["result"]

    def save_progress():
        atomic_write(prog_path, json.dumps(state).encode())

    merged, entries = merge_memo_files()
    print(f"memo: {entries} saved bounds loaded", flush=True)
    S = np.arange(n, dtype=np.int32)
    t0 = time.perf_counter()

    while True:
        k = state["k"]
        decided, cand = root_candidates(S, k)
        done = set(state["done"])
        todo = [g for g in cand if g not in done]
        found = None
        if decided is None and todo:
            print(f"k={k}: {len(cand)} first guesses, {len(todo)} left, {workers} workers", flush=True)
            tk, last_save, last_print, finished = time.perf_counter(), 0.0, 0.0, 0
            pool = mp.Pool(workers, initializer=init_worker, initargs=(merged, mp.Barrier(workers)))
            try:
                for g, ok in pool.imap_unordered(check_first_guess, [(n, k, g) for g in todo]):
                    finished += 1
                    now = time.perf_counter()
                    if ok:
                        found = g
                        break
                    state["done"].append(g)
                    if now - last_save >= PROGRESS_EVERY:
                        save_progress()
                        last_save = now
                    if now - last_print >= 10:
                        last_print = now
                        el = now - tk
                        eta = el / finished * (len(todo) - finished)
                        print(f"  k={k}: {len(state['done'])}/{len(cand)} ruled out, "
                              f"{el:.0f}s this run, worst-case ETA ~ {eta / 3600:.2f} h", flush=True)
                    if limit and now - t0 > limit:
                        raise TimeoutError
            except (KeyboardInterrupt, TimeoutError) as e:
                pool.terminate()
                pool.join()
                save_progress()
                print(f"stopped ({type(e).__name__}); progress saved, run the same command to resume", flush=True)
                return None
            if found is None:            # queue drained: let every worker save its memo
                flush_all(pool, workers)
            pool.terminate()
            pool.join()
            merged, entries = merge_memo_files()
        ok = decided if decided is not None else found is not None
        print(f"k={k}: {'solvable' if ok else 'not solvable'}"
              + (f" (first guess {words[found]})" if found is not None else ""), flush=True)
        if ok:
            state["result"] = k
            state["best_guess"] = words[found] if found is not None else None
            save_progress()
            print(f"m n={n}: {k} guesses, total {time.perf_counter() - t0:.1f}s this run", flush=True)
            return k
        state["k"], state["done"] = k + 1, []
        save_progress()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "build":
        verify(pat)
        print(f"cache OK: {pat.shape[0]} guesses x {pat.shape[1]} answers", flush=True)
        sys.exit()
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = float(args[i + 1])
        del args[i:i + 2]
    workers = os.cpu_count() or 1
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    mode = args[0]
    n = int(args[1]) if len(args) > 1 else len(answers)
    if mode == "m":
        minimax_resumable(n, workers, limit)
        sys.exit()
    S = np.arange(n, dtype=np.int32)
    t = time.perf_counter()
    try:
        r = search(absurdle_le if mode == "a" else solve, S, limit)
        print(f"{mode} n={n}: {r} guesses, total {time.perf_counter() - t:.1f}s", flush=True)
    except TimeoutError:
        print(f"stopped after --limit {limit}s", flush=True)
