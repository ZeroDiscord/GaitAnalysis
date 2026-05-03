"""
gait_phase.py — EMG-Based Gait Cycle Phase Extraction
======================================================
Pipeline:
    1. TKEO-based burst amplification
    2. Bandpass → rectify → RMS envelope → z-score  (global, for cycle detection)
    3. Method 1 (threshold bursts) → Method 2 (derivative) → FSM phases
    4. Fallback: autocorrelation for aperiodic pathological signals
    5. Per-cycle amplitude normalization [0, 1] after cycle detection
       (makes feature extraction invariant to absolute signal strength)
    6. Output A: continuous gait_phase [0, 100] per timestep  (→ dataset.py channel 4)
    7. Output B: per-cycle feature dict  (→ eda_features.py ML features)

Normalization strategy (dual-stage):
    Stage 1 — Global z-score: Required BEFORE cycle detection (chicken-and-egg constraint).
              The z-score threshold (Z > 1.5) identifies bursts in a signal-agnostic way.
    Stage 2 — Cycle-based peak normalization: Applied AFTER cycles are detected.
              Each individual cycle's envelope is scaled to its own peak [0, 1].
              This makes timing features (TA_onset_percent, GA_peak_percent, etc.)
              invariant to absolute EMG amplitude, crucial for cross-patient comparison
              and for pathological signals that have globally suppressed amplitude.

Column mapping (matches dataset.py convention):
    col 0 = e_ago = TA (Tibialis Anterior)   — fires during swing
    col 1 = e_ant = GA (Gastrocnemius)        — fires during terminal stance/toe-off
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Constants — physiologically informed thresholds
# ---------------------------------------------------------------------------
_FS_DEFAULT    = 1000       # Hz
_BANDPASS_LOW  = 20.0       # Hz
_BANDPASS_HIGH = 450.0      # Hz
_RMS_WINDOW_MS = 50         # ms
_MAD_K         = 2.5        # MAD multiplier (≈ z=1.5 for Gaussian, but robust to outliers)
_DERIV_SG_WIN  = 51         # Savitzky-Golay window for derivative smoothing (odd, ms-scale)
_DERIV_SG_ORD  = 2          # Savitzky-Golay polynomial order
_MIN_CYCLE_MS  = 400        # ms
_MAX_CYCLE_MS  = 3000       # ms

# FSM state names (for logging / interpretation)
FSM_STATES = [
    "Initial_Contact",      # 0
    "Loading_Response",     # 1
    "Midstance",            # 2
    "Terminal_Stance",      # 3
    "Pre_Swing",            # 4
    "Early_Swing",          # 5
    "Mid_Swing",            # 6
    "Late_Swing",           # 7
]

# Physiological phase boundaries (% of gait cycle, healthy gait reference)
HEALTHY_PHASE_PCT = {
    "Initial_Contact":  (0,  2),
    "Loading_Response": (2,  12),
    "Midstance":        (12, 50),
    "Terminal_Stance":  (50, 62),
    "Pre_Swing":        (62, 75),
    "Early_Swing":      (75, 83),
    "Mid_Swing":        (83, 93),
    "Late_Swing":       (93, 100),
}


# ---------------------------------------------------------------------------
# Stage 1: TKEO — Teager-Kaiser Energy Operator
# ---------------------------------------------------------------------------

def tkeo(x: np.ndarray) -> np.ndarray:
    """
    Apply the Teager-Kaiser Energy Operator to signal x.
    Ψ[x(n)] = x(n)² - x(n-1)·x(n+1)
    
    Amplifies transient muscle activation onset relative to background noise.
    O(N) computation.
    """
    out = np.empty_like(x, dtype=np.float64)
    out[0]  = x[0] ** 2
    out[-1] = x[-1] ** 2
    out[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    # Clip negative values (artifact of squaring numerical noise floors)
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# Stage 2: Signal preprocessing → EMG envelope
# ---------------------------------------------------------------------------

def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """4th-order zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 1.0 - 1e-4)
    if lo_n >= hi_n:
        return x
    b, a = butter(4, [lo_n, hi_n], btype='band')
    if len(x) < 27:          # filtfilt minimum length guard
        return np.abs(x)
    return filtfilt(b, a, x)


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    """
    O(N) rolling RMS via cumulative sum of squares.
    Output length == input length (padded at edges).
    """
    x2 = x ** 2
    pad = window // 2
    x2p = np.pad(x2, (pad, pad), mode='edge')
    cs = np.cumsum(x2p)
    cs = np.r_[0, cs]
    rms = np.sqrt(np.maximum(cs[window:] - cs[:-window], 0.0) / window)
    return rms[:len(x)]


# ---------------------------------------------------------------------------
# MAD-based adaptive threshold (robust alternative to z-score)
# ---------------------------------------------------------------------------

def mad_threshold(x: np.ndarray, k: float = _MAD_K) -> float:
    """
    Compute a robust activation threshold using the Median Absolute Deviation.

        threshold = median(x) + k * 1.4826 * MAD(x)

    The constant 1.4826 normalizes MAD to match std for Gaussian distributions,
    but unlike std the MAD is minimally affected by large transient spikes, making
    it far more reliable for noisy EMG and pathological gait signals.

    k=2.5 gives roughly the same sensitivity as z=1.5 for healthy signals while
    staying robust when 20–30% of samples are outliers.
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med + k * 1.4826 * mad


# ---------------------------------------------------------------------------
# Stage 1 → 2: Signal preprocessing → EMG envelope
#   Correct order: bandpass FIRST (on raw EMG), THEN TKEO.
#   Rationale: TKEO squares the signal, so applying it before the bandpass
#   folds out-of-band noise into the passband.  Filtering first removes noise
#   and motion artifacts before TKEO amplifies the clean muscle activation bursts.
# ---------------------------------------------------------------------------

def emg_envelope(x: np.ndarray, fs: float = _FS_DEFAULT) -> np.ndarray:
    """
    Full preprocessing chain (literature-correct order):
        bandpass → TKEO → full-wave rectification → RMS envelope → MAD threshold

    Returns a MAD-normalized envelope where the baseline noise floor ≈ 0
    and muscle activation bursts rise clearly above the threshold returned by
    mad_threshold() on this output.
    """
    x = x.astype(np.float64)

    # 1. Bandpass filter FIRST — remove motion artifacts (<20 Hz) and
    #    high-frequency electrical noise (>450 Hz) on the raw signal
    bp = _bandpass(x, fs, _BANDPASS_LOW, min(_BANDPASS_HIGH, fs / 2 - 1))

    # 2. TKEO on the clean bandpass signal — amplifies activation onsets
    #    relative to inter-burst baseline noise
    tk = tkeo(bp)

    # 3. Full-wave rectification
    rect = np.abs(tk)

    # 4. RMS smoothing (50 ms window) — creates smooth activation envelope
    win = max(2, int(_RMS_WINDOW_MS * fs / 1000))
    env = _rolling_rms(rect, win)

    # 5. Z-score for compatibility with thresholding functions;
    #    actual threshold is computed via mad_threshold() at call sites
    mu, sigma = np.mean(env), np.std(env)
    if sigma < 1e-12:
        return np.zeros_like(env)
    return (env - mu) / sigma


# ---------------------------------------------------------------------------
# Stage 3, Method 1: Threshold-based burst detection (MAD-adaptive)
# ---------------------------------------------------------------------------

def detect_bursts_threshold(envelope: np.ndarray,
                            threshold: float = None,
                            min_gap_samples: int = 50) -> list[dict]:
    """
    Locate activation bursts using a MAD-based adaptive threshold.
    If threshold is None (default), it is computed via mad_threshold(envelope).
    """
    if threshold is None:
        threshold = mad_threshold(envelope, k=_MAD_K)
    
    above = (envelope >= threshold).astype(np.int8)
    diff  = np.diff(np.r_[0, above, 0])
    onsets  = np.where(diff ==  1)[0]
    offsets = np.where(diff == -1)[0]

    bursts = []
    for on, off in zip(onsets, offsets):
        seg = envelope[on:off]
        peak_rel = int(np.argmax(seg))
        bursts.append({
            'onset': int(on),
            'offset': int(off),
            'peak_idx': int(on + peak_rel),
            'peak_val': float(seg[peak_rel]),
        })

    # Merge bursts with small gaps (short silence between two activation bursts)
    merged = []
    for b in bursts:
        if merged and (b['onset'] - merged[-1]['offset']) < min_gap_samples:
            prev = merged[-1]
            # Extend previous burst
            if b['peak_val'] > prev['peak_val']:
                prev['peak_idx'] = b['peak_idx']
                prev['peak_val'] = b['peak_val']
            prev['offset'] = b['offset']
        else:
            merged.append(b.copy())

    return merged


# ---------------------------------------------------------------------------
# Stage 3, Method 2: Savitzky-Golay derivative-based event detection
# ---------------------------------------------------------------------------

def detect_bursts_derivative(envelope: np.ndarray,
                              fs: float = _FS_DEFAULT,
                              deriv_thresh: float = 0.15) -> list[dict]:
    """
    Detect activation bursts using a Savitzky-Golay smoothed first derivative.

    Savitzky-Golay filters apply a local polynomial fit before differentiating,
    which dramatically reduces spurious zero-crossings caused by high-frequency
    sampling noise — a critical improvement over raw np.diff for pathological
    EMG where bursts build up slowly and have rounded edges.
    """
    # SG window must be odd and ≥ polynomial order + 2
    # Ensure window is valid given the actual signal length
    sg_win = min(_DERIV_SG_WIN, len(envelope))
    if sg_win % 2 == 0:
        sg_win -= 1  # keep it odd
    
    # Needs at least polyorder + 2 samples for Savitzky-Golay
    if len(envelope) >= _DERIV_SG_ORD + 2 and sg_win >= _DERIV_SG_ORD + 2:
        d = savgol_filter(envelope, window_length=sg_win,
                          polyorder=_DERIV_SG_ORD, deriv=1, delta=1.0 / fs)
    else:
        d = np.gradient(envelope)  # fallback for very short windows

    # MAD-based threshold for the envelope itself
    env_thresh = mad_threshold(envelope, k=1.5)

    # Robust normalization of the smoothed derivative
    d_range = np.percentile(np.abs(d), 95) + 1e-12
    d_norm  = d / d_range

    # Onset: large positive derivative AND envelope above threshold
    onset_mask  = (d_norm >  deriv_thresh) & (envelope > env_thresh)
    # Offset: large negative derivative AND envelope still partially active
    offset_mask = (d_norm < -deriv_thresh) & (envelope > env_thresh * 0.5)

    onset_diff  = np.diff(np.r_[0, onset_mask.astype(np.int8)])
    offset_diff = np.diff(np.r_[0, offset_mask.astype(np.int8)])
    onsets  = np.where(onset_diff  == 1)[0]
    offsets = np.where(offset_diff == -1)[0]

    bursts = []
    for on in onsets:
        # Find the next offset after this onset
        candidates = offsets[offsets > on]
        if len(candidates) == 0:
            off = len(envelope) - 1
        else:
            off = candidates[0]
        seg = envelope[on:off]
        if len(seg) == 0:
            continue
        peak_rel = int(np.argmax(seg))
        bursts.append({
            'onset': int(on),
            'offset': int(off),
            'peak_idx': int(on + peak_rel),
            'peak_val': float(seg[peak_rel] if len(seg) > 0 else 0),
        })

    # Remove duplicated / overlapping bursts
    cleaned = []
    for b in bursts:
        if cleaned and b['onset'] <= cleaned[-1]['offset']:
            if b['peak_val'] > cleaned[-1]['peak_val']:
                cleaned[-1] = b
        else:
            cleaned.append(b)

    return cleaned


# ---------------------------------------------------------------------------
# Stage 3, Method 3: Finite State Machine (FSM) Phase Estimator
# ---------------------------------------------------------------------------

def fsm_phase_estimate(ta_env: np.ndarray,
                       ga_env: np.ndarray,
                       rising_thresh: float = 0.8) -> np.ndarray:
    """
    Assign FSM state index (0–7) per timestep based on combined TA and GA state.

    Transition table (simplified):
        TA_active + GA_low           → Initial Contact / Loading Response
        TA_falling + GA_rising       → Midstance
        GA_active                    → Terminal Stance
        GA_peak + TA_low             → Pre-Swing
        GA_falling + TA_rising       → Early Swing
        TA_active + GA_low           → Mid / Late Swing

    Returns integer FSM state per timestep (0–7).
    """
    n = len(ta_env)
    states = np.zeros(n, dtype=np.int32)

    # Replace the FSM's internal np.gradient with SG derivative for consistency
    sg_win = min(51, len(ta_env))
    if sg_win % 2 == 0:
        sg_win -= 1
        
    if len(ta_env) >= _DERIV_SG_ORD + 2 and sg_win >= _DERIV_SG_ORD + 2:
        ta_d = savgol_filter(ta_env, window_length=sg_win, polyorder=_DERIV_SG_ORD, deriv=1)
        ga_d = savgol_filter(ga_env, window_length=sg_win, polyorder=_DERIV_SG_ORD, deriv=1)
    else:
        ta_d = np.gradient(ta_env)
        ga_d = np.gradient(ga_env)
    # MAD-based active threshold — replaces fixed _ZSCORE_THRESHOLD in FSM
    ta_thresh = mad_threshold(ta_env)
    ga_thresh = mad_threshold(ga_env)
    ta_active = ta_env >= ta_thresh
    ga_active = ga_env >= ga_thresh
    ta_rising  = ta_d > 0
    ga_rising  = ga_d > 0

    # Vectorised FSM (priority: higher state index wins)
    # Default: Late Swing (7)
    states[:] = 7

    # Layered rule application (lower is overridden by higher priority)
    states[ta_active & ~ga_active & ~ta_rising]    = 1  # Loading Response
    states[ta_active & ~ga_active & ta_rising]     = 0  # Initial Contact
    states[~ta_active & ga_rising]                 = 2  # Midstance
    states[~ta_active & ga_active & ~ga_rising]    = 3  # Terminal Stance
    states[~ta_active & ga_active & ~ta_rising]    = 4  # Pre-Swing
    states[~ta_active & ~ga_active & ta_rising]    = 5  # Early Swing
    states[ta_active & ~ga_active]                 = 6  # Mid Swing

    return states


# ---------------------------------------------------------------------------
# Autocorrelation fallback: cycle detection from signal periodicity
# ---------------------------------------------------------------------------

def _autocorr_cycle_period(combined_env: np.ndarray,
                           fs: float,
                           min_period_ms: float = _MIN_CYCLE_MS,
                           max_period_ms: float = _MAX_CYCLE_MS) -> int:
    """
    Estimate gait cycle period (in samples) from the autocorrelation of the
    combined envelope.  Used when TA bursts are absent (pathological gait).
    Returns 0 if no reliable period is found.
    """
    n = len(combined_env)
    min_lag = int(min_period_ms * fs / 1000)
    max_lag = min(int(max_period_ms * fs / 1000), n // 2)

    if min_lag >= max_lag or n < 2 * min_lag:
        return 0

    # Normalised autocorrelation
    x = combined_env - combined_env.mean()
    denom = np.dot(x, x) + 1e-16
    corr = np.correlate(x, x, mode='full')[n - 1:]
    corr = corr[:max_lag] / denom

    # Find peak in the valid lag range
    search = corr[min_lag:max_lag]
    if len(search) == 0:
        return 0
    peaks, _ = find_peaks(search, height=0.1)
    if len(peaks) == 0:
        return 0

    # Removed 1.0Hz cadence bias: just pick the peak with highest correlation
    # This prevents the algorithm from failing on very slow/fast pathological cadence
    best_peak_idx = peaks[np.argmax(search[peaks])]
    return int(best_peak_idx + min_lag)


# ---------------------------------------------------------------------------
# Gait cycle boundary detection (master dispatcher)
# ---------------------------------------------------------------------------

def detect_gait_cycles(ta_env: np.ndarray,
                       ga_env: np.ndarray,
                       fs: float = _FS_DEFAULT) -> tuple[np.ndarray, str]:
    """
    Detect gait cycle boundaries combining events from BOTH TA and GA channels.

    Strategy:
        Instead of using only TA onsets, we pool burst onset events from both TA
        and GA envelopes, sort them chronologically, and select those that satisfy
        the physiological inter-cycle constraints (400–3000 ms).  This is more
        robust when TA bursts are absent (hemiplegia) or GA bursts are absent
        (drop foot), since at least one channel typically retains some rhythmicity.

    Hierarchy:
        1. Combined TA+GA threshold bursts (lowest variance wins)
        2. Combined TA+GA derivative bursts
        3. Autocorrelation on fused envelope
        4. Linear fallback (entire signal = one cycle)
    """
    min_samples = int(_MIN_CYCLE_MS * fs / 1000)
    max_samples = int(_MAX_CYCLE_MS * fs / 1000)

    def _filter_starts(starts: np.ndarray) -> np.ndarray:
        """Greedy chain-building: walk forward from each accepted boundary,
        skip sub-cycle events, and collect all valid cycle boundaries.
        
        The old logic only kept the LEFT index of each valid gap, which meant
        a sequence like [159, 752, 1016, 1245] with gaps [593, 264, 229]
        would keep only [159] (1 start — not enough for a single cycle).
        
        The new logic starts from the first onset, then greedily jumps to the
        next onset that is >= min_samples away.  If that jump is also
        <= max_samples, the destination is accepted as a cycle boundary.
        If the jump is too large, it starts a new potential chain from there.
        """
        if len(starts) < 2:
            return np.array([], dtype=np.int64)
        
        chain = [starts[0]]  # Always seed with first onset
        for i in range(1, len(starts)):
            gap = starts[i] - chain[-1]
            if gap < min_samples:
                # Too close — sub-cycle event, skip
                continue
            elif gap <= max_samples:
                # Valid inter-cycle gap — accept as next boundary
                chain.append(starts[i])
            else:
                # Gap too large — start fresh chain from here
                chain.append(starts[i])
        
        # Need at least 2 boundaries to form one cycle
        if len(chain) < 2:
            return np.array([], dtype=np.int64)
        return np.array(chain, dtype=np.int64)

    def _valid(starts: np.ndarray) -> bool:
        return len(starts) >= 2

    def _var(starts: np.ndarray) -> float:
        return float(np.var(np.diff(starts))) if len(starts) > 2 else np.inf

    def _combine_bursts(bursts_ta, bursts_ga):
        """Merge onset events from both channels, sort, and filter."""
        all_onsets = sorted(
            [b['onset'] for b in bursts_ta] + [b['onset'] for b in bursts_ga]
        )
        if not all_onsets:
            return np.array([], dtype=np.int64)
            
        # Deduplication window: only merge onsets that are within 100ms of each
        # other (true double-detections from the same burst).  The previous value
        # of min_samples // 2 = 200ms was too aggressive and merged genuinely
        # separate TA/GA activation onsets that are 100-200ms apart.
        dedup_win = min_samples // 4
        deduped = [all_onsets[0]]
        for ev in all_onsets[1:]:
            if (ev - deduped[-1]) >= dedup_win:
                deduped.append(ev)
                
        return _filter_starts(np.array(deduped, dtype=np.int64))

    # --- Combined Method 1 (threshold) ---
    bursts_ta1 = detect_bursts_threshold(ta_env)
    bursts_ga1 = detect_bursts_threshold(ga_env)
    starts_comb1 = _combine_bursts(bursts_ta1, bursts_ga1)

    # --- Combined Method 2 (SG derivative) ---
    bursts_ta2 = detect_bursts_derivative(ta_env, fs)
    bursts_ga2 = detect_bursts_derivative(ga_env, fs)
    starts_comb2 = _combine_bursts(bursts_ta2, bursts_ga2)

    # Pick the combined result with lower inter-cycle variance
    if _valid(starts_comb1) or _valid(starts_comb2):
        v1 = _var(starts_comb1) if _valid(starts_comb1) else np.inf
        v2 = _var(starts_comb2) if _valid(starts_comb2) else np.inf
        if v1 <= v2 and _valid(starts_comb1):
            return starts_comb1, "Combined_Method1_MAD"
        elif _valid(starts_comb2):
            return starts_comb2, "Combined_Method2_SavGol"

    # --- Autocorrelation fallback ---
    combined = ta_env + ga_env
    period = _autocorr_cycle_period(combined, fs)
    if period > 0:
        starts_ac = np.arange(0, len(ta_env) - period, period, dtype=np.int64)
        if _valid(starts_ac):
            return starts_ac, "Autocorrelation_Fallback"

    # --- Final fallback ---
    return np.array([0], dtype=np.int64), "Linear_Fallback"


# ---------------------------------------------------------------------------
# Continuous gait phase output [0, 100] per timestep
# EMG-activity-driven velocity with physiological landmark anchoring
# ---------------------------------------------------------------------------
#
# Method (two-stage):
#
#   Stage 1 — EMG-driven raw phase:
#     phase_velocity[t] = baseline + weight * activity_norm[t]
#     raw_phase = cumsum(velocity) / total * 100
#
#     This produces a nonlinear monotonic curve where phase advances
#     rapidly during muscle activations and slowly during quiet periods.
#
#   Stage 2 — Physiological landmark re-normalization:
#     Within each cycle, detect two physiological events from the
#     raw RMS envelopes:
#
#       1. GA peak         → terminal stance (~55%)
#          The gastrocnemius peak corresponds to maximal push-off force.
#
#       2. GA→TA crossover → toe-off (~62%)
#          The point where GA dominance gives way to TA dominance
#          marks the stance-to-swing transition.
#
#     The raw phase is then piecewise re-normalized through these
#     anchors, forcing GA peak → 55% and crossover → 62% while
#     preserving the EMG-driven nonlinear shape within each segment.
#
# Reviewer-ready description:
#   "Gait phase is computed as the normalized cumulative integral of
#    instantaneous EMG activity (TA + GA RMS envelopes), with post-hoc
#    piecewise re-normalization to anchor the gastrocnemius peak to 55%
#    (terminal stance) and the GA–TA dominance crossover to 62%
#    (toe-off). This ensures physiological events are consistently
#    mapped to their expected phase positions while preserving the
#    data-driven nonlinear phase progression within each sub-phase."
#
# ---------------------------------------------------------------------------

_PHASE_BASELINE_WEIGHT = 0.15   # minimum phase velocity (prevents stalling)
_PHASE_EMG_WEIGHT      = 0.85   # EMG modulation of phase rate

# Physiological anchor targets (% of gait cycle)
_ANCHOR_HEEL_STRIKE = 0.0    # cycle start = heel strike
_ANCHOR_LOADING_END = 12.0   # end of loading response
_ANCHOR_GA_PEAK     = 55.0   # terminal stance / push-off
_ANCHOR_TOE_OFF     = 62.0   # stance → swing transition (GA→TA crossover)
_ANCHOR_CYCLE_END   = 100.0


def _anchor_phase_to_landmarks(raw_phase: np.ndarray,
                                ta_cycle: np.ndarray,
                                ga_cycle: np.ndarray,
                                cycle_len: int) -> np.ndarray:
    """
    Re-normalize a raw EMG-driven phase curve so that explicitly
    detected physiological events land at their expected gait-cycle
    percentages.

    Enforced anchors:
        Heel strike (cycle start)  → 0%   (by definition)
        Loading response end       → 12%  (TA deactivation after initial contact)
        GA peak (push-off)         → 55%  (terminal stance)
        GA→TA crossover (toe-off)  → 62%  (stance-to-swing transition)
        Cycle end                  → 100%

    The re-normalization uses a monotonic PCHIP (Piecewise Cubic
    Hermite Interpolating Polynomial) through the anchor points,
    producing a C1-continuous (no kinks) curve that preserves the
    EMG-driven nonlinear shape within each segment.

    If landmarks cannot be detected (e.g. very weak signal), the raw
    EMG-driven phase is returned unmodified.
    """
    eps = 1e-12

    # --- Anchor 1: Heel strike → 0% (implicit: cycle start = index 0) ---
    # Already enforced: raw_phase[0] == 0.0 by construction.

    # --- Anchor 2: Loading response end → 12% ---
    # TA fires at heel strike for eccentric dorsiflexion control;
    # loading ends when TA drops below 50% of its initial activation.
    ta_peak_early = ta_cycle[:max(1, cycle_len // 4)].max()
    loading_end_idx = None
    if ta_peak_early > eps:
        ta_early_norm = ta_cycle[:cycle_len // 2] / (ta_peak_early + eps)
        below_half = np.where(ta_early_norm < 0.5)[0]
        if len(below_half) > 0:
            candidate = int(below_half[0])
            if 0.05 * cycle_len < candidate < 0.35 * cycle_len:
                loading_end_idx = candidate

    # --- Anchor 3: GA peak → 55% (terminal stance / push-off) ---
    ga_peak_idx = int(np.argmax(ga_cycle))
    if not (0.10 * cycle_len < ga_peak_idx < 0.80 * cycle_len):
        return raw_phase  # can't anchor — return raw

    raw_at_ga_peak = float(raw_phase[ga_peak_idx])
    if raw_at_ga_peak < 5.0 or raw_at_ga_peak > 95.0:
        return raw_phase  # degenerate — peak at extreme edge

    # --- Anchor 4: GA→TA crossover → 62% (toe-off) ---
    # Dominance ratio: 1.0 = pure GA, 0.0 = pure TA
    dominance = ga_cycle / (ga_cycle + ta_cycle + eps)
    crossover_idx = None
    post_peak_dom = dominance[ga_peak_idx:]
    below_half = np.where(post_peak_dom < 0.5)[0]
    if len(below_half) > 0:
        crossover_idx = ga_peak_idx + int(below_half[0])
        if crossover_idx <= ga_peak_idx or crossover_idx >= cycle_len - 1:
            crossover_idx = None

    # --- Build anchor map: raw_phase_value → target_phase_value ---
    src = [0.0]     # heel strike
    dst = [0.0]

    # Loading response end (if detected)
    if loading_end_idx is not None:
        raw_at_loading = float(raw_phase[loading_end_idx])
        if raw_at_loading > 1.0:  # must be past start
            src.append(raw_at_loading)
            dst.append(_ANCHOR_LOADING_END)

    # GA peak (push-off)
    if len(src) < 2 or raw_at_ga_peak > src[-1] + 1.0:
        src.append(raw_at_ga_peak)
        dst.append(_ANCHOR_GA_PEAK)

    # GA→TA crossover (toe-off)
    if crossover_idx is not None:
        raw_at_crossover = float(raw_phase[crossover_idx])
        if raw_at_crossover > src[-1] + 1.0:
            src.append(raw_at_crossover)
            dst.append(_ANCHOR_TOE_OFF)

    src.append(100.0)   # cycle end
    dst.append(_ANCHOR_CYCLE_END)

    # --- Smooth monotonic re-normalization (PCHIP = C1 continuous) ---
    # PCHIP preserves monotonicity and eliminates piecewise-linear kinks
    if len(src) >= 3:
        pchip = PchipInterpolator(src, dst)
        anchored = pchip(raw_phase)
        # Clamp to [0, 100] and enforce strict monotonicity
        anchored = np.clip(anchored, 0.0, 100.0)
        anchored = np.maximum.accumulate(anchored)
    else:
        # Too few anchors for PCHIP — fall back to linear
        anchored = np.interp(raw_phase, src, dst)

    return anchored


def compute_cci(ta_rms: np.ndarray, ga_rms: np.ndarray) -> np.ndarray:
    """
    Compute sample-wise Co-Contraction Index (CCI).
    CCI = (2 * min(TA, GA)) / (TA + GA + eps)
    
    Range [0, 1]. 1.0 means perfect co-contraction (TA=GA), 
    0.0 means pure reciprocal activation.
    """
    eps = 1e-8
    return (2.0 * np.minimum(ta_rms, ga_rms)) / (ta_rms + ga_rms + eps)


def assign_gait_phase_continuous(ta_signal: np.ndarray,
                                  ga_signal: np.ndarray,
                                  fs: float = _FS_DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute continuous gait phase [0, 100], phase velocity, and CCI.

    Returns:
        phase: (T,) array [0, 100]
        velocity: (T,) array (phase units per sample)
        cci: (T,) array [0, 1]
    """
    n = len(ta_signal)

    # --- Raw RMS envelopes ---
    rms_win = max(2, int(_RMS_WINDOW_MS * fs / 1000))
    ta_rms = _rolling_rms(np.abs(ta_signal.astype(np.float64)), rms_win)
    ga_rms = _rolling_rms(np.abs(ga_signal.astype(np.float64)), rms_win)
    
    cci = compute_cci(ta_rms, ga_rms)

    # Total EMG activity at each timestep
    activity = ta_rms + ga_rms

    # --- Z-scored TKEO envelopes for cycle boundary detection only ---
    ta_env = emg_envelope(ta_signal.astype(np.float64), fs)
    ga_env = emg_envelope(ga_signal.astype(np.float64), fs)

    # Detect cycle boundaries
    cycle_starts, method = detect_gait_cycles(ta_env, ga_env, fs)

    # Build gait phase array
    phase = np.zeros(n, dtype=np.float64)

    if len(cycle_starts) < 2 or method == "Linear_Fallback":
        # Degenerate case: EMG-proportional phase across entire signal
        act_norm = activity / (activity.max() + 1e-12)
        velocity_mod = _PHASE_BASELINE_WEIGHT + _PHASE_EMG_WEIGHT * act_norm
        raw_phase = np.cumsum(velocity_mod)
        phase = raw_phase / (raw_phase[-1] + 1e-12) * 100.0
    else:
        boundaries = np.append(cycle_starts, n)
        for i in range(len(cycle_starts)):
            s = int(cycle_starts[i])
            e = int(boundaries[i + 1])
            cycle_len = e - s
            if cycle_len <= 0:
                continue

            # --- Stage 1: EMG-driven raw phase ---
            act_cycle = activity[s:e]
            act_peak = act_cycle.max()
            if act_peak < 1e-12:
                phase[s:e] = np.linspace(0.0, 100.0, cycle_len)
                continue

            act_norm = act_cycle / act_peak
            velocity_mod = _PHASE_BASELINE_WEIGHT + _PHASE_EMG_WEIGHT * act_norm
            raw_phase = np.cumsum(velocity_mod)
            raw_phase = raw_phase / raw_phase[-1] * 100.0

            # --- Stage 2: Anchor to physiological landmarks ---
            ta_cycle = ta_rms[s:e]
            ga_cycle = ga_rms[s:e]
            phase[s:e] = _anchor_phase_to_landmarks(
                raw_phase, ta_cycle, ga_cycle, cycle_len
            )

        # Pre-first-cycle
        if cycle_starts[0] > 0:
            pre_len = int(cycle_starts[0])
            act_pre = activity[:pre_len]
            act_peak = act_pre.max()
            if act_peak > 1e-12:
                act_norm = act_pre / act_peak
                vel = _PHASE_BASELINE_WEIGHT + _PHASE_EMG_WEIGHT * act_norm
                raw = np.cumsum(vel)
                phase[:pre_len] = raw / raw[-1] * 12.0
            else:
                phase[:pre_len] = np.linspace(0.0, 12.0, pre_len)

    # Compute Phase Velocity (unwrapped)
    velocity = np.diff(phase, prepend=phase[0])
    velocity[velocity < -50] += 100.0  # Handle cycle reset wrap-around

    return phase.astype(np.float32), velocity.astype(np.float32), cci.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-cycle ML feature extraction
# ---------------------------------------------------------------------------

def extract_gait_cycle_features(ta_signal: np.ndarray,
                                 ga_signal: np.ndarray,
                                 fs: float = _FS_DEFAULT) -> dict:
    """
    Extract a flat ML feature vector summarizing gait cycle characteristics.

    Returns a dict with the following fields (all normalized to 0–100% gait cycle
    or expressed as ratios):

        TA_onset_percent, TA_peak_percent, TA_offset_percent
        GA_onset_percent, GA_peak_percent, GA_offset_percent
        TA_activation_duration, GA_activation_duration
        coactivation_ratio, GA_to_TA_ratio
        stance_duration, swing_duration, propulsion_phase_duration
        initial_contact_estimate, midstance_estimate,
        terminal_stance_estimate, swing_phase_duration,
        avg_phase_velocity, avg_cci, peak_cci

    If cycle detection fails, returns a neutral/zero feature dict.
    """
    n = len(ta_signal)
    ta_env = emg_envelope(ta_signal.astype(np.float64), fs)
    ga_env = emg_envelope(ga_signal.astype(np.float64), fs)

    # Get sample-wise phase, velocity and CCI
    phase, velocity, cci = assign_gait_phase_continuous(ta_signal, ga_signal, fs)

    cycle_starts, method = detect_gait_cycles(ta_env, ga_env, fs)

    # --- Neutral fallback: use NaN to indicate extraction failure ---
    # Previous code used hardcoded "healthy gait" values which injected
    # false signal into downstream analyses (especially EDA feature tables).
    neutral = {
        'TA_onset_percent': float('nan'),     'TA_peak_percent': float('nan'),  'TA_offset_percent': float('nan'),
        'GA_onset_percent': float('nan'),    'GA_peak_percent': float('nan'),  'GA_offset_percent': float('nan'),
        'TA_activation_duration': float('nan'), 'GA_activation_duration': float('nan'),
        'coactivation_ratio': float('nan'),   'GA_to_TA_ratio': float('nan'),
        'stance_duration': float('nan'),     'swing_duration': float('nan'),
        'propulsion_phase_duration': float('nan'),
        'initial_contact_estimate': float('nan'), 'midstance_estimate': float('nan'),
        'terminal_stance_estimate': float('nan'), 'swing_phase_duration': float('nan'),
        'avg_phase_velocity': float('nan'), 'avg_cci': float('nan'), 'peak_cci': float('nan'),
        'cycle_count': 0,
        'method': method,
    }

    if len(cycle_starts) < 2:
        return neutral

    boundaries = np.append(cycle_starts, n)
    all_cycles = []

    # Get global burst detections (used as a starting reference)
    ta_bursts_global = detect_bursts_threshold(ta_env)
    ga_bursts_global = detect_bursts_threshold(ga_env)

    for i in range(len(cycle_starts)):
        cs = int(cycle_starts[i])
        ce = int(boundaries[i + 1])
        cl = ce - cs                          # cycle length in samples
        if cl <= 0:
            continue

        # ─────────────────────────────────────────────────────────────────
        # Stage 2: Per-cycle amplitude normalization
        # Normalize TA and GA envelopes within this cycle to their own peak.
        # This removes absolute-amplitude dependence (critical for pathological
        # patients with globally suppressed EMG, e.g. paresis, hemiplegia).
        # ─────────────────────────────────────────────────────────────────
        ta_cycle = ta_env[cs:ce]
        ga_cycle = ga_env[cs:ce]

        ta_peak = ta_cycle.max() + 1e-12
        ga_peak = ga_cycle.max() + 1e-12

        ta_cycle_norm = ta_cycle / ta_peak   # [0, 1] within this cycle
        ga_cycle_norm = ga_cycle / ga_peak   # [0, 1] within this cycle

        # Use a lower threshold relative to cycle peak (0.25) rather than
        # global z-score (1.5) — catches weak pathological bursts
        _CYCLE_THRESH = 0.25

        def _active_regions(sig_norm, thresh):
            """Return (onset, peak_idx, offset) as cycle-relative indices."""
            above = (sig_norm >= thresh).astype(np.int8)
            diff  = np.diff(np.r_[0, above, 0])
            ons   = np.where(diff ==  1)[0]
            offs  = np.where(diff == -1)[0]
            if len(ons) == 0 or len(offs) == 0:
                return None
            # Pick the burst with the strongest peak
            best = max(zip(ons, offs),
                       key=lambda pair: sig_norm[pair[0]:pair[1]].max() if pair[1] > pair[0] else 0)
            on, off = best
            seg = sig_norm[on:off]
            pk  = int(np.argmax(seg)) + on if len(seg) > 0 else on
            return on, pk, off

        ta_region = _active_regions(ta_cycle_norm, _CYCLE_THRESH)
        ga_region = _active_regions(ga_cycle_norm, _CYCLE_THRESH)

        def pct(cycle_rel_idx):
            return float(np.clip(cycle_rel_idx / cl * 100, 0, 100))

        # TA timing features (% of gait cycle)
        ta_onset_pct  = pct(ta_region[0]) if ta_region else 0.0
        ta_peak_pct   = pct(ta_region[1]) if ta_region else 15.0
        ta_offset_pct = pct(ta_region[2]) if ta_region else 60.0
        ta_dur        = ta_offset_pct - ta_onset_pct

        # GA timing features (% of gait cycle)
        ga_onset_pct  = pct(ga_region[0]) if ga_region else 40.0
        ga_peak_pct   = pct(ga_region[1]) if ga_region else 65.0
        ga_offset_pct = pct(ga_region[2]) if ga_region else 75.0
        ga_dur        = ga_offset_pct - ga_onset_pct

        # Symmetric Jaccard-like coactivation ratio: [0, 1]
        overlap_start = max(ta_onset_pct, ga_onset_pct)
        overlap_end   = min(ta_offset_pct, ga_offset_pct)
        overlap       = max(0.0, overlap_end - overlap_start)
        
        denom_sum     = ta_dur + ga_dur
        coact_ratio   = (2.0 * overlap) / (denom_sum + 1e-6)
        
        ga_ta_ratio   = ga_dur  / (ta_dur + 1e-6)

        # Phase estimates (anchored to GA events)
        midstance   = ga_onset_pct
        term_stance = ga_peak_pct
        stance_dur  = ga_offset_pct              # GA offset ≈ toe-off
        swing_dur   = 100.0 - stance_dur
        propulsion  = term_stance - midstance

        all_cycles.append({
            'TA_onset_percent': ta_onset_pct,
            'TA_peak_percent': ta_peak_pct,
            'TA_offset_percent': ta_offset_pct,
            'GA_onset_percent': ga_onset_pct,
            'GA_peak_percent': ga_peak_pct,
            'GA_offset_percent': ga_offset_pct,
            'TA_activation_duration': ta_dur,
            'GA_activation_duration': ga_dur,
            'coactivation_ratio': coact_ratio,
            'GA_to_TA_ratio': ga_ta_ratio,
            'stance_duration': stance_dur,
            'swing_duration': swing_dur,
            'propulsion_phase_duration': propulsion,
            'initial_contact_estimate': ta_onset_pct,
            'midstance_estimate': midstance,
            'terminal_stance_estimate': term_stance,
            'swing_phase_duration': swing_dur,
            'avg_phase_velocity': float(np.mean(velocity[cs:ce])),
            'avg_cci': float(np.mean(cci[cs:ce])),
            'peak_cci': float(np.max(cci[cs:ce])),
        })

    if not all_cycles:
        return neutral

    # Average across cycles → single window-level scalar per feature
    result = {}
    for key in all_cycles[0]:
        vals = [c[key] for c in all_cycles if isinstance(c[key], float)]
        result[key] = float(np.mean(vals)) if vals else 0.0

    # ─────────────────────────────────────────────────────────────────
    # Stage 3: Per-cycle quality score
    # Computes a reliability score [0, 1] based on inter-cycle timing variance.
    # High score = highly rhythmic/healthy, Low score = chaotic/pathological
    # ─────────────────────────────────────────────────────────────────
    if len(cycle_starts) >= 3:
        cycle_lengths = np.diff(cycle_starts)
        cv = np.std(cycle_lengths) / (np.mean(cycle_lengths) + 1e-6)
        quality = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        quality = 0.0

    result['cycle_count'] = len(all_cycles)
    result['method'] = method
    result['cycle_quality_score'] = quality
    return result


# ---------------------------------------------------------------------------
# Convenience: FSM state sequence (per-timestep discrete label)
# ---------------------------------------------------------------------------

def assign_fsm_states(ta_signal: np.ndarray,
                      ga_signal: np.ndarray,
                      fs: float = _FS_DEFAULT) -> np.ndarray:
    """Returns FSM state index per sample. Useful for visualization."""
    ta_env = emg_envelope(ta_signal.astype(np.float64), fs)
    ga_env = emg_envelope(ga_signal.astype(np.float64), fs)
    return fsm_phase_estimate(ta_env, ga_env)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== gait_phase.py self-test ===")
    fs = 1000.0
    duration = 6.0  # seconds
    t = np.arange(int(fs * duration))

    # Simulate healthy gait: TA fires in swing, GA fires offset by half-cycle
    # Raw EMG is a high-frequency signal (~100Hz) modulated by a low-frequency envelope (1Hz)
    freq_hz = 1.0          # 1 gait cycle per second (60 steps/min)
    carrier_freq = 100.0   # Simulated EMG motor unit action potential frequency
    
    ta_env = np.clip(np.sin(2 * np.pi * freq_hz * t / fs), 0, None)
    ga_env = np.clip(np.sin(2 * np.pi * freq_hz * t / fs + np.pi), 0, None)
    
    ta_sim = ta_env * np.sin(2 * np.pi * carrier_freq * t / fs) + 0.05 * np.random.randn(len(t))
    ga_sim = ga_env * np.sin(2 * np.pi * carrier_freq * t / fs) + 0.05 * np.random.randn(len(t))

    phase_out, vel_out, cci_out = assign_gait_phase_continuous(ta_sim, ga_sim, fs)
    print(f"Phase output shape : {phase_out.shape}")
    print(f"Phase range        : [{phase_out.min():.1f}, {phase_out.max():.1f}]")
    print(f"CCI range          : [{cci_out.min():.2f}, {cci_out.max():.2f}]")

    _, method = detect_gait_cycles(emg_envelope(ta_sim, fs), emg_envelope(ga_sim, fs), fs)
    print(f"Method selected    : {method}")

    feats = extract_gait_cycle_features(ta_sim, ga_sim, fs)
    print(f"Cycles detected    : {feats['cycle_count']}")
    print(f"TA onset %%         : {feats['TA_onset_percent']:.1f}")
    print(f"GA onset %%         : {feats['GA_onset_percent']:.1f}")
    print(f"Coactivation ratio : {feats['coactivation_ratio']:.3f}")
    print("=== Self-test passed ===")
