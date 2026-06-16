import time
import bisect
import heapq
from collections import deque
from typing import Optional, Tuple, List, Dict, Any, Union


# ============================================================================
# 1. 严格语义滑动窗口限流器
# ============================================================================

class StrictSlidingWindowLimiter:
    """
    严格语义的滑动窗口限流器

    保证：对于任意时刻 t，区间 (t - window, t] 内通过的请求数 <= N
    （注意左开右闭）
    """

    def __init__(self, max_per_second: int, window_seconds: float = 1.0):
        self.max_per_second = max_per_second
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self.last_activity: float = 0.0

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def allow(self, now: Optional[float] = None) -> Tuple[bool, int]:
        if now is None:
            now = time.monotonic()
        self.last_activity = now

        self._evict_expired(now)

        current_count = len(self._timestamps)
        if current_count < self.max_per_second:
            self._timestamps.append(now)
            return True, current_count + 1
        return False, current_count

    def count_in_window(self, t_end: float) -> int:
        """返回严格左开右闭窗口 (t_end - window, t_end] 内的请求数"""
        t_start = t_end - self.window_seconds
        count = 0
        for ts in self._timestamps:
            if t_start < ts <= t_end:
                count += 1
        return count

    def get_all_timestamps(self) -> List[float]:
        return list(self._timestamps)

    def reset(self) -> None:
        self._timestamps.clear()
        self.last_activity = 0.0

    @property
    def state_size(self) -> int:
        return len(self._timestamps)


# ============================================================================
# 2. 标准令牌桶（用于对比）
# ============================================================================

class TokenBucketLimiter:
    """
    标准令牌桶

    tokens(t) = min(capacity, tokens(t_prev) + (t - t_prev) * rate)
    每次请求消耗 1 个令牌

    注意：不满足严格滑动窗口语义
    """

    def __init__(self, rate_per_second: int, capacity: Optional[int] = None):
        self.rate = rate_per_second
        self.capacity = capacity if capacity is not None else rate_per_second
        self._tokens = float(self.capacity)
        self._last_refill = None
        self._allowed_times: List[float] = []
        self.last_activity: float = 0.0

    def _refill(self, now: float) -> None:
        if self._last_refill is None:
            self._last_refill = now
            return
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def allow(self, now: Optional[float] = None) -> Tuple[bool, float]:
        if now is None:
            now = time.monotonic()
        self.last_activity = now

        self._refill(now)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            self._allowed_times.append(now)
            return True, self._tokens
        return False, self._tokens

    def count_in_window(self, t_end: float, window_seconds: float = 1.0) -> int:
        t_start = t_end - window_seconds
        return sum(1 for ts in self._allowed_times if t_start < ts <= t_end)

    def get_all_timestamps(self) -> List[float]:
        return list(self._allowed_times)

    def reset(self) -> None:
        self._tokens = float(self.capacity)
        self._last_refill = None
        self._allowed_times.clear()
        self.last_activity = 0.0

    @property
    def state_size(self) -> int:
        return 1


# ============================================================================
# 3. 压缩状态滑动窗口限流器（百万级 QPS 场景）
# ============================================================================

class CompressedSlidingWindowLimiter:
    """
    压缩状态滑动窗口限流器

    将 1 秒窗口划分为 K 个微桶，只记录每桶的计数。
    内存 = O(K) 与 N 无关；误差 <= 1 个桶宽内的请求数。
    """

    def __init__(
        self,
        max_per_second: int,
        window_seconds: float = 1.0,
        bucket_width_ms: int = 1,
    ):
        self.max_per_second = max_per_second
        self.window_seconds = window_seconds
        self.bucket_width_ms = bucket_width_ms
        self.bucket_width_sec = bucket_width_ms / 1000.0
        self.last_activity: float = 0.0

        self.num_buckets = int(window_seconds * 1000 / bucket_width_ms) + 2

        self._counts: List[int] = [0] * self.num_buckets
        self._bucket_ids: List[int] = [0] * self.num_buckets
        self._decision_log: List[Tuple[float, bool, int]] = []

    def _get_bucket_id(self, now: float) -> int:
        return int(now / self.bucket_width_sec)

    def _update_window(self, now: float) -> Tuple[int, int]:
        current_bucket_id = self._get_bucket_id(now)
        oldest_allowed_bucket_id = current_bucket_id - self.num_buckets + 1

        total = 0
        for i in range(self.num_buckets):
            bid = self._bucket_ids[i]
            if bid >= oldest_allowed_bucket_id and bid <= current_bucket_id:
                total += self._counts[i]
            else:
                self._counts[i] = 0
                self._bucket_ids[i] = 0

        return total, current_bucket_id

    def allow(self, now: Optional[float] = None) -> Tuple[bool, int]:
        if now is None:
            now = time.monotonic()
        self.last_activity = now

        current_count, current_bucket_id = self._update_window(now)

        if current_count < self.max_per_second:
            idx = current_bucket_id % self.num_buckets
            if self._bucket_ids[idx] != current_bucket_id:
                self._bucket_ids[idx] = current_bucket_id
                self._counts[idx] = 0
            self._counts[idx] += 1
            decision = True
            new_count = current_count + 1
        else:
            decision = False
            new_count = current_count

        self._decision_log.append((now, decision, new_count))
        return decision, new_count

    def error_report(self) -> Dict[str, Any]:
        """
        输出直观的误差分析报告
        """
        N = self.max_per_second
        B_ms = self.bucket_width_ms
        W_sec = self.window_seconds

        max_extra_allowed = int(N * B_ms / (W_sec * 1000))
        error_ratio_upper = max_extra_allowed / N * 100
        memory = self.memory_bytes()
        strict_memory = N * 8

        if B_ms <= 1:
            scenario = "金融/计费/精确配额管控等高精确场景"
            qps_range = "10万 ~ 100万 QPS"
        elif B_ms <= 5:
            scenario = "一般 API 网关限流、防刷"
            qps_range = "100万 ~ 500万 QPS"
        elif B_ms <= 10:
            scenario = "大流量粗粒度限流、防 DDoS 外层"
            qps_range = "500万 ~ 1000万 QPS"
        elif B_ms <= 50:
            scenario = "超大规模边缘防护，只关心数量级"
            qps_range = "> 1000万 QPS"
        else:
            scenario = "仅做防止极端雪崩的保护"
            qps_range = "极高流量"

        return {
            "params": {
                "N (max_per_second)": N,
                "window_seconds": W_sec,
                "bucket_width_ms": B_ms,
                "num_buckets": self.num_buckets,
            },
            "memory": {
                "compressed_bytes": memory,
                "compressed_kb": memory / 1024,
                "strict_bytes": strict_memory,
                "strict_mb": strict_memory / 1024 / 1024,
                "savings_ratio": (1 - memory / strict_memory) * 100 if strict_memory > 0 else 0,
            },
            "precision": {
                "worst_case_extra_allowed": max_extra_allowed,
                "worst_case_extra_blocked": max_extra_allowed,
                "relative_error_pct_max": error_ratio_upper,
                "note": "误差来源于桶边界无法精确切割；最坏情况下 1 个整桶的请求被多/少算",
            },
            "recommendation": {
                "适用场景": scenario,
                "推荐QPS范围": qps_range,
            },
        }

    def format_error_report(self) -> str:
        r = self.error_report()
        p = r["params"]
        m = r["memory"]
        pr = r["precision"]
        rec = r["recommendation"]

        lines = [
            "=" * 60,
            "压缩滑动窗口误差报告",
            "=" * 60,
            f"  参数: N={p['N (max_per_second)']:,}/s, 窗口={p['window_seconds']}s, "
            f"桶宽={p['bucket_width_ms']}ms, 桶数={p['num_buckets']}",
            "",
            "  内存占用:",
            f"    压缩版本:   {m['compressed_bytes']:,} B = {m['compressed_kb']:.2f} KB",
            f"    严格逐记录: {m['strict_bytes']:,} B = {m['strict_mb']:.2f} MB",
            f"    节省比例:   {m['savings_ratio']:.1f}%",
            "",
            "  精度分析:",
            f"    最坏情况多放行: {pr['worst_case_extra_allowed']:,} 个请求",
            f"    最坏情况多拒绝: {pr['worst_case_extra_blocked']:,} 个请求",
            f"    相对误差上界:   ≤ {pr['relative_error_pct_max']:.3f}%",
            f"    说明: {pr['note']}",
            "",
            "  推荐:",
            f"    适用场景: {rec['适用场景']}",
            f"    QPS 范围: {rec['推荐QPS范围']}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        for i in range(self.num_buckets):
            self._counts[i] = 0
            self._bucket_ids[i] = 0
        self._decision_log.clear()
        self.last_activity = 0.0

    @property
    def state_size(self) -> int:
        return self.num_buckets

    def memory_bytes(self) -> int:
        return self.num_buckets * (8 + 8)


# ============================================================================
# 4. 令牌桶违规分析（严格左开右闭窗口）
# ============================================================================

def analyze_token_bucket_violation_strict(
    rate_per_second: int,
    capacity: Optional[int] = None,
    window_seconds: float = 1.0,
) -> Dict[str, Any]:
    """
    严格分析令牌桶违反滑动窗口语义的情况

    选取的检查窗口：
      - 窗口 W_A = (T_A - window, T_A]，其中 T_A 是所有请求通过后
      - 逐一遍历每个通过请求的时间戳 t，检查窗口 (t - window, t]

    这样保证：
      - 窗口严格左开右闭
      - 只检查真实落在窗口内的请求
      - 明确指出违规窗口的起止时间和包含的请求
    """
    if capacity is None:
        capacity = rate_per_second

    N = rate_per_second

    tb = TokenBucketLimiter(N, capacity=capacity)
    ss = StrictSlidingWindowLimiter(N, window_seconds=window_seconds)

    request_events: List[Tuple[float, str]] = []

    # 场景：在 T=0 突发发送 capacity 个请求（即令牌桶满桶时的容量）
    T_BURST = 0.0
    tb_allowed_times: List[float] = []
    ss_allowed_times: List[float] = []

    for i in range(capacity * 3):
        ok, _ = tb.allow(now=T_BURST)
        if ok:
            tb_allowed_times.append(T_BURST)
        else:
            break

    for i in range(N * 3):
        ok, _ = ss.allow(now=T_BURST)
        if ok:
            ss_allowed_times.append(T_BURST)
        else:
            break

    request_events.append((T_BURST, f"突发 {len(tb_allowed_times)} 个请求"))

    # 然后在 T = window - ε（刚好窗口结束前）再持续发送
    T_CONTINUE = window_seconds - 1e-6
    while True:
        ok, _ = tb.allow(now=T_CONTINUE)
        if ok:
            tb_allowed_times.append(T_CONTINUE)
        else:
            break
    # 严格滑动窗口
    while True:
        ok, _ = ss.allow(now=T_CONTINUE)
        if ok:
            ss_allowed_times.append(T_CONTINUE)
        else:
            break

    request_events.append((T_CONTINUE, f"窗口结束前再通过 {len([t for t in tb_allowed_times if t == T_CONTINUE])} 个"))

    # --- 严格检查：找出令牌桶违规的那个窗口 ---
    worst_window = None
    worst_count = 0
    worst_t_end = None

    all_tb_times = sorted(tb_allowed_times)

    # 对每个通过的请求，以它的时间为窗口右端点
    for t_end in all_tb_times:
        t_start = t_end - window_seconds
        count = sum(1 for ts in all_tb_times if t_start < ts <= t_end)
        if count > worst_count:
            worst_count = count
            worst_t_end = t_end
            worst_window = (t_start, t_end)

    violation_count = max(0, worst_count - N)

    # 收集落在最坏窗口内的请求明细
    if worst_window:
        ws, we = worst_window
        inside = [ts for ts in all_tb_times if ws < ts <= we]
        inside_summary = {}
        for ts in inside:
            key = f"{ts:.6f}s"
            inside_summary[key] = inside_summary.get(key, 0) + 1
    else:
        inside_summary = {}
        inside = []

    # 对比严格滑动窗口的结果
    ss_worst_count = 0
    all_ss_times = sorted(ss_allowed_times)
    for t_end in all_ss_times:
        t_start = t_end - window_seconds
        c = sum(1 for ts in all_ss_times if t_start < ts <= t_end)
        ss_worst_count = max(ss_worst_count, c)

    return {
        "params": {
            "N": N,
            "capacity": capacity,
            "window_seconds": window_seconds,
        },
        "scenario_events": request_events,
        "token_bucket": {
            "total_allowed": len(tb_allowed_times),
            "worst_window": worst_window,
            "worst_window_count": worst_count,
            "worst_window_t_end": worst_t_end,
            "violation_count": violation_count,
            "violation_ratio": worst_count / N if N > 0 else 0,
            "requests_in_worst_window": inside_summary,
            "request_times_sample": all_tb_times[:20],
        },
        "strict_sliding_window": {
            "total_allowed": len(ss_allowed_times),
            "worst_window_count": ss_worst_count,
            "violation": ss_worst_count > N,
        },
        "explanation": (
            f"令牌桶在 T=0 时因为桶是满的，立即通过了 capacity={capacity} 个请求；"
            f"经过 {window_seconds}s 的充能，又补充了约 {N} 个令牌；"
            f"因此在跨越 0 时刻的窗口内，合计通过了 {worst_count} > {N} 个请求。"
        ),
    }


def format_violation_report(report: Dict[str, Any]) -> str:
    """将违规分析格式化为易读的终端输出"""
    p = report["params"]
    tb = report["token_bucket"]
    ss = report["strict_sliding_window"]

    lines = [
        "=" * 70,
        "令牌桶违反严格滑动窗口语义 - 分析报告（严格左开右闭窗口）",
        "=" * 70,
        "",
        f"  参数配置:",
        f"    rate = N = {p['N']}/s, capacity = {p['capacity']}, 窗口 = {p['window_seconds']}s",
        "",
        f"  场景事件序列:",
    ]

    for t, desc in report["scenario_events"]:
        lines.append(f"    T = {t:.6f}s: {desc}")

    lines += [
        "",
        f"  ──────────────────────────────────────────",
        f"  令牌桶实际行为:",
        f"    总通过请求数: {tb['total_allowed']}",
        "",
    ]

    if tb["worst_window"]:
        ws, we = tb["worst_window"]
        lines += [
            f"  ★ 发现违规窗口:",
            f"    W = ({ws:.6f}s, {we:.6f}s]     ← 左开右闭",
            f"    窗口内请求数: {tb['worst_window_count']}",
            f"    限制 N = {p['N']}",
            f"    超出: {tb['violation_count']} 个 ({tb['violation_ratio']*100:.1f}%)",
            "",
            f"    窗口内各时间点请求明细:",
        ]
        for ts_key, cnt in sorted(tb["requests_in_worst_window"].items()):
            lines.append(f"      t = {ts_key}: {cnt} 个请求")

        lines += [
            "",
            f"    为什么会超？{report['explanation']}",
        ]

    lines += [
        "",
        f"  ──────────────────────────────────────────",
        f"  严格滑动窗口对比:",
        f"    总通过请求数: {ss['total_allowed']}",
        f"    任意 1s 窗口内最大计数: {ss['worst_window_count']}",
        f"    是否违规: {'是 ❌' if ss['violation'] else '否 ✓'}",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)


# ============================================================================
# 5. 按 Key 限流的管理器（真实网关场景）
# ============================================================================

LimiterType = Union[StrictSlidingWindowLimiter, CompressedSlidingWindowLimiter, TokenBucketLimiter]


class MultiKeyRateLimiter:
    """
    按 Key 限流的管理器（类似真实网关按 IP / 用户 ID 维度限流）

    特性：
      - 每个 key 拥有独立的限流器实例，互不干扰
      - 支持三种限流器后端：strict / compressed / token-bucket
      - 自动清理长时间未活动的 key，防止内存泄漏
      - 清理策略：惰性清理 + 定期主动清理
    """

    def __init__(
        self,
        max_per_second: int,
        window_seconds: float = 1.0,
        limiter_type: str = "strict",
        bucket_width_ms: int = 1,
        idle_ttl_seconds: float = 60.0,
        cleanup_interval_seconds: float = 10.0,
    ):
        """
        Args:
            max_per_second: 每个 key 的限流阈值
            window_seconds: 窗口大小
            limiter_type: "strict" | "compressed" | "token-bucket"
            bucket_width_ms: 仅 compressed 模式使用，微桶宽度
            idle_ttl_seconds: key 空闲超过此时间自动清理
            cleanup_interval_seconds: 主动清理的最小间隔
        """
        self.max_per_second = max_per_second
        self.window_seconds = window_seconds
        self.limiter_type = limiter_type
        self.bucket_width_ms = bucket_width_ms
        self.idle_ttl_seconds = idle_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds

        self._limiters: Dict[str, LimiterType] = {}
        self._last_activity: Dict[str, float] = {}
        self._last_cleanup: float = 0.0

    def _create_limiter(self) -> LimiterType:
        if self.limiter_type == "strict":
            return StrictSlidingWindowLimiter(self.max_per_second, self.window_seconds)
        elif self.limiter_type == "compressed":
            return CompressedSlidingWindowLimiter(
                self.max_per_second, self.window_seconds, self.bucket_width_ms
            )
        elif self.limiter_type == "token-bucket":
            return TokenBucketLimiter(self.max_per_second)
        else:
            raise ValueError(f"Unknown limiter_type: {self.limiter_type}")

    def _maybe_cleanup(self, now: float) -> None:
        if now - self._last_cleanup < self.cleanup_interval_seconds:
            return

        self._last_cleanup = now
        cutoff = now - self.idle_ttl_seconds

        expired_keys = [k for k, t in self._last_activity.items() if t <= cutoff]
        for k in expired_keys:
            del self._limiters[k]
            del self._last_activity[k]

    def allow(self, key: str, now: Optional[float] = None) -> Tuple[bool, int]:
        if now is None:
            now = time.monotonic()

        self._maybe_cleanup(now)

        if key not in self._limiters:
            self._limiters[key] = self._create_limiter()

        limiter = self._limiters[key]
        ok, count = limiter.allow(now=now)
        self._last_activity[key] = now
        return ok, count

    def cleanup_expired(self, now: Optional[float] = None) -> int:
        """强制触发一次清理，返回被清理的 key 数量"""
        if now is None:
            now = time.monotonic()

        cutoff = now - self.idle_ttl_seconds
        expired_keys = [k for k, t in self._last_activity.items() if t <= cutoff]

        for k in expired_keys:
            del self._limiters[k]
            del self._last_activity[k]

        self._last_cleanup = now
        return len(expired_keys)

    @property
    def active_key_count(self) -> int:
        return len(self._limiters)

    def get_active_keys(self) -> List[str]:
        return list(self._limiters.keys())

    def reset_key(self, key: str) -> None:
        if key in self._limiters:
            self._limiters[key].reset()

    def reset_all(self) -> None:
        self._limiters.clear()
        self._last_activity.clear()
        self._last_cleanup = 0.0


# ============================================================================
# 6. 压缩版 vs 严格版逐点差异对比
# ============================================================================

def compare_strict_vs_compressed(
    max_per_second: int,
    request_times: List[float],
    bucket_width_ms: int = 10,
    window_seconds: float = 1.0,
) -> Dict[str, Any]:
    """
    对给定请求时间序列，逐请求对比严格版和压缩版的决策差异
    """
    strict = StrictSlidingWindowLimiter(max_per_second, window_seconds)
    comp = CompressedSlidingWindowLimiter(max_per_second, window_seconds, bucket_width_ms)

    diffs: List[Dict[str, Any]] = []
    strict_allowed: List[float] = []
    comp_allowed: List[float] = []
    false_positives = 0  # 压缩版放行但严格版拒绝（多放行）
    false_negatives = 0  # 压缩版拒绝但严格版放行（多拒绝）
    matches = 0

    for idx, t in enumerate(request_times):
        ok_s, cnt_s = strict.allow(now=t)
        ok_c, cnt_c = comp.allow(now=t)

        if ok_s:
            strict_allowed.append(t)
        if ok_c:
            comp_allowed.append(t)

        if ok_s == ok_c:
            matches += 1
        elif ok_c and not ok_s:
            false_positives += 1
            diffs.append({
                "idx": idx,
                "time": t,
                "type": "多放行(false_positive)",
                "strict_decision": f"REJECT (count={cnt_s})",
                "compressed_decision": f"ALLOW (count={cnt_c})",
            })
        elif ok_s and not ok_c:
            false_negatives += 1
            diffs.append({
                "idx": idx,
                "time": t,
                "type": "多拒绝(false_negative)",
                "strict_decision": f"ALLOW (count={cnt_s})",
                "compressed_decision": f"REJECT (count={cnt_c})",
            })

    # 检查两者的最坏窗口计数
    def worst_window_count(times: List[float]) -> Tuple[int, Tuple[float, float]]:
        worst = 0
        worst_win = (0.0, 0.0)
        for t_end in times:
            t_start = t_end - window_seconds
            c = sum(1 for ts in times if t_start < ts <= t_end)
            if c > worst:
                worst = c
                worst_win = (t_start, t_end)
        return worst, worst_win

    strict_worst, strict_win = worst_window_count(strict_allowed)
    comp_worst, comp_win = worst_window_count(comp_allowed)

    return {
        "params": {
            "N": max_per_second,
            "window": window_seconds,
            "bucket_width_ms": bucket_width_ms,
            "total_requests": len(request_times),
        },
        "summary": {
            "strict_allowed_count": len(strict_allowed),
            "compressed_allowed_count": len(comp_allowed),
            "matching_decisions": matches,
            "false_positives (多放行)": false_positives,
            "false_negatives (多拒绝)": false_negatives,
            "match_ratio": matches / len(request_times) * 100 if request_times else 100.0,
        },
        "worst_window": {
            "strict": {"count": strict_worst, "window": strict_win},
            "compressed": {"count": comp_worst, "window": comp_win},
        },
        "diffs": diffs,
    }


def format_comparison_report(report: Dict[str, Any]) -> str:
    p = report["params"]
    s = report["summary"]
    w = report["worst_window"]

    lines = [
        "=" * 70,
        "严格版 vs 压缩版 - 逐请求决策对比报告",
        "=" * 70,
        f"  参数: N={p['N']}/s, 窗口={p['window']}s, 桶宽={p['bucket_width_ms']}ms",
        f"  请求总数: {p['total_requests']}",
        "",
        f"  ──────────────────────────────────────────",
        f"  汇总:",
        f"    严格版通过总数:   {s['strict_allowed_count']}",
        f"    压缩版通过总数:   {s['compressed_allowed_count']}",
        f"    决策一致数:       {s['matching_decisions']} ({s['match_ratio']:.1f}%)",
        f"    压缩版多放行:     {s['false_positives (多放行)']}",
        f"    压缩版多拒绝:     {s['false_negatives (多拒绝)']}",
        "",
        f"  ──────────────────────────────────────────",
        f"  最坏窗口对比:",
        f"    严格版最大计数:   {w['strict']['count']} (窗口 {w['strict']['window'][0]:.4f}~{w['strict']['window'][1]:.4f}s)",
        f"    压缩版最大计数:   {w['compressed']['count']} (窗口 {w['compressed']['window'][0]:.4f}~{w['compressed']['window'][1]:.4f}s)",
        f"    限制 N =          {p['N']}",
        f"    压缩版相对严格版差值: {w['compressed']['count'] - w['strict']['count']}",
    ]

    if report["diffs"]:
        lines += [
            "",
            f"  ──────────────────────────────────────────",
            f"  差异明细 (最多显示前 15 条):",
        ]
        for d in report["diffs"][:15]:
            lines.append(
                f"    [#{d['idx']:>4}] t={d['time']:.6f}s  {d['type']:>25}  | "
                f"严格={d['strict_decision']:>18}  压缩={d['compressed_decision']:>18}"
            )
        if len(report["diffs"]) > 15:
            lines.append(f"    ... 另有 {len(report['diffs']) - 15} 条差异未展示")

    lines.append("=" * 70)
    return "\n".join(lines)
