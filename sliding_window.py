import time
import bisect
import math
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

        低阈值修正：当 N * B / (W * 1000) < 1 时，int() 截断为 0，
        但实际误差至少为 1 个桶——因为桶边界无法精确切割窗口，
        即使该桶内只有 1 个请求，也可能被多/少算。
        所以：
          worst_case_extra = max(1, int(N * B / (W * 1000)))
          当 N 很小时，1 个桶内的请求数就是 1（最坏情况），不可忽略。
        """
        N = self.max_per_second
        B_ms = self.bucket_width_ms
        W_sec = self.window_seconds

        requests_per_bucket = N * B_ms / (W_sec * 1000)
        max_extra = max(1, int(requests_per_bucket))
        if requests_per_bucket > 0 and int(requests_per_bucket) == 0:
            max_extra = 1
        error_ratio_upper = max_extra / N * 100 if N > 0 else 0

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
                "requests_per_bucket": requests_per_bucket,
            },
            "memory": {
                "compressed_bytes": memory,
                "compressed_kb": memory / 1024,
                "strict_bytes": strict_memory,
                "strict_mb": strict_memory / 1024 / 1024,
                "savings_ratio": (1 - memory / strict_memory) * 100 if strict_memory > 0 else 0,
            },
            "precision": {
                "worst_case_extra_allowed": max_extra,
                "worst_case_extra_blocked": max_extra,
                "relative_error_pct_max": error_ratio_upper,
                "note": (
                    f"每桶平均请求={requests_per_bucket:.2f}, 最坏情况1个整桶被多/少算; "
                    f"当 N={N}, B={B_ms}ms 时, 每桶最多 {max(1, math.ceil(requests_per_bucket))} 个请求可能被错误计入"
                ),
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
            f"  每桶平均请求数: {p['requests_per_bucket']:.2f}",
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
# 3b. 按目标误差反推桶宽
# ============================================================================

def recommend_bucket_width(
    max_per_second: int,
    target_error_pct: float,
    window_seconds: float = 1.0,
    max_bucket_ms: int = 1000,
) -> Dict[str, Any]:
    """
    给定 N 和可接受的误差比例，反推推荐桶宽方案

    原理：
      相对误差上界 = max(1, ceil(N * B / (W * 1000))) / N * 100
      要让误差 <= target_error_pct:
        max(1, ceil(N * B / (W * 1000))) / N * 100 <= target_error_pct

      当 N 较大时 (N * B / 1000 >= 1):
        B <= target_error_pct / 100 * W * 1000
      当 N 较小时 (N * B / 1000 < 1), 误差至少 1/N*100:
        若 1/N * 100 > target_error_pct, 则无法满足

    返回多组方案（偏精确 / 均衡 / 偏省内存）
    """
    N = max_per_second
    W = window_seconds
    min_possible_error = 1.0 / N * 100 if N > 0 else 100.0

    if target_error_pct < min_possible_error:
        return {
            "feasible": False,
            "reason": (
                f"N={N} 时，即使桶宽无限小，分桶误差下界为 1/{N}*100 = {min_possible_error:.3f}%, "
                f"无法满足目标误差 {target_error_pct}%。"
                f"建议提高 N 或放宽误差要求至 ≥ {min_possible_error:.3f}%"
            ),
            "min_possible_error_pct": min_possible_error,
            "plans": [],
        }

    B_max_for_target = int(target_error_pct / 100 * W * 1000)
    B_max_for_target = max(1, B_max_for_target)

    candidates = []
    for B_ms in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        if B_ms > B_max_for_target:
            continue

        comp = CompressedSlidingWindowLimiter(N, W, B_ms)
        report = comp.error_report()
        actual_error = report["precision"]["relative_error_pct_max"]

        if actual_error <= target_error_pct:
            candidates.append({
                "bucket_width_ms": B_ms,
                "num_buckets": comp.num_buckets,
                "actual_error_pct": actual_error,
                "memory_bytes": comp.memory_bytes(),
                "memory_kb": comp.memory_bytes() / 1024,
                "extra_requests": report["precision"]["worst_case_extra_allowed"],
                "style": "",
            })

    if not candidates:
        return {
            "feasible": False,
            "reason": f"在目标误差 {target_error_pct}% 下未找到合适桶宽，请放宽误差或增大 N",
            "min_possible_error_pct": min_possible_error,
            "plans": [],
        }

    candidates.sort(key=lambda c: c["bucket_width_ms"])

    if len(candidates) >= 3:
        candidates[0]["style"] = "偏精确（桶窄，内存稍多）"
        candidates[len(candidates) // 2]["style"] = "均衡"
        candidates[-1]["style"] = "偏省内存（桶宽，精度稍低）"
    elif len(candidates) == 2:
        candidates[0]["style"] = "偏精确"
        candidates[1]["style"] = "偏省内存"
    else:
        candidates[0]["style"] = "唯一可行方案"

    return {
        "feasible": True,
        "params": {
            "N": N,
            "window_seconds": W,
            "target_error_pct": target_error_pct,
            "min_possible_error_pct": min_possible_error,
            "max_bucket_ms_for_target": B_max_for_target,
        },
        "plans": candidates,
    }


def format_recommend_report(rec: Dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        "压缩滑动窗口 - 桶宽推荐报告",
        "=" * 70,
    ]

    if not rec["feasible"]:
        lines += [
            "",
            f"  ⚠️  无法满足目标误差",
            f"  原因: {rec['reason']}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)

    p = rec["params"]
    lines += [
        f"  参数: N={p['N']:,}/s, 窗口={p['window_seconds']}s, "
        f"目标误差≤{p['target_error_pct']}%",
        f"  误差下界(理论最小): {p['min_possible_error_pct']:.3f}% (1/N*100)",
        f"  满足条件的最大桶宽: {p['max_bucket_ms_for_target']}ms",
        "",
        f"  {'桶宽(ms)':>9}  {'桶数':>5}  {'实际误差%':>9}  "
        f"{'内存(B)':>9}  {'多放行':>6}  {'方案特点'}",
        "  " + "-" * 65,
    ]

    for plan in rec["plans"]:
        lines.append(
            f"  {plan['bucket_width_ms']:>9}  {plan['num_buckets']:>5}  "
            f"{plan['actual_error_pct']:>9.3f}  {plan['memory_bytes']:>9,}  "
            f"{plan['extra_requests']:>6}  {plan['style']}"
        )

    lines += [
        "",
        "  建议:",
        "    选偏精确方案 → 精度优先，适合计费/配额场景",
        "    选均衡方案   → 精度与内存兼顾，适合一般 API 网关",
        "    选偏省内存   → 内存优先，适合海量 key / 边缘部署",
        "=" * 70,
    ]
    return "\n".join(lines)


# ============================================================================
# 4. 令牌桶违规分析（严格左开右闭窗口，含无违规提示）
# ============================================================================

def analyze_token_bucket_violation_strict(
    rate_per_second: int,
    capacity: Optional[int] = None,
    window_seconds: float = 1.0,
) -> Dict[str, Any]:
    """
    严格分析令牌桶违反滑动窗口语义的情况

    新增：当参数不足以触发违规时，明确说明并给出建议参数
    """
    if capacity is None:
        capacity = rate_per_second

    N = rate_per_second

    tb = TokenBucketLimiter(N, capacity=capacity)
    ss = StrictSlidingWindowLimiter(N, window_seconds=window_seconds)

    request_events: List[Tuple[float, str]] = []

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

    T_CONTINUE = window_seconds - 1e-6
    tb_continue_count = 0
    while True:
        ok, _ = tb.allow(now=T_CONTINUE)
        if ok:
            tb_allowed_times.append(T_CONTINUE)
            tb_continue_count += 1
        else:
            break

    ss_continue_count = 0
    while True:
        ok, _ = ss.allow(now=T_CONTINUE)
        if ok:
            ss_allowed_times.append(T_CONTINUE)
            ss_continue_count += 1
        else:
            break

    request_events.append(
        (T_CONTINUE, f"窗口结束前令牌桶再通过 {tb_continue_count} 个, 严格版再通过 {ss_continue_count} 个")
    )

    worst_window = None
    worst_count = 0
    worst_t_end = None

    all_tb_times = sorted(tb_allowed_times)

    for t_end in all_tb_times:
        t_start = t_end - window_seconds
        count = sum(1 for ts in all_tb_times if t_start < ts <= t_end)
        if count > worst_count:
            worst_count = count
            worst_t_end = t_end
            worst_window = (t_start, t_end)

    violation_count = max(0, worst_count - N)
    is_violated = worst_count > N

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

    ss_worst_count = 0
    all_ss_times = sorted(ss_allowed_times)
    for t_end in all_ss_times:
        t_start = t_end - window_seconds
        c = sum(1 for ts in all_ss_times if t_start < ts <= t_end)
        ss_worst_count = max(ss_worst_count, c)

    suggestion = None
    if not is_violated:
        if capacity < N:
            suggestion = (
                f"当前 capacity={capacity} < rate={N}，桶容量不足以填满 1 秒窗口。"
                f"建议将 capacity 设为 N={N} 或更大，使 T=0 时桶满突发通过 N 个，"
                f"1 秒后再补充 N 个令牌，才能触发 2N 的违规。"
            )
        elif capacity == 1:
            suggestion = (
                f"当前 capacity=1 太小，无法积累突发。"
                f"建议 capacity >= {N}，令牌桶才能在满桶时瞬间放行足够请求。"
            )
        else:
            suggestion = (
                f"当前参数下未触发违规，可能是因为 capacity={capacity} 不够大"
                f"或 window={window_seconds}s 的充能不足以产生跨越窗口的令牌重叠。"
                f"建议: capacity >= {N} 且 window_seconds >= 1.0"
            )
    else:
        suggestion = None

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
            "is_violated": is_violated,
            "violation_ratio": worst_count / N if N > 0 else 0,
            "requests_in_worst_window": inside_summary,
            "request_times_sample": all_tb_times[:20],
        },
        "strict_sliding_window": {
            "total_allowed": len(ss_allowed_times),
            "worst_window_count": ss_worst_count,
            "violation": ss_worst_count > N,
        },
        "is_violated": is_violated,
        "suggestion": suggestion,
        "explanation": (
            f"令牌桶在 T=0 时因为桶是满的，立即通过了 capacity={capacity} 个请求；"
            f"经过 {window_seconds}s 的充能，又补充了约 {N} 个令牌；"
            f"因此在跨越 0 时刻的窗口内，合计通过了 {worst_count}"
            + (f" > {N} 个请求。" if is_violated else f" ≤ {N} 个请求，未超出限制。")
        ),
    }


def format_violation_report(report: Dict[str, Any]) -> str:
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

    if tb["is_violated"]:
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
    else:
        lines += [
            f"  ✓ 当前参数下未发现违规",
            f"    最坏窗口内请求数: {tb['worst_window_count']}",
            f"    限制 N = {p['N']}",
            f"    令牌桶在此参数下恰好满足严格语义",
        ]
        if report["suggestion"]:
            lines += [
                "",
                f"  💡 如需复现违规，可调整参数:",
                f"    {report['suggestion']}",
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
    按 Key 限流的管理器

    特性：
      - 每个 key 独立限流器实例
      - 支持三种后端
      - 自动清理长时间未活动的 key
      - 支持批量非交互模式
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

    def run_batch(
        self,
        requests: List[Tuple[float, str]],
        cleanup_check_times: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        非交互批量模式

        Args:
            requests: [(time, key), ...] 按时间排序的请求序列
            cleanup_check_times: 在这些时间点额外检查清理效果

        Returns:
            每个请求的决策、每个 key 的汇总、活跃 key 数变化、清理结果
        """
        results: List[Dict[str, Any]] = []
        key_stats: Dict[str, Dict[str, int]] = {}
        key_activity_timeline: List[Dict[str, Any]] = []
        cleanup_events: List[Dict[str, Any]] = []

        for t, key in requests:
            ok, count = self.allow(key, now=t)
            results.append({
                "time": t,
                "key": key,
                "allowed": ok,
                "count": count,
            })

            if key not in key_stats:
                key_stats[key] = {"allowed": 0, "rejected": 0}
            if ok:
                key_stats[key]["allowed"] += 1
            else:
                key_stats[key]["rejected"] += 1

            key_activity_timeline.append({
                "time": t,
                "active_keys": self.active_key_count,
                "event": f"key={key} {'ALLOW' if ok else 'REJECT'}",
            })

        if cleanup_check_times:
            for ct in cleanup_check_times:
                before = self.active_key_count
                cleaned = self.cleanup_expired(now=ct)
                cleanup_events.append({
                    "time": ct,
                    "active_keys_before": before,
                    "active_keys_after": self.active_key_count,
                    "cleaned_count": cleaned,
                })

        return {
            "results": results,
            "key_stats": key_stats,
            "key_activity_timeline": key_activity_timeline,
            "cleanup_events": cleanup_events,
            "final_active_key_count": self.active_key_count,
            "final_active_keys": self.get_active_keys(),
        }


def format_multikey_batch_report(report: Dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        "MultiKey 批量限流报告",
        "=" * 70,
        "",
        "  ──────────────────────────────────────────",
        "  逐请求结果:",
        f"  {'#':>4}  {'时间(s)':>10}  {'Key':>10}  {'结果':>6}  {'计数':>5}",
        "  " + "-" * 45,
    ]

    for idx, r in enumerate(report["results"]):
        status = "ALLOW" if r["allowed"] else "REJECT"
        lines.append(
            f"  {idx:>4}  {r['time']:>10.4f}  {r['key']:>10}  {status:>6}  {r['count']:>5}"
        )

    lines += [
        "  " + "-" * 45,
        "",
        "  ──────────────────────────────────────────",
        "  各 Key 汇总:",
    ]
    for key, stats in sorted(report["key_stats"].items()):
        total = stats["allowed"] + stats["rejected"]
        lines.append(
            f"    {key:>15}: 通过 {stats['allowed']:>3}/{total:<3}, "
            f"拒绝 {stats['rejected']:>3}"
        )

    lines += [
        "",
        "  ──────────────────────────────────────────",
        "  活跃 Key 数变化:",
    ]
    for entry in report["key_activity_timeline"]:
        lines.append(
            f"    t={entry['time']:>8.4f}s  active_keys={entry['active_keys']:>3}  {entry['event']}"
        )

    if report["cleanup_events"]:
        lines += [
            "",
            "  ──────────────────────────────────────────",
            "  清理事件:",
        ]
        for ce in report["cleanup_events"]:
            lines.append(
                f"    t={ce['time']:>8.4f}s  清理前={ce['active_keys_before']}  "
                f"清理后={ce['active_keys_after']}  清理数={ce['cleaned_count']}"
            )

    lines += [
        "",
        f"  最终活跃 key 数: {report['final_active_key_count']}",
        f"  最终活跃 keys: {report['final_active_keys']}",
        "=" * 70,
    ]
    return "\n".join(lines)


# ============================================================================
# 6. 压缩版 vs 严格版逐点差异对比
# ============================================================================

def compare_strict_vs_compressed(
    max_per_second: int,
    request_times: List[float],
    bucket_width_ms: int = 10,
    window_seconds: float = 1.0,
) -> Dict[str, Any]:
    strict = StrictSlidingWindowLimiter(max_per_second, window_seconds)
    comp = CompressedSlidingWindowLimiter(max_per_second, window_seconds, bucket_width_ms)

    diffs: List[Dict[str, Any]] = []
    strict_allowed: List[float] = []
    comp_allowed: List[float] = []
    false_positives = 0
    false_negatives = 0
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

    comp_err = CompressedSlidingWindowLimiter(max_per_second, window_seconds, bucket_width_ms)

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


# ============================================================================
# 7. 批量实验：多组请求序列 + 三种模式对比
# ============================================================================

def run_batch_experiment(
    groups: List[Dict[str, Any]],
    N: int,
    window_seconds: float = 1.0,
    bucket_width_ms: int = 10,
    token_bucket_capacity: Optional[int] = None,
) -> Dict[str, Any]:
    """
    批量实验：对每组请求时间序列，分别跑三种限流器

    Args:
        groups: [{"name": "场景1", "times": [0.0, 0.1, ...]}, ...]
        N: 限流阈值
        window_seconds: 窗口大小
        bucket_width_ms: 压缩版桶宽
        token_bucket_capacity: 令牌桶容量

    Returns:
        每组的通过数、拒绝数、最坏窗口、差异摘要
    """
    capacity = token_bucket_capacity if token_bucket_capacity else N
    group_results = []

    for group in groups:
        name = group.get("name", "unnamed")
        times = [float(t) for t in group["times"]]

        strict = StrictSlidingWindowLimiter(N, window_seconds)
        comp = CompressedSlidingWindowLimiter(N, window_seconds, bucket_width_ms)
        tb = TokenBucketLimiter(N, capacity=capacity)

        strict_allowed = []
        comp_allowed = []
        tb_allowed = []

        for t in times:
            ok_s, _ = strict.allow(now=t)
            ok_c, _ = comp.allow(now=t)
            ok_t, _ = tb.allow(now=t)
            if ok_s:
                strict_allowed.append(t)
            if ok_c:
                comp_allowed.append(t)
            if ok_t:
                tb_allowed.append(t)

        def worst_count(allowed_times: List[float]) -> Tuple[int, Tuple[float, float]]:
            worst = 0
            win = (0.0, 0.0)
            for t_end in allowed_times:
                t_start = t_end - window_seconds
                c = sum(1 for ts in allowed_times if t_start < ts <= t_end)
                if c > worst:
                    worst = c
                    win = (t_start, t_end)
            return worst, win

        s_worst, s_win = worst_count(strict_allowed)
        c_worst, c_win = worst_count(comp_allowed)
        t_worst, t_win = worst_count(tb_allowed)

        group_results.append({
            "name": name,
            "total_requests": len(times),
            "strict": {
                "allowed": len(strict_allowed),
                "rejected": len(times) - len(strict_allowed),
                "worst_window_count": s_worst,
                "worst_window": s_win,
                "violates": s_worst > N,
            },
            "compressed": {
                "allowed": len(comp_allowed),
                "rejected": len(times) - len(comp_allowed),
                "worst_window_count": c_worst,
                "worst_window": c_win,
                "violates": c_worst > N,
            },
            "token_bucket": {
                "allowed": len(tb_allowed),
                "rejected": len(times) - len(tb_allowed),
                "worst_window_count": t_worst,
                "worst_window": t_win,
                "violates": t_worst > N,
            },
        })

    return {
        "params": {
            "N": N,
            "window_seconds": window_seconds,
            "bucket_width_ms": bucket_width_ms,
            "token_bucket_capacity": capacity,
        },
        "groups": group_results,
    }


def format_batch_report(report: Dict[str, Any]) -> str:
    p = report["params"]

    lines = [
        "=" * 78,
        "批量实验报告 - 三种限流器对比",
        "=" * 78,
        f"  公共参数: N={p['N']}/s, 窗口={p['window_seconds']}s, "
        f"压缩桶宽={p['bucket_width_ms']}ms, 令牌桶容量={p['token_bucket_capacity']}",
        "",
    ]

    for g in report["groups"]:
        lines += [
            f"  ──────────────────────────────────────────────────────────────────",
            f"  场景: {g['name']}  (总请求={g['total_requests']})",
            f"  {'':>12} {'通过':>6} {'拒绝':>6} {'最坏窗口计数':>12} {'是否违规':>8}",
            f"  {'严格滑动窗口':>12} {g['strict']['allowed']:>6} {g['strict']['rejected']:>6} "
            f"{g['strict']['worst_window_count']:>12} {'❌' if g['strict']['violates'] else '✓':>8}",
            f"  {'压缩滑动窗口':>12} {g['compressed']['allowed']:>6} {g['compressed']['rejected']:>6} "
            f"{g['compressed']['worst_window_count']:>12} {'❌' if g['compressed']['violates'] else '✓':>8}",
            f"  {'令牌桶':>12} {g['token_bucket']['allowed']:>6} {g['token_bucket']['rejected']:>6} "
            f"{g['token_bucket']['worst_window_count']:>12} {'❌' if g['token_bucket']['violates'] else '✓':>8}",
            "",
        ]

        strict_vs_tb = g['token_bucket']['allowed'] - g['strict']['allowed']
        strict_vs_comp = g['compressed']['allowed'] - g['strict']['allowed']

        lines.append(f"    差异摘要:")
        if strict_vs_tb != 0:
            sign = "+" if strict_vs_tb > 0 else ""
            lines.append(
                f"      令牌桶 vs 严格: 通过数差 {sign}{strict_vs_tb}, "
                f"最坏窗口差 {g['token_bucket']['worst_window_count'] - g['strict']['worst_window_count']}"
            )
        else:
            lines.append(f"      令牌桶 vs 严格: 通过数相同")

        if strict_vs_comp != 0:
            sign = "+" if strict_vs_comp > 0 else ""
            lines.append(
                f"      压缩版 vs 严格: 通过数差 {sign}{strict_vs_comp}, "
                f"最坏窗口差 {g['compressed']['worst_window_count'] - g['strict']['worst_window_count']}"
            )
        else:
            lines.append(f"      压缩版 vs 严格: 通过数相同")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


def parse_batch_input(text: str) -> List[Dict[str, Any]]:
    """
    从文本解析多组请求序列

    格式：
      # 注释行
      场景名: 0.0 0.1 0.2 0.3 ...
      另一个场景: 0.0 0.05 0.1 ...

    或带 key 的格式（用于 multikey）：
      场景名:
        0.0 user_A
        0.1 user_B
        0.2 user_A
    """
    groups = []
    current_name = None
    current_times = []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line and not line[0].isdigit():
            if current_name is not None and current_times:
                groups.append({"name": current_name, "times": current_times})
            parts = line.split(":", 1)
            current_name = parts[0].strip()
            rest = parts[1].strip()
            current_times = []
            if rest:
                for token in rest.split():
                    try:
                        current_times.append(float(token))
                    except ValueError:
                        pass
        else:
            try:
                current_times.append(float(line.split()[0]))
            except (ValueError, IndexError):
                pass

    if current_name is not None and current_times:
        groups.append({"name": current_name, "times": current_times})

    return groups


def parse_batch_input_with_keys(text: str) -> List[Tuple[float, str]]:
    """
    从文本解析带 key 的请求序列

    格式（每行: 时间 key）：
      0.0 user_A
      0.1 user_B
      0.2 user_A
    """
    requests = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                t = float(parts[0])
                key = parts[1]
                requests.append((t, key))
            except ValueError:
                pass
    return requests
