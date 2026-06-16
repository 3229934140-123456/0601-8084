import time
import bisect
from collections import deque
from typing import Optional, Tuple


class StrictSlidingWindowLimiter:
    """
    严格语义的滑动窗口限流器

    保证：对于任意时刻 t，区间 (t - 1s, t] 内通过的请求数 <= N

    数据结构：使用 deque 记录每个通过请求的精确时间戳（单调递增队列）
    每次请求到来时：
      1. 弹出队首所有早于 (now - window_size) 的时间戳
      2. 若队列长度 < N，则记录当前时间戳并放行
      3. 否则拒绝
    """

    def __init__(self, max_per_second: int, window_seconds: float = 1.0):
        self.max_per_second = max_per_second
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def allow(self, now: Optional[float] = None) -> Tuple[bool, int]:
        """
        返回 (是否允许通过, 当前窗口内已有请求数)
        """
        if now is None:
            now = time.monotonic()

        self._evict_expired(now)

        current_count = len(self._timestamps)
        if current_count < self.max_per_second:
            self._timestamps.append(now)
            return True, current_count + 1
        return False, current_count

    def reset(self) -> None:
        self._timestamps.clear()

    @property
    def state_size(self) -> int:
        return len(self._timestamps)


class TokenBucketLimiter:
    """
    标准令牌桶

    令牌以 rate_per_second 的速率产生，桶容量为 capacity
    每次请求消耗 1 个令牌，有则通过，无则拒绝

    注意：这是标准实现，但在突发场景下不满足"任意1秒窗口不超过N"的严格语义
    """

    def __init__(self, rate_per_second: int, capacity: Optional[int] = None):
        self.rate = rate_per_second
        self.capacity = capacity if capacity is not None else rate_per_second
        self._tokens = float(self.capacity)
        self._last_refill = None

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

        self._refill(now)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True, self._tokens
        return False, self._tokens

    def reset(self) -> None:
        self._tokens = float(self.capacity)
        self._last_refill = None

    @property
    def state_size(self) -> int:
        return 1


class CompressedSlidingWindowLimiter:
    """
    压缩状态的滑动窗口限流器（用于百万级 QPS 场景）

    核心思想：将时间轴划分为固定大小的微桶（micro-bucket），
    只记录每个微桶内的请求计数，而非逐请求记录时间戳。

    设：
      - W = 窗口大小（1秒）
      - B = 微桶宽度（例如 1ms、10ms）
      - K = W / B = 微桶总数

    代价分析：
      - 内存：O(K) = O(W/B)，与 N 无关！1ms 分桶只需 1000 个计数槽
      - 精度：最大误差 <= 1个微桶的边界请求被错误计入/排除
      - 最坏情况：在微桶边界附近，实际通过数可能比 N 多出 (B-1)/W 的比例

    当 B=1ms 时：
      - 1000 个计数槽，每槽 4 字节 = 4KB 内存（对比逐记录需 ~8MB）
      - 最大误差 < 0.1%（对 N=1e6 来说最多多放约 1000 个请求在最坏边界）

    当 B=10ms 时：
      - 100 个计数槽 = 400 字节
      - 最大误差约 1%
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

        self.num_buckets = int(window_seconds * 1000 / bucket_width_ms) + 2

        self._counts: list = [0] * self.num_buckets
        self._bucket_ids: list = [0] * self.num_buckets
        self._last_bucket_idx = 0

    def _get_bucket_id(self, now: float) -> int:
        return int(now / self.bucket_width_sec)

    def _update_window(self, now: float) -> int:
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

        current_count, current_bucket_id = self._update_window(now)

        if current_count < self.max_per_second:
            idx = current_bucket_id % self.num_buckets
            if self._bucket_ids[idx] != current_bucket_id:
                self._bucket_ids[idx] = current_bucket_id
                self._counts[idx] = 0
            self._counts[idx] += 1
            return True, current_count + 1

        return False, current_count

    def reset(self) -> None:
        for i in range(self.num_buckets):
            self._counts[i] = 0
            self._bucket_ids[i] = 0
        self._last_bucket_idx = 0

    @property
    def state_size(self) -> int:
        return self.num_buckets

    def memory_bytes(self) -> int:
        return self.num_buckets * (8 + 8)


def analyze_token_bucket_violation(
    rate_per_second: int,
    capacity: Optional[int] = None,
) -> dict:
    """
    分析令牌桶在突发场景下违反严格语义的情况

    场景：
      T=0 时桶是满的（capacity 个令牌）
      在 T=0 时刻突发通过 capacity 个请求（瞬间）
      然后在 T=0+ε（ε→0）到 T=1 之间，令牌桶又产生了 rate 个令牌
      那么在窗口 (0, 1] 内总共通过了 capacity + rate 个请求！

    当 capacity = rate 时：
      窗口内通过了 2*rate 个请求，是严格限制的 2 倍！
    """
    if capacity is None:
        capacity = rate_per_second

    result = {}

    tb = TokenBucketLimiter(rate_per_second, capacity=capacity)

    T0 = 0.0
    burst_count = 0
    for _ in range(capacity * 3):
        ok, _ = tb.allow(now=T0)
        if ok:
            burst_count += 1
        else:
            break

    result["instant_burst_at_T0"] = burst_count

    T1 = 1.0 - 1e-9
    additional_count = 0
    for _ in range(rate_per_second * 3):
        ok, _ = tb.allow(now=T1)
        if ok:
            additional_count += 1
        else:
            break

    result["additional_before_T1"] = additional_count
    result["total_in_window_0_to_1"] = burst_count + additional_count
    result["strict_limit"] = rate_per_second
    result["violation_ratio"] = (burst_count + additional_count) / rate_per_second

    tb2 = TokenBucketLimiter(rate_per_second, capacity=capacity)
    T2 = 1.0
    count_after_1s = 0
    for _ in range(rate_per_second * 3):
        ok, _ = tb2.allow(now=T2)
        if ok:
            count_after_1s += 1
        else:
            break

    result["at_exactly_T1_can_do"] = count_after_1s

    return result
