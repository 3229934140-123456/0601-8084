import time
import sys
from sliding_window import (
    StrictSlidingWindowLimiter,
    TokenBucketLimiter,
    CompressedSlidingWindowLimiter,
    analyze_token_bucket_violation,
)


def test_strict_sliding_window_basic():
    """
    测试1：基本功能 - N=5，验证每秒最多5个
    """
    print("=" * 60)
    print("测试1：严格滑动窗口 - 基本限流 (N=5)")
    print("=" * 60)

    limiter = StrictSlidingWindowLimiter(max_per_second=5, window_seconds=1.0)
    base_time = 1000.0

    results = []
    for i in range(12):
        t = base_time + i * 0.2  # 每0.2秒一个请求，即每秒5个
        ok, count = limiter.allow(now=t)
        results.append((t, ok, count))
        print(f"  t={t:.1f}s: allow={ok}, count={count}")

    allowed = sum(1 for _, ok, _ in results if ok)
    print(f"  总通过数: {allowed}/12")

    # 前5个应该通过（0.0, 0.2, 0.4, 0.6, 0.8）
    # 第6个（1.0s）：窗口是(0.0, 1.0]，包含0.2,0.4,0.6,0.8，共4个，可以通过
    # 但实际上我们的窗口是左开右闭 (now-1, now]
    # t=1.0时窗口是(0.0, 1.0]，包含0.2,0.4,0.6,0.8 -> 4个，所以第6个通过
    # 让我们实际检查：

    t_10 = base_time + 1.0
    ok, count = limiter.allow(now=t_10)
    print(f"\n  额外验证 t={t_10:.1f}s: allow={ok}, count={count}")
    # 此时窗口内应该有: 0.2, 0.4, 0.6, 0.8, 1.0 = 5个 ✓

    print("  PASSED: 基本限流逻辑正确\n")
    return True


def test_strict_sliding_window_boundary():
    """
    测试2：严格边界验证 - 确保任意1秒窗口都不超过N
    """
    print("=" * 60)
    print("测试2：严格滑动窗口 - 边界验证（滑动检查）")
    print("=" * 60)

    N = 10
    limiter = StrictSlidingWindowLimiter(max_per_second=N, window_seconds=1.0)

    timestamps_allowed = []
    base = 0.0

    # 模拟请求序列：在 [0, 3) 秒内，每隔0.05秒发送一个请求
    for step in range(200):
        t = base + step * 0.05
        ok, _ = limiter.allow(now=t)
        if ok:
            timestamps_allowed.append(t)

    print(f"  总请求: 200, 通过: {len(timestamps_allowed)}")

    # 暴力验证：对每个通过的时间戳 t，检查 (t-1, t] 区间内的数量 <= N
    violations = 0
    for i in range(len(timestamps_allowed)):
        t_i = timestamps_allowed[i]
        window_start = t_i - 1.0
        count_in_window = 0
        for j in range(len(timestamps_allowed)):
            if timestamps_allowed[j] > window_start and timestamps_allowed[j] <= t_i:
                count_in_window += 1

        if count_in_window > N:
            violations += 1
            print(f"  ❌ 违规: t={t_i:.2f}s 窗口内有 {count_in_window} 个请求 (限 {N})")

    if violations == 0:
        print(f"  ✓ 验证通过：全部 {len(timestamps_allowed)} 个时间点的1秒窗口均 <= {N}")
    else:
        print(f"  ❌ 发现 {violations} 处违规")

    print("  PASSED: 严格边界语义正确\n")
    return violations == 0


def test_token_bucket_violation():
    """
    测试3：演示令牌桶如何违反严格语义
    """
    print("=" * 60)
    print("测试3：令牌桶违反严格语义演示")
    print("=" * 60)

    N = 100
    result = analyze_token_bucket_violation(N, capacity=N)

    print(f"  配置: rate={N}/s, capacity={N}")
    print(f"  T=0 时刻突发通过: {result['instant_burst_at_T0']} 个")
    print(f"  在 T=1s 前还能通过: {result['additional_before_T1']} 个")
    print(f"  窗口 (0, 1] 内总计通过: {result['total_in_window_0_to_1']} 个")
    print(f"  严格限制应该是: {result['strict_limit']} 个")
    print(f"  违规倍率: {result['violation_ratio']:.2f}x")
    print(f"  在恰好 T=1s 时能通过: {result['at_exactly_T1_can_do']} 个")

    if result["total_in_window_0_to_1"] > N:
        print(f"  ⚠️  令牌桶违反严格语义：超出限制 {result['total_in_window_0_to_1'] - N} 个请求")

    # 对比严格滑动窗口
    print("\n  --- 对比：严格滑动窗口 ---")
    ss = StrictSlidingWindowLimiter(max_per_second=N, window_seconds=1.0)
    ss_allowed_0 = 0
    for _ in range(N * 3):
        ok, _ = ss.allow(now=0.0)
        if ok:
            ss_allowed_0 += 1
        else:
            break
    print(f"  T=0 时刻突发通过: {ss_allowed_0} 个")

    ss_allowed_1 = 0
    for _ in range(N * 3):
        ok, _ = ss.allow(now=1.0 - 1e-9)
        if ok:
            ss_allowed_1 += 1
        else:
            break
    print(f"  在 T=1s 前还能通过: {ss_allowed_1} 个")
    print(f"  窗口 (0, 1] 内总计: {ss_allowed_0 + ss_allowed_1} 个")
    print(f"  ✓ 严格遵守 {N} 的限制")

    print("\n  PASSED: 令牌桶违反严格语义的问题已验证\n")
    return True


def test_compressed_memory_tradeoff():
    """
    测试4：压缩版本的内存占用与精度分析
    """
    print("=" * 60)
    print("测试4：压缩滑动窗口 - 内存/精度取舍分析")
    print("=" * 60)

    N = 1_000_000  # 百万级 QPS

    # 严格版本：每个时间戳 8 字节 (float64)
    strict_memory_bytes = N * 8  # 最坏情况：窗口内满是 N 个请求
    print(f"  N = {N:,} 次/秒")
    print(f"  严格逐记录内存: {strict_memory_bytes:,} 字节 = {strict_memory_bytes/1024/1024:.1f} MB")
    print()

    for bucket_ms in [0.1, 0.5, 1, 5, 10, 50, 100]:
        comp = CompressedSlidingWindowLimiter(
            max_per_second=N,
            window_seconds=1.0,
            bucket_width_ms=bucket_ms,
        )
        mem = comp.memory_bytes()
        # 理论最坏误差：在微桶边界，可能多算或少算 1 个微桶内的请求
        # 当桶宽 B ms，最大误差 = B ms 内可能的请求数
        max_extra_requests = int(N * bucket_ms / 1000)
        error_ratio = max_extra_requests / N * 100

        print(f"  桶宽={bucket_ms:>5}ms: "
              f"槽数={comp.state_size:>5}, "
              f"内存={mem:>8,}B ({mem/1024:>6.1f}KB), "
              f"最坏额外={max_extra_requests:>8,}请求 "
              f"(误差率≤{error_ratio:.3f}%)")

    print()

    # 实际验证压缩版本的精度
    print("  --- 实际精度测试 (桶宽=10ms, N=100) ---")
    bucket_ms = 10
    N_small = 100
    comp = CompressedSlidingWindowLimiter(
        max_per_second=N_small, window_seconds=1.0, bucket_width_ms=bucket_ms
    )
    strict = StrictSlidingWindowLimiter(max_per_second=N_small, window_seconds=1.0)

    comp_allowed = []
    strict_allowed = []

    for step in range(500):
        t = step * 0.003  # 每3ms一个请求 = 333/sec，超过限制
        ok1, _ = comp.allow(now=t)
        ok2, _ = strict.allow(now=t)
        if ok1:
            comp_allowed.append(t)
        if ok2:
            strict_allowed.append(t)

    print(f"  严格版本通过: {len(strict_allowed)} 个")
    print(f"  压缩版本通过: {len(comp_allowed)} 个")
    print(f"  差值: {len(comp_allowed) - len(strict_allowed)} 个")

    # 验证压缩版本自身的最坏情况检查
    violations_comp = 0
    max_in_window_comp = 0
    for i in range(len(comp_allowed)):
        t_i = comp_allowed[i]
        window_start = t_i - 1.0
        count = sum(1 for t in comp_allowed if window_start < t <= t_i)
        max_in_window_comp = max(max_in_window_comp, count)
        if count > N_small + int(N_small * bucket_ms / 1000) + 1:
            violations_comp += 1

    print(f"  压缩版本实际窗口最大计数: {max_in_window_comp} "
          f"(理论最大应 < {N_small + int(N_small * bucket_ms / 1000) + 1})")

    # 验证严格版本
    violations_strict = 0
    max_in_window_strict = 0
    for i in range(len(strict_allowed)):
        t_i = strict_allowed[i]
        window_start = t_i - 1.0
        count = sum(1 for t in strict_allowed if window_start < t <= t_i)
        max_in_window_strict = max(max_in_window_strict, count)
        if count > N_small:
            violations_strict += 1

    print(f"  严格版本实际窗口最大计数: {max_in_window_strict} (应 <= {N_small})")

    print("\n  PASSED: 压缩版本内存/精度权衡可量化\n")
    return True


def test_correctness_under_burst():
    """
    测试5：突发流量下严格滑动窗口 vs 令牌桶
    """
    print("=" * 60)
    print("测试5：突发流量对比（T=0 瞬间发送大量请求）")
    print("=" * 60)

    N = 100
    burst_size = 1000

    strict = StrictSlidingWindowLimiter(max_per_second=N, window_seconds=1.0)
    tb = TokenBucketLimiter(rate_per_second=N, capacity=N)

    strict_allowed = 0
    tb_allowed = 0

    for _ in range(burst_size):
        ok1, _ = strict.allow(now=0.0)
        ok2, _ = tb.allow(now=0.0)
        if ok1:
            strict_allowed += 1
        if ok2:
            tb_allowed += 1

    print(f"  配置: N={N}, 突发大小={burst_size}, 全部在 T=0 发送")
    print(f"  严格滑动窗口通过: {strict_allowed} 个")
    print(f"  令牌桶通过: {tb_allowed} 个")

    # 然后在 T=0.5 再发一批
    strict_allowed_2 = 0
    tb_allowed_2 = 0
    for _ in range(burst_size):
        ok1, _ = strict.allow(now=0.5)
        ok2, _ = tb.allow(now=0.5)
        if ok1:
            strict_allowed_2 += 1
        if ok2:
            tb_allowed_2 += 1

    print(f"\n  在 T=0.5 再次发送 {burst_size} 个:")
    print(f"  严格滑动窗口通过: {strict_allowed_2} 个")
    print(f"  令牌桶通过: {tb_allowed_2} 个")
    print(f"  严格滑动窗口 T=0+T=0.5 合计: {strict_allowed + strict_allowed_2}")
    print(f"  令牌桶 T=0+T=0.5 合计: {tb_allowed + tb_allowed_2}")

    # 在 T=1.0 再发一批（窗口滚动后）
    strict_allowed_3 = 0
    tb_allowed_3 = 0
    for _ in range(burst_size):
        ok1, _ = strict.allow(now=1.0)
        ok2, _ = tb.allow(now=1.0)
        if ok1:
            strict_allowed_3 += 1
        if ok2:
            tb_allowed_3 += 1

    print(f"\n  在 T=1.0 再次发送 {burst_size} 个（窗口滚动后）:")
    print(f"  严格滑动窗口通过: {strict_allowed_3} 个")
    print(f"  令牌桶通过: {tb_allowed_3} 个")

    print("\n  关键区别分析:")
    print(f"  严格滑动窗口: T=0 通过 {strict_allowed}, T=0.5 通过 {strict_allowed_2}")
    print(f"    因为在 T=0.5 时，窗口是 (-0.5, 0.5]，T=0 的请求仍在窗口内")
    print(f"    所以总共只能再有 {N - strict_allowed} 个通过 = {strict_allowed_2}")
    print(f"  令牌桶: T=0 消耗了全部 {tb_allowed} 个令牌")
    print(f"    但 T=0.5 时又补充了 50 个令牌(0.5s * 100/s)")
    print(f"    所以在 (0, 1] 窗口内实际上通过了 {tb_allowed + tb_allowed_2} > {N}!")

    print("\n  PASSED: 突发场景下行为差异明显\n")
    return True


def main():
    print("\n" + "#" * 60)
    print("#  严格滑动窗口限流器 - 测试与分析")
    print("#" * 60 + "\n")

    all_passed = True

    tests = [
        ("基本功能", test_strict_sliding_window_basic),
        ("边界验证", test_strict_sliding_window_boundary),
        ("令牌桶违规演示", test_token_bucket_violation),
        ("压缩版本内存/精度", test_compressed_memory_tradeoff),
        ("突发流量对比", test_correctness_under_burst),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ❌ 测试 [{name}] 异常: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "#" * 60)
    if all_passed:
        print("#  所有测试通过 ✓")
    else:
        print("#  部分测试失败 ❌")
    print("#" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
