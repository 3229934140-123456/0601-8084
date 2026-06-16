import sys
import random
from sliding_window import (
    StrictSlidingWindowLimiter,
    TokenBucketLimiter,
    CompressedSlidingWindowLimiter,
    MultiKeyRateLimiter,
    analyze_token_bucket_violation_strict,
    format_violation_report,
    compare_strict_vs_compressed,
    format_comparison_report,
)


def test_1_token_bucket_violation_strict():
    """测试1：令牌桶违规分析（严格左开右闭窗口）"""
    print("=" * 70)
    print("测试1：令牌桶违规演示（严格左开右闭窗口）")
    print("=" * 70)

    N = 100
    report = analyze_token_bucket_violation_strict(
        rate_per_second=N,
        capacity=N,
        window_seconds=1.0,
    )
    print(format_violation_report(report))

    tb = report["token_bucket"]
    ss = report["strict_sliding_window"]

    assert tb["worst_window_count"] > N, (
        f"令牌桶应违规但未违规: 最坏窗口计数={tb['worst_window_count']}, N={N}"
    )
    assert ss["worst_window_count"] <= N, (
        f"严格滑动窗口不应违规: 最坏窗口计数={ss['worst_window_count']}, N={N}"
    )
    assert tb["worst_window"] is not None

    ws, we = tb["worst_window"]
    assert ws < we, "窗口左端点应小于右端点"
    print(f"  ✓ 违规窗口合法: ({ws:.6f}, {we:.6f}]")
    print(f"  ✓ 令牌桶违规: 窗口内 {tb['worst_window_count']} > N={N}")
    print(f"  ✓ 严格滑动窗口合规: 最坏窗口 {ss['worst_window_count']} <= N={N}")
    print()

    return True


def test_2_multi_key_limiter():
    """测试2：按 Key 限流管理器 + 自动过期清理"""
    print("=" * 70)
    print("测试2：MultiKeyRateLimiter - 按 key 独立限流 + 自动清理")
    print("=" * 70)

    N = 5
    mgr = MultiKeyRateLimiter(
        max_per_second=N,
        window_seconds=1.0,
        limiter_type="strict",
        idle_ttl_seconds=5.0,
        cleanup_interval_seconds=0.0,
    )

    # Part A: 各 key 独立限流
    print("\n  Part A: 各 key 独立限流验证")
    base = 0.0

    for i in range(N):
        for key in ["user_A", "user_B", "user_C"]:
            ok, cnt = mgr.allow(key, now=base + i * 0.01)
            assert ok, f"key={key} 第{i}个请求应通过"

    print(f"    3 个用户各发 {N} 个请求，全部通过")
    assert mgr.active_key_count == 3
    print(f"    活跃 key 数 = {mgr.active_key_count} (应为 3) ✓")

    for key in ["user_A", "user_B", "user_C"]:
        ok, cnt = mgr.allow(key, now=base + 0.1)
        assert not ok, f"key={key} 第{N+1}个请求应被拒绝"
        print(f"    key={key} 第{N+1}个请求: REJECT ✓ (count={cnt})")

    # Part B: key 自动过期清理
    print("\n  Part B: 自动过期清理")

    cleaned = mgr.cleanup_expired(now=base + 10.0)
    print(f"    推进时间到 T={base + 10.0}s，强制清理 (idle_ttl=5s)")
    print(f"    被清理的 key 数: {cleaned} (应为 3)")
    assert cleaned == 3
    assert mgr.active_key_count == 0
    print(f"    清理后活跃 key 数 = {mgr.active_key_count} (应为 0) ✓")

    # Part C: 惰性清理
    print("\n  Part C: 惰性清理 (在 allow 调用时触发)")
    mgr2 = MultiKeyRateLimiter(
        max_per_second=10,
        idle_ttl_seconds=2.0,
        cleanup_interval_seconds=0.0,
    )

    mgr2.allow("ip_1", now=0.0)
    mgr2.allow("ip_2", now=0.0)
    mgr2.allow("ip_3", now=0.5)
    print(f"    注册了 3 个 IP, 活跃数={mgr2.active_key_count}")

    mgr2.allow("ip_3", now=3.0)
    print(f"    T=3s 时只有 ip_3 活跃, 惰性清理触发...")
    print(f"    活跃数={mgr2.active_key_count} (应为 1: ip_1/ip_2 都已过期)")
    assert mgr2.active_key_count == 1
    assert "ip_3" in mgr2.get_active_keys()
    print(f"    活跃 keys = {mgr2.get_active_keys()} ✓")

    # Part D: 不同 limiter 类型
    print("\n  Part D: 三种后端类型")
    for ltype in ["strict", "compressed", "token-bucket"]:
        m = MultiKeyRateLimiter(max_per_second=10, limiter_type=ltype)
        ok, _ = m.allow("test_key", now=0.0)
        assert ok
        print(f"    limiter_type='{ltype}': allow() 正常 ✓")

    print("\n  PASSED: MultiKeyRateLimiter 全部通过 ✓\n")
    return True


def test_3_compressed_error_report():
    """测试3：压缩版误差报告 + 逐请求对比"""
    print("=" * 70)
    print("测试3：压缩版误差报告 & 严格版 vs 压缩版逐点差异")
    print("=" * 70)

    N = 500
    bucket_ms = 10

    # Part A: 误差报告
    print("\n  Part A: 误差报告输出")
    comp = CompressedSlidingWindowLimiter(
        max_per_second=N, window_seconds=1.0, bucket_width_ms=bucket_ms
    )
    report = comp.error_report()
    print(comp.format_error_report())

    assert report["params"]["N (max_per_second)"] == N
    assert report["params"]["bucket_width_ms"] == bucket_ms
    assert report["precision"]["worst_case_extra_allowed"] > 0
    print(f"  ✓ 误差报告字段完整: 最坏多放行={report['precision']['worst_case_extra_allowed']}")
    print(f"  ✓ 节省内存比例: {report['memory']['savings_ratio']:.1f}%")

    # Part B: 逐请求对比 - 构造桶边界处的请求序列
    print("\n  Part B: 逐请求决策对比 (桶边界附近触发误差)")
    random.seed(42)

    request_times = []
    t = 0.0
    for i in range(3000):
        t += random.uniform(0.0005, 0.003)
        request_times.append(t)

    cmp_report = compare_strict_vs_compressed(
        max_per_second=100,
        request_times=request_times,
        bucket_width_ms=10,
        window_seconds=1.0,
    )
    print(format_comparison_report(cmp_report))

    s = cmp_report["summary"]
    print(f"\n  ✓ 决策一致率: {s['match_ratio']:.1f}%")
    print(f"  ✓ 压缩版多放行: {s['false_positives (多放行)']}, 多拒绝: {s['false_negatives (多拒绝)']}")

    diff_total = s['false_positives (多放行)'] + s['false_negatives (多拒绝)']
    expected_max_diff = cmp_report["params"]["total_requests"] * 0.1
    assert diff_total <= expected_max_diff, (
        f"差异过多: {diff_total} > {expected_max_diff}"
    )
    print(f"  ✓ 差异总数在合理范围内 (< 10%): {diff_total}")

    # Part C: 不同桶宽的误差对比
    print("\n  Part C: 不同桶宽的误差特性对比 (N=100)")
    for bw in [1, 5, 10, 50]:
        c = CompressedSlidingWindowLimiter(100, 1.0, bw)
        r = c.error_report()
        print(
            f"    桶宽={bw:>3}ms: "
            f"多放行≤{r['precision']['worst_case_extra_allowed']:>3}, "
            f"误差≤{r['precision']['relative_error_pct_max']:>5.2f}%, "
            f"内存={r['memory']['compressed_bytes']:>5}B, "
            f"适用: {r['recommendation']['推荐QPS范围']}"
        )

    print("\n  PASSED: 压缩版误差分析全部通过 ✓\n")
    return True


def test_4_strict_sliding_window_verification():
    """测试4：严格滑动窗口语义验证（暴力枚举所有窗口）"""
    print("=" * 70)
    print("测试4：严格滑动窗口语义 - 暴力验证所有滑动窗口")
    print("=" * 70)

    N = 10
    limiter = StrictSlidingWindowLimiter(max_per_second=N, window_seconds=1.0)

    random.seed(123)
    all_allowed: list = []

    t = 0.0
    for i in range(500):
        t += random.uniform(0.01, 0.2)
        ok, cnt = limiter.allow(now=t)
        if ok:
            all_allowed.append(t)

    print(f"  总请求: 500, 通过: {len(all_allowed)}")

    violations = 0
    max_count = 0
    for i, t_end in enumerate(all_allowed):
        t_start = t_end - 1.0
        count = sum(1 for ts in all_allowed if t_start < ts <= t_end)
        max_count = max(max_count, count)
        if count > N:
            violations += 1
            print(f"  ❌ 违规: t={t_end:.3f}s 窗口内有 {count} 个请求 (限 {N})")

    print(f"  任意窗口最大请求数: {max_count} (限制 N={N})")
    assert violations == 0, f"发现 {violations} 处违规"
    assert max_count <= N
    print(f"  ✓ 所有窗口均合规, 最大计数={max_count} <= N={N}")

    print("\n  PASSED: 严格滑动窗口语义验证通过 ✓\n")
    return True


def main():
    print("\n" + "#" * 70)
    print("#  严格滑动窗口限流器 v2 - 完整测试")
    print("#" * 70 + "\n")

    tests = [
        ("令牌桶违规严格演示", test_1_token_bucket_violation_strict),
        ("MultiKey 限流管理器", test_2_multi_key_limiter),
        ("压缩版误差报告与对比", test_3_compressed_error_report),
        ("严格语义暴力验证", test_4_strict_sliding_window_verification),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            passed = fn()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ❌ 测试 [{name}] 异常: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "#" * 70)
    if all_passed:
        print("#  所有测试通过 ✓")
    else:
        print("#  部分测试失败 ❌")
    print("#" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
