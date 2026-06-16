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
    recommend_bucket_width,
    format_recommend_report,
    run_batch_experiment,
    format_batch_report,
    parse_batch_input,
    parse_batch_input_with_keys,
    format_multikey_batch_report,
)


def test_1_token_bucket_violation_strict():
    print("=" * 70)
    print("测试1：令牌桶违规演示（严格左开右闭窗口）")
    print("=" * 70)

    N = 100
    report = analyze_token_bucket_violation_strict(
        rate_per_second=N, capacity=N, window_seconds=1.0
    )
    print(format_violation_report(report))

    tb = report["token_bucket"]
    ss = report["strict_sliding_window"]

    assert tb["is_violated"], f"令牌桶应违规但未违规: 最坏窗口={tb['worst_window_count']}"
    assert not ss["violation"], f"严格滑动窗口不应违规: 最坏窗口={ss['worst_window_count']}"
    assert tb["worst_window_count"] > N
    print(f"  ✓ 令牌桶违规: {tb['worst_window_count']} > N={N}")
    print(f"  ✓ 严格滑动窗口合规: {ss['worst_window_count']} <= N={N}")
    print()
    return True


def test_2_token_bucket_no_violation():
    print("=" * 70)
    print("测试2：令牌桶参数不足时的提示（N=10, capacity=1 不触发违规）")
    print("=" * 70)

    report = analyze_token_bucket_violation_strict(
        rate_per_second=10, capacity=1, window_seconds=1.0
    )
    print(format_violation_report(report))

    assert not report["is_violated"], "capacity=1 不应触发违规"
    assert report["suggestion"] is not None, "未违规时应给出建议"
    assert "capacity" in report["suggestion"]
    print(f"  ✓ 未违规时正确提示，建议: {report['suggestion'][:60]}...")
    print()
    return True


def test_3_multi_key_limiter():
    print("=" * 70)
    print("测试3：MultiKeyRateLimiter")
    print("=" * 70)

    N = 5
    mgr = MultiKeyRateLimiter(
        max_per_second=N, window_seconds=1.0, limiter_type="strict",
        idle_ttl_seconds=5.0, cleanup_interval_seconds=0.0,
    )

    for i in range(N):
        for key in ["user_A", "user_B"]:
            ok, _ = mgr.allow(key, now=i * 0.01)
            assert ok

    for key in ["user_A", "user_B"]:
        ok, _ = mgr.allow(key, now=0.1)
        assert not ok

    assert mgr.active_key_count == 2
    print(f"  ✓ 各 key 独立限流正确, 活跃数={mgr.active_key_count}")

    cleaned = mgr.cleanup_expired(now=10.0)
    assert cleaned == 2
    assert mgr.active_key_count == 0
    print(f"  ✓ 过期清理正确, 清理={cleaned}, 活跃数={mgr.active_key_count}")

    for ltype in ["strict", "compressed", "token-bucket"]:
        m = MultiKeyRateLimiter(max_per_second=10, limiter_type=ltype)
        ok, _ = m.allow("test", now=0.0)
        assert ok
    print(f"  ✓ 三种后端均正常")
    print()
    return True


def test_4_multikey_batch():
    print("=" * 70)
    print("测试4：MultiKey 非交互批量模式")
    print("=" * 70)

    mgr = MultiKeyRateLimiter(
        max_per_second=3, window_seconds=1.0, limiter_type="strict",
        idle_ttl_seconds=2.0, cleanup_interval_seconds=0.0,
    )

    requests = [
        (0.0, "ip_1"),
        (0.1, "ip_1"),
        (0.2, "ip_1"),
        (0.3, "ip_1"),
        (0.0, "ip_2"),
        (0.1, "ip_2"),
        (0.2, "ip_2"),
        (0.3, "ip_2"),
        (0.0, "ip_3"),
    ]

    report = mgr.run_batch(requests, cleanup_check_times=[5.0])
    print(format_multikey_batch_report(report))

    assert len(report["results"]) == 9
    assert "ip_1" in report["key_stats"]
    assert "ip_2" in report["key_stats"]
    assert "ip_3" in report["key_stats"]

    assert report["key_stats"]["ip_1"]["allowed"] == 3
    assert report["key_stats"]["ip_1"]["rejected"] == 1
    assert report["key_stats"]["ip_2"]["allowed"] == 3
    assert report["key_stats"]["ip_2"]["rejected"] == 1
    assert report["key_stats"]["ip_3"]["allowed"] == 1
    assert report["key_stats"]["ip_3"]["rejected"] == 0

    print(f"  ✓ ip_1: 通过={report['key_stats']['ip_1']['allowed']}, 拒绝={report['key_stats']['ip_1']['rejected']}")
    print(f"  ✓ ip_2: 通过={report['key_stats']['ip_2']['allowed']}, 拒绝={report['key_stats']['ip_2']['rejected']}")
    print(f"  ✓ ip_3: 通过={report['key_stats']['ip_3']['allowed']}, 拒绝={report['key_stats']['ip_3']['rejected']}")

    assert len(report["cleanup_events"]) == 1
    assert report["cleanup_events"][0]["cleaned_count"] == 3
    print(f"  ✓ 清理事件: T=5.0s 清理了 {report['cleanup_events'][0]['cleaned_count']} 个过期 key")
    print()
    return True


def test_5_compressed_error_report_low_N():
    print("=" * 70)
    print("测试5：压缩版误差报告 - 低阈值场景 (N=1, N=5)")
    print("=" * 70)

    # N=1, B=1ms: 每桶平均 0.001 个请求，但误差至少 1
    comp1 = CompressedSlidingWindowLimiter(max_per_second=1, window_seconds=1.0, bucket_width_ms=1)
    report1 = comp1.error_report()
    print(comp1.format_error_report())

    assert report1["precision"]["worst_case_extra_allowed"] >= 1, (
        f"N=1 时最坏多放行应至少 1, 实际={report1['precision']['worst_case_extra_allowed']}"
    )
    assert report1["precision"]["relative_error_pct_max"] == 100.0, (
        f"N=1 时相对误差应为 100%, 实际={report1['precision']['relative_error_pct_max']}"
    )
    print(f"  ✓ N=1, B=1ms: 最坏多放行={report1['precision']['worst_case_extra_allowed']}, "
          f"误差={report1['precision']['relative_error_pct_max']:.1f}%")

    # N=5, B=10ms: 每桶平均 0.05 个，误差至少 1
    comp2 = CompressedSlidingWindowLimiter(max_per_second=5, window_seconds=1.0, bucket_width_ms=10)
    report2 = comp2.error_report()
    print(comp2.format_error_report())

    assert report2["precision"]["worst_case_extra_allowed"] >= 1
    assert report2["precision"]["relative_error_pct_max"] == 20.0
    print(f"  ✓ N=5, B=10ms: 最坏多放行={report2['precision']['worst_case_extra_allowed']}, "
          f"误差={report2['precision']['relative_error_pct_max']:.1f}%")

    # N=100, B=10ms: 每桶平均 1 个，误差为 1
    comp3 = CompressedSlidingWindowLimiter(max_per_second=100, window_seconds=1.0, bucket_width_ms=10)
    report3 = comp3.error_report()
    assert report3["precision"]["worst_case_extra_allowed"] == 1
    assert report3["precision"]["relative_error_pct_max"] == 1.0
    print(f"  ✓ N=100, B=10ms: 最坏多放行={report3['precision']['worst_case_extra_allowed']}, "
          f"误差={report3['precision']['relative_error_pct_max']:.1f}%")

    print()
    return True


def test_6_recommend_bucket_width():
    print("=" * 70)
    print("测试6：按目标误差反推桶宽")
    print("=" * 70)

    # 可行场景: N=1e6, 目标误差 0.1%
    rec1 = recommend_bucket_width(max_per_second=1_000_000, target_error_pct=0.1)
    print(format_recommend_report(rec1))

    assert rec1["feasible"]
    assert len(rec1["plans"]) > 0
    for plan in rec1["plans"]:
        assert plan["actual_error_pct"] <= 0.1
    print(f"  ✓ N=1e6, 目标0.1%: 找到 {len(rec1['plans'])} 个方案")

    # 不可行场景: N=1, 目标误差 0.01%
    rec2 = recommend_bucket_width(max_per_second=1, target_error_pct=0.01)
    print(format_recommend_report(rec2))

    assert not rec2["feasible"]
    assert rec2["min_possible_error_pct"] == 100.0
    print(f"  ✓ N=1, 目标0.01%: 正确判定不可行 (下界={rec2['min_possible_error_pct']}%)")

    # N=100, 目标误差 5%
    rec3 = recommend_bucket_width(max_per_second=100, target_error_pct=5.0)
    print(format_recommend_report(rec3))

    assert rec3["feasible"]
    print(f"  ✓ N=100, 目标5%: 找到 {len(rec3['plans'])} 个方案")
    for p in rec3["plans"]:
        print(f"    桶宽={p['bucket_width_ms']}ms, 误差={p['actual_error_pct']:.3f}%, {p['style']}")

    print()
    return True


def test_7_batch_experiment():
    print("=" * 70)
    print("测试7：批量实验 - 三种模式对比")
    print("=" * 70)

    groups = [
        {
            "name": "均匀低频",
            "times": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        },
        {
            "name": "突发集中",
            "times": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        },
    ]

    report = run_batch_experiment(groups, N=5, window_seconds=1.0, bucket_width_ms=10)
    print(format_batch_report(report))

    assert len(report["groups"]) == 2
    for g in report["groups"]:
        assert "strict" in g
        assert "compressed" in g
        assert "token_bucket" in g
        assert not g["strict"]["violates"], f"严格版不应违规: {g['name']}"
    print(f"  ✓ 两组场景均通过, 严格版无违规")
    print()
    return True


def test_8_parse_batch_input():
    print("=" * 70)
    print("测试8：解析批量输入格式")
    print("=" * 70)

    text = """
# 这是注释
场景A: 0.0 0.1 0.2 0.3 0.4 0.5
场景B: 0.0 0.01 0.02 0.03
"""
    groups = parse_batch_input(text)
    assert len(groups) == 2
    assert groups[0]["name"] == "场景A"
    assert len(groups[0]["times"]) == 6
    assert groups[1]["name"] == "场景B"
    assert len(groups[1]["times"]) == 4
    print(f"  ✓ 解析出 {len(groups)} 组场景")

    key_text = """
0.0 user_A
0.1 user_B
0.2 user_A
0.3 user_B
"""
    requests = parse_batch_input_with_keys(key_text)
    assert len(requests) == 4
    assert requests[0] == (0.0, "user_A")
    assert requests[3] == (0.3, "user_B")
    print(f"  ✓ 解析出 {len(requests)} 条 key 请求")
    print()
    return True


def test_9_strict_semantics_verification():
    print("=" * 70)
    print("测试9：严格滑动窗口语义 - 暴力验证")
    print("=" * 70)

    N = 10
    limiter = StrictSlidingWindowLimiter(max_per_second=N, window_seconds=1.0)

    random.seed(123)
    all_allowed: list = []

    t = 0.0
    for i in range(500):
        t += random.uniform(0.01, 0.2)
        ok, _ = limiter.allow(now=t)
        if ok:
            all_allowed.append(t)

    violations = 0
    max_count = 0
    for t_end in all_allowed:
        t_start = t_end - 1.0
        count = sum(1 for ts in all_allowed if t_start < ts <= t_end)
        max_count = max(max_count, count)
        if count > N:
            violations += 1

    print(f"  总请求: 500, 通过: {len(all_allowed)}, 最大窗口计数: {max_count}")
    assert violations == 0
    assert max_count <= N
    print(f"  ✓ 所有窗口均合规, 最大计数={max_count} <= N={N}")
    print()
    return True


def test_10_compare_strict_vs_compressed():
    print("=" * 70)
    print("测试10：压缩版 vs 严格版逐点差异")
    print("=" * 70)

    random.seed(42)
    request_times = []
    t = 0.0
    for i in range(2000):
        t += random.uniform(0.0005, 0.003)
        request_times.append(t)

    report = compare_strict_vs_compressed(
        max_per_second=100, request_times=request_times, bucket_width_ms=10
    )
    print(format_comparison_report(report))

    s = report["summary"]
    assert s["match_ratio"] > 80, f"决策一致率过低: {s['match_ratio']}%"
    print(f"  ✓ 决策一致率: {s['match_ratio']:.1f}%")
    print(f"  ✓ 多放行: {s['false_positives (多放行)']}, 多拒绝: {s['false_negatives (多拒绝)']}")
    print()
    return True


def main():
    print("\n" + "#" * 70)
    print("#  严格滑动窗口限流器 v3 - 完整测试")
    print("#" * 70 + "\n")

    tests = [
        ("令牌桶违规严格演示", test_1_token_bucket_violation_strict),
        ("令牌桶无违规提示", test_2_token_bucket_no_violation),
        ("MultiKey 限流管理器", test_3_multi_key_limiter),
        ("MultiKey 批量模式", test_4_multikey_batch),
        ("压缩版低阈值误差", test_5_compressed_error_report_low_N),
        ("按误差反推桶宽", test_6_recommend_bucket_width),
        ("批量实验对比", test_7_batch_experiment),
        ("解析输入格式", test_8_parse_batch_input),
        ("严格语义验证", test_9_strict_semantics_verification),
        ("压缩版差异对比", test_10_compare_strict_vs_compressed),
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
