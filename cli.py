import argparse
import sys
from typing import List

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


def cmd_run(args):
    N = args.N
    times = args.times

    if args.mode == "strict":
        limiter = StrictSlidingWindowLimiter(max_per_second=N, window_seconds=args.window)
        name = "严格滑动窗口"
    elif args.mode == "compressed":
        limiter = CompressedSlidingWindowLimiter(
            max_per_second=N, window_seconds=args.window, bucket_width_ms=args.bucket
        )
        name = f"压缩滑动窗口 (桶宽={args.bucket}ms)"
    elif args.mode == "token-bucket":
        cap = args.capacity if args.capacity else N
        limiter = TokenBucketLimiter(rate_per_second=N, capacity=cap)
        name = f"令牌桶 (rate={N}/s, capacity={cap})"
    else:
        print(f"未知模式: {args.mode}")
        return 1

    print("=" * 70)
    print(f"限流模式: {name}")
    print(f"N = {N}/s, 窗口 = {args.window}s")
    print(f"请求时间序列: {times}")
    print("=" * 70)
    print(f"  {'#':>4}  {'时间(s)':>12}  {'结果':>6}  {'窗口内计数':>10}")
    print("  " + "-" * 42)

    all_allowed_times = []
    for idx, t in enumerate(times):
        ok, info = limiter.allow(now=float(t))
        status = "ALLOW" if ok else "REJECT"
        if isinstance(info, float):
            info_str = f"tokens={info:.2f}"
        else:
            info_str = f"count={info}"
        print(f"  {idx:>4}  {float(t):>12.6f}  {status:>6}  {info_str:>16}")
        if ok:
            all_allowed_times.append(float(t))

    print("  " + "-" * 42)
    print(f"  总计: 通过 {len(all_allowed_times)}/{len(times)} 个请求")

    if all_allowed_times:
        worst_count = 0
        worst_win = None
        for t_end in all_allowed_times:
            t_start = t_end - args.window
            c = sum(1 for ts in all_allowed_times if t_start < ts <= t_end)
            if c > worst_count:
                worst_count = c
                worst_win = (t_start, t_end)
        print(f"  最坏窗口: {worst_win}, 窗口内计数={worst_count}, 限制N={N}")
        if worst_count > N:
            print(f"  ⚠️  警告: 该限流器违反严格语义（超出 {worst_count - N} 个）")
        else:
            print(f"  ✓ 语义合规")

    if args.mode == "compressed":
        print()
        print(limiter.format_error_report())

    return 0


def cmd_tb_violate(args):
    report = analyze_token_bucket_violation_strict(
        rate_per_second=args.N,
        capacity=args.capacity if args.capacity else args.N,
        window_seconds=args.window,
    )
    print(format_violation_report(report))
    return 0


def cmd_compare(args):
    times = args.times

    report = compare_strict_vs_compressed(
        max_per_second=args.N,
        request_times=[float(t) for t in times],
        bucket_width_ms=args.bucket,
        window_seconds=args.window,
    )
    print(format_comparison_report(report))

    comp = CompressedSlidingWindowLimiter(args.N, args.window, args.bucket)
    print()
    print(comp.format_error_report())
    return 0


def cmd_recommend(args):
    rec = recommend_bucket_width(
        max_per_second=args.N,
        target_error_pct=args.error,
        window_seconds=args.window,
    )
    print(format_recommend_report(rec))
    return 0


def cmd_batch(args):
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.input:
        text = args.input
    else:
        print("错误: 需要 --file 或 --input 指定输入")
        return 1

    groups = parse_batch_input(text)
    if not groups:
        print("错误: 未能从输入中解析出任何场景")
        return 1

    report = run_batch_experiment(
        groups=groups,
        N=args.N,
        window_seconds=args.window,
        bucket_width_ms=args.bucket,
        token_bucket_capacity=args.capacity,
    )
    print(format_batch_report(report))
    return 0


def cmd_multikey(args):
    mgr = MultiKeyRateLimiter(
        max_per_second=args.N,
        window_seconds=args.window,
        limiter_type=args.mode,
        bucket_width_ms=args.bucket,
        idle_ttl_seconds=args.ttl,
        cleanup_interval_seconds=0.0,
    )

    # 非交互模式
    if args.requests:
        requests = []
        for pair in args.requests:
            parts = pair.split(",")
            if len(parts) == 2:
                try:
                    t = float(parts[0])
                    key = parts[1]
                    requests.append((t, key))
                except ValueError:
                    print(f"  忽略无效输入: {pair}")

        cleanup_times = []
        if args.cleanup_at:
            cleanup_times = [float(x) for x in args.cleanup_at]

        report = mgr.run_batch(requests, cleanup_check_times=cleanup_times)
        print(format_multikey_batch_report(report))
        return 0

    # 从文件读取
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        requests = parse_batch_input_with_keys(text)
        if not requests:
            print("错误: 文件中未找到有效的 key 请求序列")
            return 1

        cleanup_times = []
        if args.cleanup_at:
            cleanup_times = [float(x) for x in args.cleanup_at]

        report = mgr.run_batch(requests, cleanup_check_times=cleanup_times)
        print(format_multikey_batch_report(report))
        return 0

    # 交互模式
    print("=" * 70)
    print(f"MultiKey 限流演示: mode={args.mode}, N={args.N}/s, idle_ttl={args.ttl}s")
    print("=" * 70)
    print(f"  输入格式: <时间(s)> <key>   例如: 0.0 user_A")
    print(f"  输入 'q' 退出, 输入 'report' 查看报告")
    print()

    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.lower() in ("q", "quit", "exit"):
            break

        if line.lower() == "report":
            print(f"  活跃 key 数: {mgr.active_key_count}")
            print(f"  活跃 keys: {mgr.get_active_keys()}")
            cleaned = mgr.cleanup_expired()
            print(f"  清理过期 key: {cleaned} 个")
            continue

        parts = line.split()
        if len(parts) != 2:
            print("  格式错误，应为: <时间> <key>")
            continue

        try:
            t = float(parts[0])
        except ValueError:
            print("  时间必须是数字")
            continue

        key = parts[1]
        ok, cnt = mgr.allow(key, now=t)
        status = "ALLOW" if ok else "REJECT"
        print(f"  t={t:.4f}s, key={key}: {status} (count={cnt})")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limiter-demo",
        description="严格滑动窗口限流器 - 命令行演示工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次限流实验
  python cli.py run --mode strict -N 5 --times 0 0.1 0.2 0.3 0.4 0.5

  # 令牌桶违规演示
  python cli.py tb-violate -N 100

  # 令牌桶未违规时自动提示建议
  python cli.py tb-violate -N 10 --capacity 1

  # 压缩版 vs 严格版对比
  python cli.py compare -N 100 --bucket 10 --times 0 0.005 0.01 ...

  # 按目标误差反推桶宽
  python cli.py recommend -N 1000000 --error 0.1

  # 批量实验（从文件）
  python cli.py batch -N 5 --file scenarios.txt

  # 按 key 非交互模式
  python cli.py multikey -N 3 --requests 0.0,user_A 0.1,user_B 0.2,user_A 0.3,user_B

  # 按 key 从文件读取
  python cli.py multikey -N 3 --file keys.txt --ttl 5 --cleanup-at 10 20
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- run --------
    p_run = subparsers.add_parser("run", help="运行一次限流实验")
    p_run.add_argument("--mode", required=True, choices=["strict", "compressed", "token-bucket"])
    p_run.add_argument("-N", type=int, required=True, help="每秒限制次数")
    p_run.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_run.add_argument("--bucket", type=int, default=1, help="压缩版桶宽(ms)")
    p_run.add_argument("--capacity", type=int, default=None, help="令牌桶容量")
    p_run.add_argument("--times", nargs="+", required=True, help="请求时间序列")
    p_run.set_defaults(func=cmd_run)

    # -------- tb-violate --------
    p_tb = subparsers.add_parser("tb-violate", help="演示令牌桶违反严格语义")
    p_tb.add_argument("-N", type=int, default=100, help="速率(次/秒)")
    p_tb.add_argument("--capacity", type=int, default=None, help="桶容量(默认=N)")
    p_tb.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_tb.set_defaults(func=cmd_tb_violate)

    # -------- compare --------
    p_cmp = subparsers.add_parser("compare", help="对比严格版 vs 压缩版")
    p_cmp.add_argument("-N", type=int, required=True, help="每秒限制次数")
    p_cmp.add_argument("--bucket", type=int, default=10, help="压缩版桶宽(ms)")
    p_cmp.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_cmp.add_argument("--times", nargs="+", required=True, help="请求时间序列")
    p_cmp.set_defaults(func=cmd_compare)

    # -------- recommend --------
    p_rec = subparsers.add_parser("recommend", help="按目标误差反推桶宽")
    p_rec.add_argument("-N", type=int, required=True, help="每秒限制次数")
    p_rec.add_argument("--error", type=float, required=True, help="可接受的误差百分比(如 0.1 表示 0.1%%)")
    p_rec.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_rec.set_defaults(func=cmd_recommend)

    # -------- batch --------
    p_batch = subparsers.add_parser("batch", help="批量实验: 多组序列 × 三种模式")
    p_batch.add_argument("-N", type=int, required=True, help="每秒限制次数")
    p_batch.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_batch.add_argument("--bucket", type=int, default=10, help="压缩版桶宽(ms)")
    p_batch.add_argument("--capacity", type=int, default=None, help="令牌桶容量")
    p_batch.add_argument("--file", type=str, default=None, help="从文件读取场景")
    p_batch.add_argument("--input", type=str, default=None, help="直接输入场景文本")
    p_batch.set_defaults(func=cmd_batch)

    # -------- multikey --------
    p_mk = subparsers.add_parser("multikey", help="按 key 限流(交互或非交互)")
    p_mk.add_argument("--mode", default="strict", choices=["strict", "compressed", "token-bucket"])
    p_mk.add_argument("-N", type=int, default=10, help="每 key 每秒限制次数")
    p_mk.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_mk.add_argument("--bucket", type=int, default=1, help="压缩版桶宽(ms)")
    p_mk.add_argument("--ttl", type=float, default=60.0, help="key 空闲过期时间(秒)")
    p_mk.add_argument("--requests", nargs="+", default=None,
                       help="非交互模式: 时间,key 序列 (如 0.0,user_A 0.1,user_B)")
    p_mk.add_argument("--file", type=str, default=None,
                       help="从文件读取 key 请求序列 (每行: 时间 key)")
    p_mk.add_argument("--cleanup-at", nargs="+", type=float, default=None,
                       help="在这些时间点检查清理效果 (如 10 20 30)")
    p_mk.set_defaults(func=cmd_multikey)

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
