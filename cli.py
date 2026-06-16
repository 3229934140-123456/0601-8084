import argparse
import sys
import ast
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
)


def cmd_run(args):
    """运行一次限流实验，逐请求打印结果"""
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
        limiter = TokenBucketLimiter(rate_per_second=N, capacity=args.capacity)
        if args.capacity:
            name = f"令牌桶 (rate={N}/s, capacity={args.capacity})"
        else:
            name = f"令牌桶 (rate={N}/s, capacity={N})"
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

    # 验证：找出最坏窗口
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
    """演示令牌桶违反严格语义"""
    report = analyze_token_bucket_violation_strict(
        rate_per_second=args.N,
        capacity=args.capacity if args.capacity else args.N,
        window_seconds=args.window,
    )
    print(format_violation_report(report))
    return 0


def cmd_compare(args):
    """对比严格版与压缩版"""
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


def cmd_multikey(args):
    """演示按 key 限流"""
    mgr = MultiKeyRateLimiter(
        max_per_second=args.N,
        window_seconds=args.window,
        limiter_type=args.mode,
        bucket_width_ms=args.bucket,
        idle_ttl_seconds=args.ttl,
    )

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
  # 基本演示 - 严格滑动窗口
  python cli.py run --mode strict -N 5 --times 0 0.1 0.2 0.3 0.4 0.5 0.99 1.0 1.01

  # 令牌桶违规演示
  python cli.py tb-violate -N 100

  # 压缩版 vs 严格版对比
  python cli.py compare -N 100 --bucket 10 --times 0 0.005 0.01 ...

  # 按 key 限流交互式演示
  python cli.py multikey --mode strict -N 10
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- run --------
    p_run = subparsers.add_parser("run", help="运行一次限流实验")
    p_run.add_argument(
        "--mode", required=True,
        choices=["strict", "compressed", "token-bucket"],
        help="限流器模式",
    )
    p_run.add_argument("-N", type=int, required=True, help="每秒限制次数")
    p_run.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_run.add_argument("--bucket", type=int, default=1, help="压缩版桶宽(ms)")
    p_run.add_argument("--capacity", type=int, default=None, help="令牌桶容量")
    p_run.add_argument(
        "--times", nargs="+", required=True,
        help="请求时间序列，例如: 0 0.1 0.2 0.5 0.99 1.0",
    )
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
    p_cmp.add_argument(
        "--times", nargs="+", required=True,
        help="请求时间序列",
    )
    p_cmp.set_defaults(func=cmd_compare)

    # -------- multikey --------
    p_mk = subparsers.add_parser("multikey", help="交互式按 key 限流演示")
    p_mk.add_argument(
        "--mode", default="strict",
        choices=["strict", "compressed", "token-bucket"],
        help="限流器模式",
    )
    p_mk.add_argument("-N", type=int, default=10, help="每 key 每秒限制次数")
    p_mk.add_argument("--window", type=float, default=1.0, help="窗口大小(秒)")
    p_mk.add_argument("--bucket", type=int, default=1, help="压缩版桶宽(ms)")
    p_mk.add_argument("--ttl", type=float, default=60.0, help="key 空闲过期时间(秒)")
    p_mk.set_defaults(func=cmd_multikey)

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
