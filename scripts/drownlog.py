#!/usr/bin/env python3
"""
溺水事件日志管理脚本（基于 logs/alert_events.jsonl）

功能：
- recent   查看近期溺水事件（按事件聚合，默认24小时内）
- show     查看某次溺水事件全链路（event_id）
- stats    统计（事件数、规则触发、推送成功/失败）

示例：
  python scripts/drownlog.py recent -n 10
  python scripts/drownlog.py show --event D-169...-a1b2c3
  python scripts/drownlog.py stats --since 7d
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_FILE = Path("./logs/alert_events.jsonl")


def parse_since(expr: Optional[str]) -> Optional[float]:
    """将 --since 表达式解析为 epoch 秒。
    支持："24h"、"7d"、"YYYY-MM-DD" 或空（返回 None）。
    维护提示：如需更复杂语法，可引入 dateparser（但本项目避免外部依赖）。
    """
    if not expr:
        return None
    expr = expr.strip()
    now = datetime.now()
    try:
        if expr.lower().endswith("h"):
            hours = float(expr[:-1])
            return (now - timedelta(hours=hours)).timestamp()
        if expr.lower().endswith("d"):
            days = float(expr[:-1])
            return (now - timedelta(days=days)).timestamp()
        # YYYY-MM-DD
        dt = datetime.strptime(expr, "%Y-%m-%d")
        return dt.timestamp()
    except Exception:
        raise SystemExit(f"无法解析 --since 值: {expr}")


def load_events(file: Path, since_ts: Optional[float] = None) -> Iterable[Dict]:
    """按行读取 JSONL 事件日志，并可基于时间阈值过滤。"""
    if not file.exists():
        raise SystemExit(f"找不到日志文件: {file}")
    with file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if since_ts is not None and float(obj.get("time", 0)) < since_ts:
                    continue
                yield obj
            except Exception:
                continue


def group_by_event_id(events: Iterable[Dict]) -> Dict[str, List[Dict]]:
    """将原始事件按 event_id 聚合，并按时间排序。"""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for e in events:
        eid = e.get("event_id") or "<unknown>"
        grouped[eid].append(e)
    for lst in grouped.values():
        lst.sort(key=lambda x: float(x.get("time", 0)))
    return grouped


def fmt_time(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def cmd_recent(args: argparse.Namespace) -> None:
    """展示近期事件的摘要（按事件聚合）。"""
    since_ts = parse_since(args.since) if args.since else parse_since("24h")
    events = list(load_events(args.file, since_ts))
    grouped = group_by_event_id(events)
    # 以最后一条记录时间排序，取前 n 个
    items: List[Tuple[str, List[Dict]]] = sorted(
        grouped.items(), key=lambda kv: kv[1][-1].get("time", 0), reverse=True
    )
    items = items[: args.n]

    if not items:
        print("近期无溺水事件。")
        return
    for eid, recs in items:
        start_ts = recs[0].get("time", 0)
        end_ts = recs[-1].get("time", 0)
        accepts = [r for r in recs if r.get("event") == "accept"]
        rules = []
        for a in accepts:
            rules.extend([r.get("name") for r in a.get("rules", [])])
        rule_top = ", ".join([f"{k}:{v}" for k, v in Counter(rules).most_common(3)])
        ok = sum(1 for r in recs if r.get("event") == "push_ok")
        fail = sum(1 for r in recs if r.get("event") == "push_fail")
        print(
            f"event_id={eid} time=[{fmt_time(start_ts)} ~ {fmt_time(end_ts)}] "
            f"tracks={len(set([t.get('track_id') for t in accepts]))} push(ok/fail)={ok}/{fail} rules[{rule_top}]"
        )


def pretty_event_chain(records: List[Dict]) -> str:
    """将单个事件的全链路记录格式化为易读文本。"""
    lines: List[str] = []
    for r in records:
        ev = r.get("event")
        ts = fmt_time(r.get("time", 0))
        if ev == "accept":
            track = r.get("track_id")
            rules = ", ".join([f"[{rule.get('id')}] {rule.get('name')}={rule.get('value')}" for rule in r.get("rules", [])])
            lines.append(f"{ts} accept  track={track} rules={r.get('rule_count')} {rules}")
        elif ev == "queue":
            lines.append(f"{ts} queue   tracks={r.get('count')} ids={r.get('track_ids')}")
        elif ev == "start":
            lines.append(f"{ts} start   tracks={r.get('count')} ids={r.get('track_ids')}")
        elif ev == "push_ok":
            lines.append(f"{ts} push_ok client={r.get('client')}")
        elif ev == "push_fail":
            lines.append(f"{ts} push_fail client={r.get('client')} error={r.get('error')}")
        elif ev == "finish":
            lines.append(f"{ts} finish  ok={r.get('ok')} fail={r.get('fail')}")
        else:
            lines.append(f"{ts} {ev}  data={{{...}}}")
    return "\n".join(lines)


def cmd_show(args: argparse.Namespace) -> None:
    """展示指定 event_id 的全链路详情。"""
    if not args.event:
        raise SystemExit("请使用 --event 指定 event_id")
    events = list(load_events(args.file, None))
    grouped = group_by_event_id(events)
    recs = grouped.get(args.event)
    if not recs:
        print(f"未找到事件: {args.event}")
        return
    print(pretty_event_chain(recs))


def cmd_stats(args: argparse.Namespace) -> None:
    """统计指定时间窗口内的事件数量、推送成功/失败、按天分布与规则Top10。"""
    since_ts = parse_since(args.since) if args.since else parse_since("7d")
    events = list(load_events(args.file, since_ts))
    grouped = group_by_event_id(events)
    total_events = len(grouped)
    total_push_ok = sum(1 for e in events if e.get("event") == "push_ok")
    total_push_fail = sum(1 for e in events if e.get("event") == "push_fail")
    # 按天统计
    per_day = Counter(datetime.fromtimestamp(e.get("time", 0)).strftime("%Y-%m-%d") for e in events if e.get("event") == "finish")
    # 规则统计
    rule_counter = Counter()
    for e in events:
        if e.get("event") == "accept":
            rule_counter.update([r.get("name") for r in e.get("rules", [])])

    if args.json:
        out = {
            "events": total_events,
            "push_ok": total_push_ok,
            "push_fail": total_push_fail,
            "per_day": dict(per_day),
            "top_rules": dict(rule_counter.most_common(10)),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"事件总数: {total_events}")
    print(f"推送成功/失败: {total_push_ok}/{total_push_fail}")
    print("按天事件数:")
    for day, cnt in sorted(per_day.items()):
        print(f"  {day}: {cnt}")
    print("最常触发规则(Top10):")
    for name, cnt in rule_counter.most_common(10):
        print(f"  {name}: {cnt}")


def build_rule_bars(counter: Counter, width: int = 20) -> List[str]:
    """构建规则计数的 ASCII 条形图摘要（用于快速人读）。"""
    total = max(counter.values() or [0]) or 1
    lines = []
    for name, cnt in counter.most_common():
        bar_len = max(1, int(cnt / total * width)) if cnt > 0 else 0
        lines.append(f"{name}: {'#' * bar_len} ({cnt})")
    return lines


def cmd_export(args: argparse.Namespace) -> None:
    """导出指定 event_id 的 JSON 报告，可附带链路文本或原始记录。"""
    if not args.event:
        raise SystemExit("请使用 --event 指定 event_id")
    events = list(load_events(args.file, None))
    grouped = group_by_event_id(events)
    recs = grouped.get(args.event)
    if not recs:
        print(f"未找到事件: {args.event}")
        return

    # 聚合
    start_ts = recs[0].get("time", 0)
    end_ts = recs[-1].get("time", 0)
    accepts = [r for r in recs if r.get("event") == "accept"]
    track_ids = sorted(list({r.get("track_id") for r in accepts if r.get("track_id")}))
    # 规则统计
    rule_counter = Counter()
    rule_examples: Dict[str, List[str]] = defaultdict(list)
    for a in accepts:
        for rule in a.get("rules", []):
            name = rule.get("name")
            val = rule.get("value")
            rule_counter.update([name])
            if len(rule_examples[name]) < 5:
                rule_examples[name].append(val)
    rule_bars = build_rule_bars(rule_counter)

    ok_clients = [r.get("client") for r in recs if r.get("event") == "push_ok"]
    fail_clients = [r.get("client") for r in recs if r.get("event") == "push_fail"]

    report = {
        "event_id": args.event,
        "period": {
            "start": fmt_time(start_ts),
            "end": fmt_time(end_ts),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_sec": max(0.0, float(end_ts) - float(start_ts)),
        },
        "tracks": track_ids,
        "summary": {
            "rule_counts": dict(rule_counter),
            "rule_examples": rule_examples,
            "rule_bars": rule_bars,
        },
        "push": {
            "ok": ok_clients,
            "fail": fail_clients,
        },
        "chain_text": pretty_event_chain(recs),
        "records": recs if args.full else None,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2 if args.pretty else None)
        print(f"已导出: {out}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器；为保持“零依赖”，仅使用 argparse。"""
    p = argparse.ArgumentParser(description="溺水事件日志管理工具")
    p.add_argument("command", choices=["recent", "show", "stats", "export"], help="子命令")
    p.add_argument("--file", default=str(DEFAULT_FILE), help="事件日志文件路径，默认 logs/alert_events.jsonl")
    p.add_argument("--since", default=None, help="时间窗口，如 24h、7d、2025-01-31")
    p.add_argument("-n", type=int, default=20, help="recent 列表数量，默认 20")
    p.add_argument("--event", help="show/export 子命令的事件ID")
    p.add_argument("--json", action="store_true", help="stats 以 JSON 输出")
    p.add_argument("-o", "--output", help="export 输出文件路径（不指定则打印到控制台）")
    p.add_argument("--pretty", action="store_true", help="export 使用缩进格式化 JSON")
    p.add_argument("--full", action="store_true", help="export 输出包含原始 records")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """入口函数：根据子命令分派到各处理函数。"""
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    # 归一化路径
    args.file = Path(args.file)

    if args.command == "recent":
        cmd_recent(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
