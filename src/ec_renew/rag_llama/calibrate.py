"""向量阈值标定：给 ``VECTOR_MIN_SCORE`` 找一个有数据支撑的取值。

为什么需要它
------------
`VECTOR_MIN_SCORE` 决定「纯向量命中算不算真实历史案例」，直接左右
``assess_grounding`` 是 ``history_backed`` 还是 ``knowledge_based``。

而不同向量模型的余弦分布差异极大：智谱 ``embedding-3`` 在本语料上整体压在
0.1~0.47，凭直觉设 0.5 会把**全部**真实命中拒掉；反过来设 0.2 又会把
「餐厅菜单」这类域外请求也当成有历史依据。所以这个值只能量出来。

原理
----
给两组带标签的 query（域内改写 / 域外无关），各取 top-1 相似度，
扫描阈值找一个把两组分开的位置：

* **域内通过率高** —— 否则该开关形同虚设；
* **域外误放率低** —— 否则会把 ``knowledge_based`` 误抬成 ``history_backed``。

用法::

    python -m ec_renew.rag_llama.calibrate
    python -m ec_renew.rag_llama.calibrate --json
    python -m ec_renew.rag_llama.calibrate --queries my_labels.json

``my_labels.json`` 形如 ``{"in_domain": [...], "out_domain": [...]}``。
换语料、换向量模型、换维度之后都应重跑一次。
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..config import settings as default_settings

# 域内改写：用本语料（118 张变更单）的部件/原因/工艺词汇写成真实改述，
# 刻意不照抄原文，因为「照抄原文」本来就轮不到向量腿出手。
IN_DOMAIN_QUERIES: tuple[str, ...] = (
    "301分段FR36污水井更换加厚板",
    "增加扶强材",
    "缆绳采购需要提供产品证书",
    "主机座区域增加扁钢",
    "货舱区贯穿孔补焊",
    "污水井位置调整导致结构修改",
    "甲板板因干涉需要改动",
    "肋板补孔",
    "纵舱壁增加拼板",
    "风道修改导致门孔移位",
    "扁钢下料遗漏需要补料",
    "锌块需要增加数量",
    "蓄电池支架焊接变形需处理",
    "投光灯底座安装位置调整",
    "雷达桅结构加强",
    "艉楼甲板开孔修改",
)

# 域外：主题要足够分散，且尽量不与船体词汇共享汉字 —— 汉字重合会抬高余弦，
# 那正是这个阈值要挡住的假阳性来源之一。
OUT_DOMAIN_QUERIES: tuple[str, ...] = (
    "餐厅菜单调整与食材采购",
    "今天股市大涨基金收益不错",
    "如何学习Python编程入门",
    "员工年度绩效考核怎么写",
    "公司财务报销流程说明",
    "健身房一周锻炼计划",
    "云南旅游攻略",
    "简历模板下载",
    "最近有什么好电影推荐",
    "两万元预算买什么手机",
    "红烧肉的家常做法",
    "考研数学复习安排",
    "猫咪换粮注意事项",
    "租房合同注意事项",
    "车险理赔流程",
    "婚礼策划方案",
    "小户型装修风格",
    "三岁孩子早教方法",
    "游戏本配置推荐",
    "周末去哪玩",
)


@dataclass
class CalibrationReport:
    in_scores: list[float] = field(default_factory=list)
    out_scores: list[float] = field(default_factory=list)
    best_threshold: float = 0.0
    best_accuracy: float = 0.0
    best_in_pass: int = 0
    best_out_pass: int = 0
    sweep: list[dict[str, float]] = field(default_factory=list)

    @property
    def gap(self) -> tuple[float, float]:
        """（域外最高, 域内最低）—— 两者之间就是可分空隙。"""
        return (
            max(self.out_scores) if self.out_scores else 0.0,
            min(self.in_scores) if self.in_scores else 1.0,
        )

    def as_dict(self) -> dict[str, Any]:
        out_max, in_min = self.gap
        return {
            "in_domain": {
                "n": len(self.in_scores),
                "min": round(in_min, 4),
                "median": round(stats.median(self.in_scores), 4) if self.in_scores else None,
                "max": round(max(self.in_scores), 4) if self.in_scores else None,
            },
            "out_domain": {
                "n": len(self.out_scores),
                "min": round(min(self.out_scores), 4) if self.out_scores else None,
                "median": round(stats.median(self.out_scores), 4) if self.out_scores else None,
                "max": round(out_max, 4),
            },
            "separating_gap": [round(out_max, 4), round(in_min, 4)],
            "best_threshold": self.best_threshold,
            "best_accuracy": round(self.best_accuracy, 4),
            "in_domain_pass": self.best_in_pass,
            "out_domain_false_positive": self.best_out_pass,
            "sweep": self.sweep,
        }


def _top1_scores(queries: Sequence[str], retriever: Any) -> list[float]:
    scores: list[float] = []
    for query in queries:
        hits = retriever.retrieve(query)
        if hits:
            scores.append(float(hits[0].score or 0.0))
    return scores


def calibrate(
    cfg: Settings | None = None,
    *,
    in_domain: Sequence[str] = IN_DOMAIN_QUERIES,
    out_domain: Sequence[str] = OUT_DOMAIN_QUERIES,
    top_k: int = 3,
) -> CalibrationReport:
    """需要真实向量服务（会真的调用 embedding 接口）。"""
    from llama_index.core import VectorStoreIndex

    from .embeddings import build_embed_model
    from .vector_store import build_client, build_vector_store

    cfg = cfg or default_settings
    embed_model = build_embed_model(cfg)
    client = build_client(cfg)
    index = VectorStoreIndex.from_vector_store(
        build_vector_store(client, cfg.qdrant_collection), embed_model=embed_model
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    report = CalibrationReport(
        in_scores=sorted(_top1_scores(in_domain, retriever)),
        out_scores=sorted(_top1_scores(out_domain, retriever)),
    )
    if not report.in_scores or not report.out_scores:
        raise RuntimeError("标定失败：向量库里没有可比对的点，请先摄取语料。")

    total = len(report.in_scores) + len(report.out_scores)
    all_scores = report.in_scores + report.out_scores

    # 在一条细网格上扫描，而不是只在观测到的分数上取值 —— 后者的最优解
    # 恰好落在「域内最低分」上，等于零安全余量，任何一次抖动都会误杀。
    # 并列最优时选**离所有观测分数最远**的那个（最大间隔），也就是落在
    # 可分空隙正中的那个阈值。
    lo, hi, step = min(all_scores), max(all_scores), 0.001
    grid = [round(lo + index * step, 4) for index in range(int((hi - lo) / step) + 2)]

    best: tuple[float, float, float, int, int] | None = None
    previous: tuple[int, int] | None = None
    for threshold in grid:
        in_pass = sum(1 for s in report.in_scores if s >= threshold)
        out_pass = sum(1 for s in report.out_scores if s >= threshold)
        accuracy = (in_pass + len(report.out_scores) - out_pass) / total
        margin = min(abs(threshold - s) for s in all_scores)
        record = (accuracy, margin, threshold, in_pass, out_pass)
        if best is None or record[:2] > best[:2]:
            best = record
        # sweep 只记录「判定发生变化」的断点，避免几百行噪声。
        if (in_pass, out_pass) != previous:
            previous = (in_pass, out_pass)
            report.sweep.append(
                {
                    "threshold": threshold,
                    "accuracy": round(accuracy, 4),
                    "in_domain_pass": in_pass,
                    "out_domain_false_positive": out_pass,
                }
            )

    assert best is not None
    report.best_accuracy, _, report.best_threshold, report.best_in_pass, report.best_out_pass = best
    closer = getattr(embed_model, "close", None)
    if callable(closer):
        closer()
    client.close()
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ec_renew.rag_llama.calibrate",
        description="用带标签的 query 集合标定 VECTOR_MIN_SCORE",
    )
    parser.add_argument("--queries", help="JSON 文件：{\"in_domain\": [...], \"out_domain\": [...]}")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    in_domain: Sequence[str] = IN_DOMAIN_QUERIES
    out_domain: Sequence[str] = OUT_DOMAIN_QUERIES
    if args.queries:
        payload = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        in_domain = payload.get("in_domain") or in_domain
        out_domain = payload.get("out_domain") or out_domain

    try:
        report = calibrate(in_domain=in_domain, out_domain=out_domain)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    out_max, in_min = report.gap
    print("向量阈值标定")
    print(f"  域内 n={payload['in_domain']['n']}  top1: {payload['in_domain']}")
    print(f"  域外 n={payload['out_domain']['n']}  top1: {payload['out_domain']}")
    print(f"  可分空隙: 域外最高 {out_max:.4f} → 域内最低 {in_min:.4f}")
    print(
        f"  推荐 VECTOR_MIN_SCORE={report.best_threshold:.4f}"
        f"（域内通过 {report.best_in_pass}/{len(report.in_scores)}，"
        f"域外误放 {report.best_out_pass}/{len(report.out_scores)}）"
    )
    print()
    print("  扫描明细（threshold, 准确率, 域内通过, 域外误放）：")
    for row in payload["sweep"]:
        print(
            f"    {row['threshold']:.4f}  {row['accuracy']:.3f}"
            f"  {row['in_domain_pass']:2d}  {row['out_domain_false_positive']:2d}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
