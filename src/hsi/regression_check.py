"""
模型回归检查 (regression_check.py)
===================================

把新模型不能比基线差变成可执行的规则。


规则：
- 越大越好的指标（F1、AUC 等）：当前值 >= 基线 * (1 - 容差)
- 越小越好的指标（RMSE）     ：当前值 <= 基线 * (1 + 容差)
容差取相对值，默认 3%，用来吸收库版本带来的微小数值波动
（随机种子已固定，同一环境下结果应逐位一致）。
"""

from __future__ import annotations

# 指标名 -> 是否越大越好。新增指标时必须在这里登记方向，防止方向搞反。
HIGHER_IS_BETTER = {
    "cls_f1_macro": True,
    "cls_object_acc": True,
    "reg_rmse_fresh": False,
    "reg_r2_fresh": True,
    "anom_auc": True,
    "anom_detect_at_1pct_fpr": True,
}


def compare_to_baseline(current: dict, baseline: dict, rel_tol: float = 0.03) -> list[str]:
    """
    比较当前指标与基线，返回"失败说明"列表；空列表表示全部通过。
    """
    failures: list[str] = []
    for name, base in baseline.items():
        if name not in HIGHER_IS_BETTER:
            raise ValueError(f"指标 {name} 没有登记方向，请加到 HIGHER_IS_BETTER")
        if name not in current:
            failures.append(f"{name}: 当前结果里缺少该指标")
            continue
        cur = current[name]
        slack = rel_tol * abs(base)
        if HIGHER_IS_BETTER[name]:
            if cur < base - slack:
                failures.append(f"{name}: {cur:.4f} < 基线 {base:.4f}（容许下限 {base - slack:.4f}）")
        else:
            if cur > base + slack:
                failures.append(f"{name}: {cur:.4f} > 基线 {base:.4f}（容许上限 {base + slack:.4f}）")
    return failures
