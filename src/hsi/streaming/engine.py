"""
流式推理引擎 (engine.py)
=========================

两个处理器，接口完全相同：process_line(line) -> LineResult
- SklearnProcessor : 逐行直接调用 sklearn Pipeline。简单，但每次调用有固定开销（输入校验等）。
- FusedEngine      : 把三个模型融合成纯 NumPy 矩阵运算，没有任何 sklearn 调用。

每一行的处理流程（两者相同）：
1. 背景掩码：行内平均反射率 > bg_threshold 的像素视为前景（传送带背景很暗）
2. 前景像素：分类（PLS-DA）、水分回归（Ridge）、异物打分（马氏距离）
3. 背景像素：label=0，水分和异常分数为 NaN

融合只支持预处理为 raw 的模型（此时三个模型都是"仿射函数"或"二次型"）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hsi.data.simulator import AGED, FRESH
from hsi.evaluation import build_data
from hsi.models.anomaly import make_detector
from hsi.models.baselines import make_baseline
from hsi.models.regressors import make_reg_baseline

BG_THRESHOLD = 0.2  # 行内平均反射率低于它的像素当作背景（背景约 0.05~0.1，食品 > 0.3）


@dataclass
class LineResult:
    """一行的推理结果，三个数组长度都是 W。"""

    label: np.ndarray     # int8：0 背景，1 新鲜，2 陈旧，3 异物
    moisture: np.ndarray  # float：水分预测，背景为 NaN
    anomaly: np.ndarray   # float：马氏距离（平方），越大越像异物，背景为 NaN


@dataclass
class FittedModels:
    """训练好的三个 sklearn Pipeline。"""

    clf: object
    reg: object
    det: object


def fit_models(params: dict) -> FittedModels:
    """
    按 params.yaml 的配置，在**全部**像素表上训练三个模型（部署用，不再做交叉验证）。
    - 分类器：前景所有类别
    - 回归器：只用食品像素（异物没有水分）
    - 检测器：只用正常食品像素（不给它看异物）
    """
    X, y, moisture, _ = build_data(params["data"])
    mdl = params["models"]
    food = np.isin(y, (FRESH, AGED))
    clf = make_baseline(mdl["classifier"]["name"], mdl["classifier"]["prep"]).fit(X, y)
    reg = make_reg_baseline(mdl["regressor"]["name"], mdl["regressor"]["prep"]).fit(
        X[food], moisture[food])
    det = make_detector(mdl["detector"]["name"], mdl["detector"]["prep"]).fit(X[food])
    return FittedModels(clf, reg, det)


def _empty_result(w: int) -> LineResult:
    """全背景的结果：label=0，其余 NaN。"""
    return LineResult(
        label=np.zeros(w, dtype=np.int8),
        moisture=np.full(w, np.nan),
        anomaly=np.full(w, np.nan),
    )


class SklearnProcessor:
    """基线实现：对前景像素直接调用 sklearn 的 predict / score_samples。"""

    def __init__(self, models: FittedModels, bg_threshold: float = BG_THRESHOLD):
        self.m = models
        self.bg_threshold = bg_threshold

    def process_line(self, line: np.ndarray) -> LineResult:
        x = np.asarray(line, dtype=np.float64)
        fg = x.mean(axis=1) > self.bg_threshold
        out = _empty_result(x.shape[0])
        if fg.any():
            xf = x[fg]
            out.label[fg] = self.m.clf.predict(xf)
            out.moisture[fg] = self.m.reg.predict(xf)
            out.anomaly[fg] = self.m.det.score_samples(xf)
        return out


def _require_raw(pipe, name: str) -> None:
    """融合只支持 raw 预处理（Pipeline 第一步是 'passthrough'）。"""
    first = pipe[0]
    if not (isinstance(first, str) and first == "passthrough"):
        raise NotImplementedError(f"{name} 使用了预处理 {type(first).__name__}，融合引擎只支持 raw")


def _affine_probe(fn, n_features: int):
    """
    提取仿射函数 f(x) = x @ A + b 的 A 和 b（不依赖 sklearn 内部属性，跨版本稳定）。

    原理：b = f(0)；f(I) 的第 i 行 = e_i @ A + b = A 的第 i 行 + b，所以 A = f(I) - b。
    fn : (N, B) -> (N,) 或 (N, K)，必须是仿射函数
    返回 A (B, K)、b (K,)
    """
    def f2d(x):
        return np.asarray(fn(x), dtype=np.float64).reshape(len(x), -1)

    b = f2d(np.zeros((1, n_features)))[0]
    A = f2d(np.eye(n_features)) - b
    return A, b


class FusedEngine:
    """融合引擎：全部是 NumPy 矩阵运算。构造时一次性算好矩阵，process_line 里只做乘法。"""

    def __init__(self, A_cls, b_cls, classes, A_reg, b_reg, loc, prec,
                 bg_threshold: float = BG_THRESHOLD, dtype=np.float64):
        self.dtype = dtype
        self.A_cls = A_cls.astype(dtype)      # (B, K) 分类得分矩阵
        self.b_cls = b_cls.astype(dtype)      # (K,)
        self.classes = np.asarray(classes, dtype=np.int8)
        self.A_reg = A_reg.astype(dtype)      # (B,) 回归系数
        self.b_reg = dtype(b_reg)             # 标量截距
        self.loc = loc.astype(dtype)          # (B,) 正常光谱的中心
        self.prec = prec.astype(dtype)        # (B, B) 精度矩阵（协方差的逆）
        self.bg_threshold = bg_threshold

    @classmethod
    def from_sklearn(cls, models: FittedModels, bg_threshold: float = BG_THRESHOLD,
                     dtype=np.float64) -> FusedEngine:
        """从训练好的 sklearn Pipeline 提取矩阵。"""
        clf, reg, det = models.clf, models.reg, models.det
        for pipe, name in ((clf, "分类器"), (reg, "回归器"), (det, "检测器")):
            _require_raw(pipe, name)
        n_feat = clf[-1].n_features_in_  # 不用 Pipeline.n_features_in_：首步是字符串时可能报错

        # 分类得分 = PLS 回归输出（one-hot 的拟合值），对"标准化后的输入"是仿射的
        def cls_scores(x):
            return clf[-1].pls_.predict(clf[:-1].transform(x))

        A_cls, b_cls = _affine_probe(cls_scores, n_feat)
        A_reg, b_reg = _affine_probe(reg.predict, n_feat)  # Ridge 整条 Pipeline 都是仿射
        cov = det[-1].cov_
        return cls(A_cls, b_cls, clf[-1].classes_, A_reg[:, 0], b_reg[0],
                   cov.location_, cov.precision_, bg_threshold, dtype)

    def process_line(self, line: np.ndarray) -> LineResult:
        x = np.asarray(line, dtype=self.dtype)
        fg = x.mean(axis=1) > self.bg_threshold
        out = _empty_result(x.shape[0])
        if fg.any():
            xf = x[fg]                                          # (n_fg, B)
            scores = xf @ self.A_cls + self.b_cls               # (n_fg, K)
            out.label[fg] = self.classes[np.argmax(scores, axis=1)]
            out.moisture[fg] = xf @ self.A_reg + self.b_reg     # (n_fg,)
            d = xf - self.loc
            # 马氏距离平方 = 每行 d_i^T P d_i；einsum 避免构造 (n_fg, n_fg) 的大矩阵
            out.anomaly[fg] = np.einsum("ij,ij->i", d @ self.prec, d)
        return out
