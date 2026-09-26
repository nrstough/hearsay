"""Pure-numpy evaluator for a LightGBM booster dump, so the learned detectors never import
lightgbm at inference.

Why: LightGBM and torch each ship their own `libomp.dylib` on macOS, and loading both in one
process segfaults or deadlocks (measured Sat Sep 26: `lightgbm` fit then XLS-R load crashes,
the other order hangs; `KMP_DUPLICATE_LIB_OK` does not help, these are LLVM runtimes). The
training script uses lightgbm alone and stores `booster_.dump_model()` in the bundle; the
detector rebuilds the trees here and runs them with numpy. The orchestrator can therefore
load XLS-R and these bundles in the same process.

Also gives per-feature contributions by path attribution (Saabas / treeinterpreter): walking
each tree, the change in node value across a split is credited to the split feature, so the
contributions plus the root values sum exactly to the raw score. LightGBM stores every node's
value (`internal_value`, `leaf_value`) with shrinkage applied, so the scale is right.
"""

from __future__ import annotations

import numpy as np


class Trees:
    """Binary-objective LightGBM booster from `dump_model()`; numeric `<=` splits only."""

    def __init__(self, dump: dict):
        self.n_features = int(dump["max_feature_idx"]) + 1
        self.trees = [self._flatten(t["tree_structure"]) for t in dump["tree_info"]]

    @staticmethod
    def _flatten(root: dict) -> dict[str, np.ndarray]:
        cols: dict[str, list] = {k: [] for k in ("feat", "thr", "left", "right", "value", "dleft")}

        def add(node: dict) -> int:
            i = len(cols["feat"])
            if "leaf_value" in node:
                for k, v in (("feat", -1), ("thr", 0.0), ("left", -1), ("right", -1),
                             ("value", float(node["leaf_value"])), ("dleft", True)):  # fmt: skip
                    cols[k].append(v)
                return i
            if node.get("decision_type", "<=") != "<=":
                raise ValueError(f"unsupported split type {node['decision_type']!r}")
            for k, v in (("feat", int(node["split_feature"])), ("thr", float(node["threshold"])),
                         ("left", -1), ("right", -1), ("value", float(node["internal_value"])),
                         ("dleft", bool(node.get("default_left", True)))):  # fmt: skip
                cols[k].append(v)
            cols["left"][i] = add(node["left_child"])
            cols["right"][i] = add(node["right_child"])
            return i

        add(root)
        return {k: np.asarray(v) for k, v in cols.items()}

    def raw(self, X: np.ndarray) -> np.ndarray:
        """Sum of leaf values over trees, one value per row (the logit for binary objectives)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < self.n_features:
            raise ValueError(f"expected (n, >={self.n_features}) features, got {X.shape}")
        out = np.zeros(len(X))
        for t in self.trees:
            node = np.zeros(len(X), dtype=int)
            while True:
                idx = np.where(t["feat"][node] >= 0)[0]
                if idx.size == 0:
                    break
                cur = node[idx]
                x = X[idx, t["feat"][cur]]
                go_left = np.where(np.isnan(x), t["dleft"][cur], x <= t["thr"][cur])
                node[idx] = np.where(go_left, t["left"][cur], t["right"][cur])
            out += t["value"][node]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.raw(X)))
        return np.stack([1.0 - p, p], axis=1)

    def contrib(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """(per-feature contributions, bias) for one row; contributions.sum() + bias == raw."""
        x = np.asarray(x, dtype=np.float64)
        c, bias = np.zeros(self.n_features), 0.0
        for t in self.trees:
            i = 0
            bias += float(t["value"][0])
            while t["feat"][i] >= 0:
                f = int(t["feat"][i])
                xv = x[f]
                left = bool(t["dleft"][i]) if np.isnan(xv) else xv <= t["thr"][i]
                nxt = int(t["left"][i] if left else t["right"][i])
                c[f] += t["value"][nxt] - t["value"][i]
                i = nxt
        return c, bias
