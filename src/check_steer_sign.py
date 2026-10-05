from __future__ import annotations

import argparse

import numpy as np


def check_prefix(prefix: str, steer_strong: float = 0.3) -> None:
    """Report whether a dataset's steering labels match the physical turning
    direction the pose data shows.

    The feedback steering sign convention has flipped between recordings more
    than once (hardware-driver edits land between bags), so every new bag
    should be checked before its data is appended to training sets recorded
    earlier. Mixing conventions teaches the model inverted steering exactly
    where the new data dominates.

    Method: heading change over the last two window steps vs the steering
    label. Plain correlation is reported but can be misleadingly weak when
    steering sits saturated near one value (e.g. a tight-corner recording), so
    the verdict uses the robust version: the mean heading change during
    strongly-steered samples. Positive steering with positive mean heading
    change (and vice versa) = matches physical convention.
    """
    X = np.load(f"{prefix}_X.npy")
    y = np.load(f"{prefix}_y.npy")

    th_last = np.arctan2(X[:, -1, 2], X[:, -1, 3])
    th_prev = np.arctan2(X[:, -2, 2], X[:, -2, 3])
    dth = np.arctan2(np.sin(th_last - th_prev), np.cos(th_last - th_prev))
    s = y[:, 0]
    v = y[:, 1]

    moving = np.abs(dth) > 0.005
    corr = np.corrcoef(s[moving], dth[moving])[0, 1] if moving.sum() > 2 else float("nan")

    forward = v > -0.1  # reversing flips the steer->heading relationship; exclude it
    pos = forward & (s > steer_strong)
    neg = forward & (s < -steer_strong)
    pos_dth = float(dth[pos].mean()) if pos.any() else float("nan")
    neg_dth = float(dth[neg].mean()) if neg.any() else float("nan")

    # Sample-weighted evidence: sign(steer)*dth is positive when the label
    # matches the physical turn direction. Pooling all strongly-steered rows
    # weights each sample equally, so a small dissenting group (e.g. messy
    # recovery maneuvers during interventions) can't veto the dominant one.
    strong = pos | neg
    if strong.sum() < 10:
        verdict = "INCONCLUSIVE (too few strongly-steered samples)"
    else:
        agreement = float((np.sign(s[strong]) * dth[strong]).mean())
        if agreement > 0:
            verdict = "MATCHES physical convention (same as demo8) -- no sign flip needed"
        else:
            verdict = "INVERTED -- negate y[:, 0] before mixing with demo8-era data"

    print(f"{prefix}")
    print(f"  samples: {len(y)}  reversing (v<-0.1): {(v < -0.1).sum()}")
    print(f"  corr(steer, heading_change) while turning: {corr:+.3f}")
    print(f"  steer>+{steer_strong}: n={pos.sum():4d} mean_dth={pos_dth:+.4f}")
    print(f"  steer<-{steer_strong}: n={neg.sum():4d} mean_dth={neg_dth:+.4f}")
    print(f"  verdict: {verdict}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a dataset's steering-label sign matches the physical "
        "turning direction (run on every new bag's dataset before appending it to older data)."
    )
    parser.add_argument("prefixes", nargs="+", help="One or more {prefix}_X.npy/_y.npy prefixes, e.g. data/cornerdagger1_train")
    parser.add_argument("--steer_strong", type=float, default=0.3, help="Threshold for 'strongly steered' samples used by the verdict")
    args = parser.parse_args()

    for prefix in args.prefixes:
        check_prefix(prefix, steer_strong=args.steer_strong)


if __name__ == "__main__":
    main()
