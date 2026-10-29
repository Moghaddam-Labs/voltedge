"""
VoltEdge — Anomaly detector evaluation.

Measures how well the Isolation Forest detector finds the anomalies the
simulator injects on purpose. The simulator marks every injected spike in
`metadata["injected_anomaly"]`, so we can score the detector against known
answers instead of trusting it blindly.

Method:
  - For each of the four demo sites, generate 90 days of hourly readings
    with several random seeds.
  - Run the same feature pipeline the app uses (EnergyTransformer).
  - Train on the first 60 days, then score the last 30 days, which the
    model has never seen (a time-based hold-out, no shuffling).
  - Count a reading as caught only if the detector flags that exact hour.

Use --cleaned-readings to score the detector on the outlier-cleaned kWh
values instead, which is how it worked before it switched to `kwh_raw`.

Usage:
    python3 scripts/evaluate_anomaly.py
    python3 scripts/evaluate_anomaly.py --seeds 5 --cleaned-readings
    python3 scripts/evaluate_anomaly.py --markdown > results.md
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voltedge.ingestion.simulators import SimulatedSiteConnector  # noqa: E402
from voltedge.models.anomaly import AnomalyDetector  # noqa: E402
from voltedge.processing.transformer import EnergyTransformer  # noqa: E402

# The same four demo sites the dashboard shows (see voltedge/db/loader.py).
DEMO_SITES: dict[str, str] = {
    "LONDON-FACTORY-01": "factory",
    "DUBAI-OFFICE-01": "office",
    "ROTTERDAM-WAREHOUSE-01": "warehouse",
    "FRANKFURT-DC-01": "data_center",
}


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def evaluate_site(
    site_id: str,
    site_type: str,
    seed: int,
    days: int = 90,
    train_days: int = 60,
    cleaned_readings: bool = False,
    artifact_dir: str | None = None,
) -> Counts:
    """Train on the first `train_days`, score the rest, and count hits and misses."""
    raw = asyncio.run(
        SimulatedSiteConnector(site_id, site_type, hours=days * 24, seed=seed).fetch()
    ).data
    df = EnergyTransformer().transform(raw)
    if cleaned_readings:
        df = df.drop(columns=["kwh_raw"])
    df["injected"] = df["metadata"].apply(
        lambda m: bool(m.get("injected_anomaly")) if isinstance(m, dict) else False
    )

    cut = df["timestamp"].iloc[0] + pd.Timedelta(days=train_days)
    train, test = df[df["timestamp"] < cut], df[df["timestamp"] >= cut]

    model = AnomalyDetector(site_id, artifact_dir=artifact_dir or tempfile.mkdtemp())
    model.train(train)
    scored = model.predict(test)

    actual = scored["injected"].astype(bool).values
    flagged = scored["is_anomaly"].astype(bool).values
    return Counts(
        tp=int((actual & flagged).sum()),
        fp=int((~actual & flagged).sum()),
        fn=int((actual & ~flagged).sum()),
    )


def evaluate(seeds: int = 10, cleaned_readings: bool = False) -> dict[str, Counts]:
    """Run every demo site across `seeds` random seeds. Returns counts per site plus 'All sites'."""
    artifact_dir = tempfile.mkdtemp()
    results: dict[str, Counts] = {}
    total = Counts()
    for site_id, site_type in DEMO_SITES.items():
        site_total = Counts()
        for seed in range(1, seeds + 1):
            site_total.add(
                evaluate_site(
                    site_id,
                    site_type,
                    seed,
                    cleaned_readings=cleaned_readings,
                    artifact_dir=artifact_dir,
                )
            )
        results[site_id] = site_total
        total.add(site_total)
    results["All sites"] = total
    return results


def format_table(results: dict[str, Counts], markdown: bool = False) -> str:
    header = ["Site", "Caught", "False alarms", "Missed", "Precision", "Recall", "F1"]
    rows = [
        [
            name,
            str(c.tp),
            str(c.fp),
            str(c.fn),
            f"{c.precision:.0%}",
            f"{c.recall:.0%}",
            f"{c.f1:.2f}",
        ]
        for name, c in results.items()
    ]
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max(len(x) for x in col) for col in zip(header, *rows, strict=True)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*header)] + [fmt.format(*r) for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the VoltEdge anomaly detector.")
    parser.add_argument("--seeds", type=int, default=10, help="Random seeds per site (default 10)")
    parser.add_argument(
        "--cleaned-readings",
        action="store_true",
        help="Score on outlier-cleaned kWh (the old behaviour), to compare",
    )
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table")
    args = parser.parse_args()

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    results = evaluate(seeds=args.seeds, cleaned_readings=args.cleaned_readings)
    mode = "cleaned readings (old behaviour)" if args.cleaned_readings else "raw readings"
    print(f"Anomaly detector, {args.seeds} seeds per site, 60-day train / 30-day test, {mode}\n")
    print(format_table(results, markdown=args.markdown))


if __name__ == "__main__":
    main()
