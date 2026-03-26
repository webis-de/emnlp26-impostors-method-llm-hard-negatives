#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from numpy import mean

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def _parse_created_at(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, TIME_FORMAT)
    raise TypeError(f"Unexpected created_at type: {type(value)}")


def _span_buckets(min_dt: datetime, max_dt: datetime, seconds: int) -> int:
    span_seconds = (max_dt - min_dt).total_seconds()
    return int(span_seconds // seconds) + 1


def _floor_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def _floor_minute(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def _build_series(start: datetime, end: datetime, step: timedelta, counts: Counter) -> list[int]:
    series = []
    current = start
    while current <= end:
        series.append(counts.get(current, 0))
        current += step
    return series


def _style_boxplot(bp, colors: list[str]) -> None:
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
    for whisker, color in zip(bp["whiskers"], [colors[0]] * 2 + [colors[1]] * 2):
        whisker.set_color(color)
    for cap, color in zip(bp["caps"], [colors[0]] * 2 + [colors[1]] * 2):
        cap.set_color(color)
    for median, color in zip(bp["medians"], colors):
        median.set_color(color)
        median.set_linewidth(2)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor(color)
        flier.set_alpha(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average entries per hour and per minute for non_naive_paraphrases."
    )
    parser.add_argument("--collection", default="non_naive_paraphrases")
    parser.add_argument(
        "--output",
        default="non_naive_paraphrases_rate.json",
        help="Output filename under results/openai_stats.",
    )
    args = parser.parse_args()

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    collection = mongoDB.db[args.collection]

    hour_counts = Counter()
    minute_counts = Counter()
    total = 0
    min_dt = None
    max_dt = None

    cursor = collection.find({}, {"created_at": 1, "_id": 0})
    for doc in cursor:
        total += 1
        dt = _parse_created_at(doc["created_at"])
        if min_dt is None or dt < min_dt:
            min_dt = dt
        if max_dt is None or dt > max_dt:
            max_dt = dt
        hour_counts[_floor_hour(dt)] += 1
        minute_counts[_floor_minute(dt)] += 1

    # threshold = 1000
    #
    # high_hours = sorted(
    #     ((ts, count) for ts, count in hour_counts.items() if count > threshold),
    #     key=lambda x: x[1],
    #     reverse=True,
    # )
    # high_minutes = sorted(
    #     ((ts, count) for ts, count in minute_counts.items() if count > threshold),
    #     key=lambda x: x[1],
    #     reverse=True,
    # )
    #
    # print(f"Hours with > {threshold} paraphrases:")
    # for ts, count in high_hours:
    #     print(ts.strftime(TIME_FORMAT), count)
    #
    # print(f"Minutes with > {threshold} paraphrases:")
    # for ts, count in high_minutes:
    #     print(ts.strftime(TIME_FORMAT), count)
    #
    # # Find and print all timestamps in a specific hour bin
    # target_hour = datetime.strptime("2026-03-04_00-00-00", TIME_FORMAT)
    # end_hour = target_hour + timedelta(hours=1)
    #
    # timestamps = []
    # cursor = collection.find(
    #     {
    #         "created_at": {
    #             "$gte": target_hour.strftime(TIME_FORMAT),
    #             "$lt": end_hour.strftime(TIME_FORMAT),
    #         }
    #     },
    #     {"created_at": 1, "_id": 0},
    # )
    #
    # for doc in cursor:
    #     timestamps.append(doc["created_at"])
    #
    # print(f"Total in {target_hour.strftime(TIME_FORMAT)}: {len(timestamps)}")
    # for ts in timestamps:
    #     print(ts)

    if total == 0:
        span_hours = 0
        span_minutes = 0
        avg_per_hour = 0.0
        avg_per_minute = 0.0
        hour_series = []
        minute_series = []
    else:
        hour_start = _floor_hour(min_dt)
        hour_end = _floor_hour(max_dt)
        minute_start = _floor_minute(min_dt)
        minute_end = _floor_minute(max_dt)

        span_hours = _span_buckets(hour_start, hour_end, 3600)
        span_minutes = _span_buckets(minute_start, minute_end, 60)
        avg_per_hour = total / span_hours
        avg_per_minute = total / span_minutes
        hour_series = [v for v in _build_series(hour_start, hour_end, timedelta(hours=1), hour_counts) if v > 0]
        minute_series = [v for v in _build_series(minute_start, minute_end, timedelta(minutes=1), minute_counts) if v > 0]

    save_dir = Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "openai_stats"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.output

    results = {
        "collection": args.collection,
        "total_documents": total,
        "min_created_at": min_dt.strftime(TIME_FORMAT) if min_dt else None,
        "max_created_at": max_dt.strftime(TIME_FORMAT) if max_dt else None,
        "span_hours": span_hours,
        "span_minutes": span_minutes,
        "avg_per_hour": round(avg_per_hour, 4),
        "avg_per_minute": round(avg_per_minute, 4),
        "time_format": TIME_FORMAT,
    }

    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    # minute_hist = Counter(minute_series)
    # print("Unique minute counts:", len(minute_hist))
    # print("Top counts:", minute_hist.most_common(10))

    if hour_series and minute_series:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        fig, ax = plt.subplots(figsize=(5, 4))
        box_colors = ["#4C78A8", "#F58518"]

        bp = ax.boxplot(
            [hour_series, minute_series],
            labels=["Per hour", "Per minute"],
            patch_artist=True,
        )
        _style_boxplot(bp, box_colors)

        for idx, (median, color) in enumerate(zip(bp["medians"], box_colors), start=1):
            median_y = float(median.get_ydata()[0])
            ax.text(
                idx + 0.08,
                median_y,
                f"{median_y:.1f}",
                color=color,
                va="center",
                ha="left",
            )

        hour_fliers = bp["fliers"][0].get_ydata() if bp["fliers"] else []
        if len(hour_fliers) > 0:
            # outlier_y = float(mean(hour_fliers))
            # ax.annotate(
            #     "between 3\nand 4 am",
            #     xy=(1.02, outlier_y),
            #     xytext=(1.45, outlier_y),  # closer to anchor -> shorter line
            #     ha="right",  # text aligns nicely when moved left
            #     va="center",
            #     color=box_colors[0],
            #     arrowprops=dict(
            #         arrowstyle="-[",
            #         mutation_scale=20,  # bigger bracket
            #         shrinkA=0,
            #         shrinkB=6,  # shortens line near bracket
            #         lw=1.2,
            #         color=box_colors[0],
            #     ),
            # )
            x = 1.055  # x position of the bracket
            dx = 0.02  # how far the "arms" extend to the right
            y0, y1 = max(250, float(min(hour_fliers))), float(max(hour_fliers))  # bracket span (lower/upper)

            ax.plot([x, x], [y0, y1], color=box_colors[0], lw=1.2)  # vertical
            ax.plot([x - dx, x], [y0, y0], color=box_colors[0], lw=1.2)  # bottom arm
            ax.plot([x - dx, x], [y1, y1], color=box_colors[0], lw=1.2)  # top arm

            ax.text(
                x + dx + 0.02,
                (y0 + y1) / 2,
                "between 3\nand 4 am",
                va="center",
                ha="left",
                color=box_colors[0],
            )

        ax.set_ylabel("# Paraphrases")
        ax.set_title("LLM-based Paraphrases")

        axins = inset_axes(ax, width="33%", height="45%", loc="upper right")
        minute_hist = Counter(minute_series)
        print("Unique minute counts:", minute_hist)
        xs = sorted(minute_hist.keys())
        ys = [minute_hist[x] for x in xs]
        axins.bar(xs, ys, color=box_colors[1], alpha=0.6, edgecolor=box_colors[1])
        axins.set_xlabel("Count")
        axins.set_yscale("log")
        axins.set_ylabel("Paraphrases/min")
        axins.set_xticks([0, 50, 100, 150, 200])
        axins.set_xlim(0, max(xs) * 1.05 if xs else 1)
        axins.tick_params(axis="x", labelrotation=45)
        plt.tight_layout()

        boxplot_path = save_dir / "non_naive_paraphrases_rate_boxplot.pdf"
        fig.savefig(boxplot_path, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"Saved boxplot to {boxplot_path}")

    print(f"Saved stats to {save_path}")


if __name__ == "__main__":
    main()
