from __future__ import annotations

import argparse
from pathlib import Path

from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions, TopicMetadata


def filter_crash_segment(
    input_bag: str | Path,
    output_bag: str | Path,
    exclude_times: list[float],
    window_seconds: float = 0.15,
) -> None:
    input_bag = Path(input_bag)
    output_bag = Path(output_bag)
    if output_bag.exists():
        raise FileExistsError(f"Output bag already exists: {output_bag}")

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(input_bag.parent), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    excluded_ns: set[int] = set()
    for t in exclude_times:
        excluded_ns.add(int(round(t * 1e9)))

    excluded_ranges: list[tuple[int, int]] = []
    for ts in excluded_ns:
        excluded_ranges.append((int(round(ts - window_seconds * 1e9)), int(round(ts + window_seconds * 1e9))))

    writer = SequentialWriter()
    bag_storage_options = StorageOptions(uri=str(output_bag), storage_id="sqlite3")
    writer.open(bag_storage_options, converter_options)

    created_topics: set[str] = set()
    written = 0
    while reader.has_next():
        topic, rawdata, timestamp_ns = reader.read_next()
        if any(lower <= timestamp_ns <= upper for lower, upper in excluded_ranges):
            continue
        if topic not in created_topics:
            msg_type = topic_types[topic]
            writer.create_topic(TopicMetadata(name=topic, type=msg_type, serialization_format="cdr", offered_qos_profiles=""))
            created_topics.add(topic)
        writer.write(topic, rawdata, timestamp_ns)
        written += 1

    writer.close()
    print(f"Wrote filtered bag to {output_bag} with {written} messages")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cleaned ROS2 bag without the crash-related messages")
    parser.add_argument("--input_bag", required=True)
    parser.add_argument("--output_bag", required=True)
    parser.add_argument("--exclude_times", nargs="+", type=float, required=True)
    parser.add_argument("--window_seconds", type=float, default=0.15)
    args = parser.parse_args()

    filter_crash_segment(
        input_bag=args.input_bag,
        output_bag=args.output_bag,
        exclude_times=args.exclude_times,
        window_seconds=args.window_seconds,
    )


if __name__ == "__main__":
    main()
