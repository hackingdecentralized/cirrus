#!/usr/bin/env python3
from __future__ import annotations

import json
import re

from dataclasses import dataclass, asdict
from typing import List

import argparse

start_pattern = r'\.*Start: {3}(.*)'
end_pattern = r'End: {5}(.*) \.*(\d+\.?\d*)([mµn]?s)'

@dataclass
class Task:
    name: str
    duration: float
    send_time: float
    recv_time: float
    compute_time: float
    send_round: int
    recv_round: int
    sub_tasks: List[Task]

    def __init__(self, name: str, duration: float, tasks: List[Task] = None):
        self.name = name
        self.duration = duration
        self.sub_tasks = tasks or []

        self.send_time = self.recv_time = self.send_round = self.recv_round = 0
        for task in self.sub_tasks:
            self.send_time += task.send_time
            self.recv_time += task.recv_time
            self.send_round += task.send_round
            self.recv_round += task.recv_round

        if "send" in self.name:
            assert self.sub_tasks == [], "Send tasks cannot have sub-tasks"
            self.send_time = self.duration
            self.send_round = 1
        elif "recv" in self.name:
            assert self.sub_tasks == [], "Recv tasks cannot have sub-tasks"
            self.recv_time = self.duration
            self.recv_round = 1

        self.compute_time = self.duration - self.send_time - self.recv_time
        assert self.compute_time >= 0, f"Negative compute time: {self.compute_time}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_file", type=str, help="Log file to analyze")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    args = parser.parse_args()

    tasks = []

    with open(args.log_file, "r") as f:

        def create_task(line: str) -> Task:
            matches = re.search(start_pattern, line)
            assert matches, f"Invalid line: {line}"

            name = matches.group(1)
            line = f.readline()
            tasks = []
            while re.search(start_pattern, line):
                tasks.append(create_task(line))
                line = f.readline()

            matches = re.search(end_pattern, line)
            assert matches and matches.group(1) == name, f"Invalid line: {line}"

            duration = float(matches.group(2))
            unit = matches.group(3)
            if unit == "ms":
                duration /= 1000
            elif unit == "µs":
                duration /= 1000000
            elif unit == "ns":
                duration /= 1000000000
            elif unit == "s":
                pass
            else:
                assert False, f"Invalid unit: {unit}"

            return Task(name, duration, tasks)

        line = f.readline()
        while line:
            if re.search(start_pattern, line):
                tasks.append(create_task(line))
            line = f.readline()


    with open(args.output, "w") as f:
        print(json.dumps([asdict(task) for task in tasks], indent=4), file=f)
