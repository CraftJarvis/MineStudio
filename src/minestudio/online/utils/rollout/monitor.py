from typing import List
from collections import deque, defaultdict
import time
from rich.table import Table
import rich

class MovingStat:
    """
    Calculates a moving statistic (average) over a specified duration.

    :param duration: The duration in seconds over which to calculate the statistic.
    """
    def __init__(self, duration: int):
        self.duration = duration
        self.data_queue = deque()
        self.sum = 0.0

    def _pop(self):
        """
        Removes data points from the queue that are older than the specified duration.
        """
        while len(self.data_queue) > 0 and time.time() - self.data_queue[0][0] > self.duration:
            self.sum -= self.data_queue[0][1]
            self.data_queue.popleft()

    def update(self, x: float):
        """
        Adds a new data point to the calculation.

        :param x: The new data point.
        """
        self.data_queue.append((time.time(), x))
        self.sum += x
        self._pop()
        
    def average(self):
        """
        Calculates the average of the data points currently within the duration window.

        :returns: The average of the data points, or float('nan') if no data is available.
        """
        self._pop()
        if len(self.data_queue) == 0:
            return float('nan')
        return self.sum / len(self.data_queue)

class PipelineMonitor:
    """
    Monitors the time spent in different stages of a pipeline.

    :param stages: A list of strings representing the names of the pipeline stages.
    :param duration: The duration in seconds over which to calculate moving statistics for each stage.
    """
    def __init__(self, stages: List[str], duration: int = 300):
        self.stages = stages
        self.records = defaultdict(lambda: [MovingStat(duration) for _ in stages])
        self.last_stage = defaultdict(lambda: len(stages) - 1)
        self.last_updated_time = {}

    def report_enter(self, stage: str, pipeline_id="default"):
        """
        Reports that a pipeline has entered a new stage.

        :param stage: The name of the stage being entered.
        :param pipeline_id: The unique identifier of the pipeline instance.
        :raises AssertionError: If the reported stage is not the expected next stage.
        """
        stage_idx = self.stages.index(stage)
        assert stage_idx == (self.last_stage[pipeline_id] + 1) % len(self.stages)
        if pipeline_id in self.last_updated_time:
            self.records[pipeline_id][self.last_stage[pipeline_id]].update(time.time() - self.last_updated_time[pipeline_id])
        self.last_updated_time[pipeline_id] = time.time()
        self.last_stage[pipeline_id] = stage_idx

    def print(self):
        """
        Prints a table summarizing the average time spent in each stage for all monitored pipelines.
        """
        table = Table()

        table.add_column("id")

        for idx in range(len(self.stages)):
            table.add_column(f"{self.stages[idx]}->{self.stages[(idx + 1) % len(self.stages)]}")
        
        table.add_column("total")

        rows = []
        for pipeline_id in self.records.keys():
            record = self.records[pipeline_id]
            row = [stat.average() for stat in record]
            row.append(sum(row))
            rows.append(row)
            table.add_row(pipeline_id,
                * ["%.6f" % r for r in row]
            )
        summary = [sum(col) / len(col) for col in zip(*rows)]
        table.add_row("Summary",
            * ["%.6f" % r for r in summary]
        )

        rich.print(table)

if __name__ == "__main__":
    pipeline = PipelineMonitor(['a', 'b', 'c'])
    pipeline.report_enter('a')
    pipeline.report_enter('b')
    pipeline.report_enter('c')
    time.sleep(1)
    pipeline.report_enter('a')
    pipeline.print()