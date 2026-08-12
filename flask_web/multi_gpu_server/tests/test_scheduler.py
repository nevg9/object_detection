from __future__ import annotations

import time
import unittest

from flask_web.multi_gpu_server.scheduler import (
    FairTaskBroker,
    InferenceTask,
    SchedulerQueueFull,
)


def task(kind: str, action: str) -> InferenceTask:
    return InferenceTask(
        kind=kind,  # type: ignore[arg-type]
        action=action,
        payload={},
        deadline=time.monotonic() + 10,
    )


class FairTaskBrokerTest(unittest.TestCase):
    def test_alternates_when_both_queues_are_backlogged(self) -> None:
        broker = FairTaskBroker(image_capacity=4, video_capacity=4)
        for index in range(2):
            broker.put(task("image", f"image-{index}"))
            broker.put(task("video", f"video-{index}"))
        self.assertEqual(
            [broker.get_next().action for _ in range(4)],  # type: ignore[union-attr]
            ["image-0", "video-0", "image-1", "video-1"],
        )

    def test_one_queue_can_use_all_slots_when_other_is_empty(self) -> None:
        broker = FairTaskBroker(image_capacity=2, video_capacity=2)
        broker.put(task("video", "video-0"))
        broker.put(task("video", "video-1"))
        self.assertEqual(broker.get_next().action, "video-0")  # type: ignore[union-attr]
        self.assertEqual(broker.get_next().action, "video-1")  # type: ignore[union-attr]

    def test_rejects_when_queue_capacity_is_reached(self) -> None:
        broker = FairTaskBroker(image_capacity=1, video_capacity=1)
        broker.put(task("image", "first"))
        with self.assertRaises(SchedulerQueueFull):
            broker.put(task("image", "second"))


if __name__ == "__main__":
    unittest.main()
