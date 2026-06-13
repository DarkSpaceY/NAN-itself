import time
from dataclasses import dataclass
from enum import Enum


class TriggerType(Enum):
    FEEDBACK = "feedback"
    FAILURE = "failure"
    CURIOSITY = "curiosity"
    DENSITY = "density"
    TIMED = "timed"


@dataclass
class LearningTrigger:
    type: TriggerType
    priority: int = 5
    reason: str = ""


_DEFAULT_PRIORITIES = {
    TriggerType.FEEDBACK: 1,
    TriggerType.FAILURE: 2,
    TriggerType.CURIOSITY: 3,
    TriggerType.DENSITY: 4,
    TriggerType.TIMED: 5,
}


class TriggerManager:
    def __init__(
        self,
        failure_threshold: int = 3,
        density_threshold: int = 100,
        timed_interval_hours: int = 24,
    ):
        self.failure_threshold = failure_threshold
        self.density_threshold = density_threshold
        self.timed_interval_hours = timed_interval_hours

        self._failure_counts: dict[str, int] = {}
        self._last_timed_trigger: float = time.time()
        self._feedback_requested: bool = False
        self._curiosity_triggered: bool = False
        self._curiosity_reason: str = ""

    def trigger_feedback(self):
        self._feedback_requested = True

    def trigger_curiosity(self, reason: str = ""):
        self._curiosity_triggered = True
        self._curiosity_reason = reason

    def record_failure(self, task_type: str):
        self._failure_counts[task_type] = self._failure_counts.get(task_type, 0) + 1

    def check_timed(self) -> bool:
        elapsed = time.time() - self._last_timed_trigger
        interval_seconds = self.timed_interval_hours * 3600
        if elapsed >= interval_seconds:
            self._last_timed_trigger = time.time()
            return True
        return False

    def check_density(self, memcell_count: int) -> bool:
        return memcell_count >= self.density_threshold

    def check_failure(self, task_type: str) -> bool:
        count = self._failure_counts.get(task_type, 0)
        if count >= self.failure_threshold:
            del self._failure_counts[task_type]
            return True
        return False

    def get_active_triggers(self, memcell_count: int = 0) -> list[LearningTrigger]:
        triggers: list[LearningTrigger] = []

        if self._feedback_requested:
            triggers.append(
                LearningTrigger(
                    type=TriggerType.FEEDBACK,
                    priority=_DEFAULT_PRIORITIES[TriggerType.FEEDBACK],
                    reason="User provided explicit feedback",
                )
            )
            self._feedback_requested = False

        for task_type, count in list(self._failure_counts.items()):
            if count >= self.failure_threshold:
                triggers.append(
                    LearningTrigger(
                        type=TriggerType.FAILURE,
                        priority=_DEFAULT_PRIORITIES[TriggerType.FAILURE],
                        reason=f"Task type '{task_type}' failed {count} times",
                    )
                )
                del self._failure_counts[task_type]

        if self._curiosity_triggered:
            triggers.append(
                LearningTrigger(
                    type=TriggerType.CURIOSITY,
                    priority=_DEFAULT_PRIORITIES[TriggerType.CURIOSITY],
                    reason=self._curiosity_reason or "Agent proactively exploring",
                )
            )
            self._curiosity_triggered = False
            self._curiosity_reason = ""

        if self.check_density(memcell_count):
            triggers.append(
                LearningTrigger(
                    type=TriggerType.DENSITY,
                    priority=_DEFAULT_PRIORITIES[TriggerType.DENSITY],
                    reason=f"MemCell count {memcell_count} >= threshold {self.density_threshold}",
                )
            )

        if self.check_timed():
            triggers.append(
                LearningTrigger(
                    type=TriggerType.TIMED,
                    priority=_DEFAULT_PRIORITIES[TriggerType.TIMED],
                    reason="Scheduled learning cycle",
                )
            )

        triggers.sort(key=lambda t: t.priority)
        return triggers