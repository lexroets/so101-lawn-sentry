"""Non-operational actuator boundary.

This module deliberately contains no torque-enable or goal-position write.
It exists so the rest of the application can be integrated and tested.
"""

class ActuationDisabled(RuntimeError):
    pass


class DisabledBaseActuator:
    def __init__(self, *args, **kwargs):
        self.stop_latched = True

    def emergency_stop(self):
        self.stop_latched = True

    def read_position(self):
        raise ActuationDisabled(
            "Base actuation/feedback backend is intentionally not implemented "
            "in this package."
        )

    def enable(self):
        raise ActuationDisabled(
            "Torque enable is intentionally unavailable."
        )

    def move_one_bounded_step(self, target_raw):
        raise ActuationDisabled(
            f"No motor command sent. Proposed target was {target_raw}."
        )
