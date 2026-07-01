from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class PoseTarget:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class IKResult:
    reachable: bool
    joint_positions: dict[str, float]
    message: str = ""


class IKSolver(Protocol):
    def solve(
        self,
        target: PoseTarget,
        current_joints: Mapping[str, float],
    ) -> IKResult:
        """Return joint positions for the requested claw pose."""


class ArmDriver(Protocol):
    def read_joints(self) -> dict[str, float]:
        """Read current arm joint positions."""

    def move_joints(self, joint_positions: Mapping[str, float]) -> None:
        """Move the arm to the requested joint positions."""

    def stop(self) -> None:
        """Stop motion and hold the current position if the driver supports it."""


class ClawPoseController:
    def __init__(self, ik_solver: IKSolver, arm_driver: ArmDriver) -> None:
        self.ik_solver = ik_solver
        self.arm_driver = arm_driver

    def move_to_pose(self, target: PoseTarget) -> bool:
        current_joints = self.arm_driver.read_joints()
        result = self.ik_solver.solve(target, current_joints)

        if not result.reachable:
            self.arm_driver.stop()
            return False

        self.arm_driver.move_joints(result.joint_positions)
        return True
