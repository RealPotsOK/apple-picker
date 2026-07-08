from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apple_picker.robot.rerun_teleop import (  # noqa: E402
    InputCommand,
    JointRangeMapper,
    POSITION_PRESET_PITCH_DEGREES,
    POSITION_PRESETS,
    RealSO100,
    TeleopConfig,
    URDFModel,
    apply_deadzone,
    differential_ik_step,
    keyboard_command_from_keys,
    merge_commands,
    update_cartesian_target,
    xbox_command_from_axes,
)


class TeleopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = TeleopConfig.load(ROOT / "config" / "so100_rerun_teleop.yaml")
        cls.model = URDFModel(cls.config.urdf_path)

    def test_urdf_chain(self) -> None:
        self.assertEqual(self.config.control_rate_hz, 50)
        self.assertEqual(self.config.viewer_rate_hz, 30)
        self.assertEqual(self.config.cartesian_speed_m_s, 0.08)
        self.assertEqual(self.config.lateral_speed_m_s, 0.10)
        self.assertEqual(self.config.vertical_speed_m_s, 0.06)
        self.assertEqual(
            self.config.controller_horizontal_speed_presets_m_s,
            {"left": 0.04, "up": 0.10, "right": 0.16, "down": 0.20},
        )
        self.assertEqual(self.config.pitch_speed_degrees_s, 45)
        self.assertEqual(self.config.roll_speed_degrees_s, 90)
        self.assertEqual(self.config.rerun_roll_offset_degrees, 180)
        self.assertEqual(self.model.root_link, "base")
        self.assertEqual(
            [joint.name for joint in self.model.joints],
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        )
        transforms, frames = self.model.forward_kinematics(self.model.midpoint_positions())
        self.assertEqual(set(transforms), self.model.links)
        self.assertEqual(set(frames), set(self.model.joints_by_name))

    def test_position_presets_have_finite_xyz_coordinates(self) -> None:
        self.assertEqual(
            set(POSITION_PRESETS),
            {"default", "middle-up", "forward", "left", "basket"},
        )
        for target in POSITION_PRESETS.values():
            self.assertEqual(len(target), 3)
            self.assertTrue(np.all(np.isfinite(target)))
        self.assertEqual(POSITION_PRESETS["basket"], (0.20, -0.25, 0.20))
        self.assertEqual(POSITION_PRESET_PITCH_DEGREES["basket"], -30.0)

    def test_wrist_roll_is_not_an_ik_joint(self) -> None:
        self.assertIn("wrist_roll", self.config.arm_joint_names)
        self.assertNotIn("wrist_roll", self.config.uncommanded_joint_names)
        self.assertIn("wrist_roll", self.config.commanded_arm_joint_names)
        self.assertNotIn("wrist_roll", self.config.ik_joint_names)
        self.assertEqual(self.config.robot_arm_p_coefficients["shoulder_pan"], 20)
        self.assertEqual(self.config.robot_arm_p_coefficients["shoulder_lift"], 40)
        self.assertEqual(self.config.robot_arm_p_coefficients["elbow_flex"], 48)
        self.assertEqual(self.config.robot_arm_p_coefficients["wrist_flex"], 40)
        self.assertEqual(self.config.robot_arm_p_coefficients["wrist_roll"], 24)

    def test_range_mapping_round_trip(self) -> None:
        mapper = JointRangeMapper(self.model, self.config)
        for name in self.config.arm_joint_names:
            for value in (-100.0, 0.0, 100.0):
                radians = mapper.normalized_to_radians(name, value)
                self.assertAlmostEqual(mapper.radians_to_normalized(name, radians), value)
        for value in (0.0, 50.0, 100.0):
            radians = mapper.normalized_to_radians("gripper", value)
            self.assertAlmostEqual(mapper.radians_to_normalized("gripper", radians), value)

    def test_ik_moves_all_cartesian_directions_without_wrist_roll(self) -> None:
        positions = self.model.midpoint_positions()
        local_tip = np.append(np.asarray(self.config.end_effector_offset_xyz), 1.0)
        start = (
            self.model.forward_kinematics(positions)[0][self.config.end_effector_link] @ local_tip
        )[:3]
        for direction in (
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ):
            desired = np.asarray(direction, dtype=float) * 0.001
            updated, success = differential_ik_step(
                self.model,
                positions,
                desired,
                self.config.ik_joint_names,
                self.config.end_effector_link,
                self.config.end_effector_offset_xyz,
                self.config.ik_damping,
                1 / self.config.control_rate_hz,
                self.config.max_joint_step_normalized,
            )
            end = (
                self.model.forward_kinematics(updated)[0][self.config.end_effector_link] @ local_tip
            )[:3]
            self.assertTrue(success)
            self.assertGreater(float(np.dot(end - start, desired)), 0.0)
            self.assertEqual(updated["wrist_roll"], positions["wrist_roll"])

    def test_pitch_ik_rotates_claw_without_wrist_roll(self) -> None:
        positions = self.model.midpoint_positions()
        start_pitch = self.model.pitch_value(positions, self.config.pitch_joint_names)
        start_tip = self.model.point_position(
            positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        updated, success = differential_ik_step(
            self.model,
            positions,
            np.zeros(3),
            self.config.ik_joint_names,
            self.config.end_effector_link,
            self.config.end_effector_offset_xyz,
            self.config.ik_damping,
            1 / self.config.control_rate_hz,
            self.config.max_joint_step_normalized,
            desired_pitch_delta=np.deg2rad(2),
            pitch_joint_names=self.config.pitch_joint_names,
            pitch_weight_m_per_rad=self.config.pitch_weight_m_per_rad,
        )
        end_pitch = self.model.pitch_value(updated, self.config.pitch_joint_names)
        end_tip = self.model.point_position(
            updated, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        self.assertTrue(success)
        self.assertGreater(end_pitch, start_pitch)
        self.assertLess(np.linalg.norm(end_tip - start_tip), 0.001)
        self.assertEqual(updated["wrist_roll"], positions["wrist_roll"])

    def test_ik_keeps_moving_when_elbow_is_at_upper_limit(self) -> None:
        positions = self.model.midpoint_positions()
        positions["elbow_flex"] = self.model.joints_by_name["elbow_flex"].upper
        start_tip = self.model.point_position(
            positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        requested_delta = np.array([0.0, 0.01, 0.0])
        updated, success = differential_ik_step(
            self.model,
            positions,
            requested_delta,
            self.config.ik_joint_names,
            self.config.end_effector_link,
            self.config.end_effector_offset_xyz,
            self.config.ik_damping,
            1 / self.config.control_rate_hz,
            self.config.max_joint_step_normalized,
            pitch_joint_names=self.config.pitch_joint_names,
            pitch_weight_m_per_rad=self.config.pitch_weight_m_per_rad,
        )
        end_tip = self.model.point_position(
            updated, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        self.assertTrue(success)
        self.assertGreater(float(np.dot(end_tip - start_tip, requested_delta)), 0.0)

    def test_held_up_target_corrects_sideways_drift(self) -> None:
        positions = self.model.midpoint_positions()
        start = self.model.point_position(
            positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        target = start.copy()
        step_distance = self.config.vertical_speed_m_s / self.config.control_rate_hz
        # Test drift over a fixed 12 cm move, independent of configured speed.
        step_count = round(0.12 / step_distance)
        for _ in range(step_count):
            current = self.model.point_position(
                positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
            )
            target[2] += step_distance
            error = target - current
            norm = np.linalg.norm(error)
            if norm > self.config.max_cartesian_error_m:
                error *= self.config.max_cartesian_error_m / norm
            positions, success = differential_ik_step(
                self.model,
                positions,
                error,
                self.config.ik_joint_names,
                self.config.end_effector_link,
                self.config.end_effector_offset_xyz,
                self.config.ik_damping,
                1 / self.config.control_rate_hz,
                self.config.max_joint_step_normalized,
            )
            self.assertTrue(success)
        end = self.model.point_position(
            positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        self.assertGreater(end[2] - start[2], 0.08)
        self.assertLess(np.linalg.norm((end - start)[:2]), 0.01)

    def test_released_key_keeps_converging_to_saved_target(self) -> None:
        positions = self.model.midpoint_positions()
        start = self.model.point_position(
            positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        target = start + np.array([0.0, -0.01, 0.0])
        # This simulates one key tap followed by idle ticks. The target remains
        # fixed while the planned joint state continues to approach it.
        for _ in range(40):
            current = self.model.point_position(
                positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
            )
            error = target - current
            if np.linalg.norm(error) <= self.config.position_tolerance_m:
                break
            positions, success = differential_ik_step(
                self.model,
                positions,
                error,
                self.config.ik_joint_names,
                self.config.end_effector_link,
                self.config.end_effector_offset_xyz,
                self.config.ik_damping,
                1 / self.config.control_rate_hz,
                self.config.max_joint_step_normalized,
            )
            self.assertTrue(success)
        end = self.model.point_position(
            positions, self.config.end_effector_link, self.config.end_effector_offset_xyz
        )
        self.assertLess(np.linalg.norm(target - end), self.config.position_tolerance_m)

    def test_input_mappings(self) -> None:
        keyboard = keyboard_command_from_keys({"up", "left", "u", "w", "o"})
        np.testing.assert_array_equal(keyboard.motion, [1, -1, 1])
        self.assertEqual(keyboard.pitch, 1)
        self.assertEqual(keyboard.gripper, 1)
        xbox = xbox_command_from_axes(
            1,
            -1,
            -0.5,
            0,
            0.75,
            left_bumper=True,
            go_to_basket=True,
            right_x=0.4,
            horizontal_speed_m_s=0.16,
        )
        np.testing.assert_array_equal(xbox.motion, [-1, 1, -0.5])
        self.assertEqual(xbox.pitch, -1)
        self.assertEqual(xbox.roll, -0.4)
        self.assertEqual(xbox.gripper, 0.75)
        self.assertTrue(xbox.go_to_basket)
        self.assertEqual(xbox.horizontal_speed_m_s, 0.16)
        stop = keyboard_command_from_keys({"emergency_stop"})
        self.assertTrue(stop.emergency_stop)

    def test_horizontal_motion_is_faster_than_vertical_motion(self) -> None:
        self.assertLessEqual(
            self.config.tap_step_m,
            self.config.vertical_speed_m_s / self.config.control_rate_hz,
        )
        origin = np.zeros(3)
        horizontal, _ = update_cartesian_target(
            origin,
            origin,
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            self.config.cartesian_speed_xyz_m_s,
            self.config.tap_step_m,
            1.0,
        )
        forward, _ = update_cartesian_target(
            origin,
            origin,
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            self.config.cartesian_speed_xyz_m_s,
            self.config.tap_step_m,
            1.0,
        )
        vertical, _ = update_cartesian_target(
            origin,
            origin,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            self.config.cartesian_speed_xyz_m_s,
            self.config.tap_step_m,
            1.0,
        )
        self.assertAlmostEqual(horizontal[0], 0.10)
        self.assertAlmostEqual(forward[1], -0.08)
        self.assertAlmostEqual(vertical[2], 0.06)

    def test_deadzone_and_merge(self) -> None:
        self.assertEqual(apply_deadzone(0.1, 0.15), 0)
        merged = merge_commands(
            InputCommand(motion=np.array([1.0, 0.0, 0.0]), gripper=1),
            InputCommand(motion=np.array([1.0, 1.0, 0.0]), gripper=1, quit=True),
        )
        self.assertLessEqual(np.linalg.norm(merged.motion), 1)
        self.assertEqual(merged.gripper, 1)
        self.assertTrue(merged.quit)
        emergency = merge_commands(InputCommand(), InputCommand(emergency_stop=True))
        self.assertTrue(emergency.emergency_stop)

    def test_new_direction_uses_desired_model_not_measured_feedback(self) -> None:
        measured_hardware = np.array([0.1, 0.2, 0.3])
        desired_model = np.array([0.2, 0.3, 0.4])
        target = desired_model.copy()
        forward = np.array([0.0, -1.0, 0.0])
        idle = np.zeros(3)
        target, replaced = update_cartesian_target(
            target,
            desired_model,
            forward,
            idle,
            self.config.cartesian_speed_m_s,
            self.config.tap_step_m,
            1 / self.config.control_rate_hz,
        )
        self.assertTrue(replaced)
        np.testing.assert_allclose(
            target,
            desired_model + np.array([0.0, -self.config.tap_step_m, 0.0]),
        )
        self.assertFalse(np.allclose(target, measured_hardware))

        left = np.array([1.0, 0.0, 0.0])
        target, replaced = update_cartesian_target(
            target,
            desired_model,
            left,
            forward,
            self.config.cartesian_speed_m_s,
            self.config.tap_step_m,
            1 / self.config.control_rate_hz,
        )
        self.assertTrue(replaced)
        np.testing.assert_allclose(
            target,
            desired_model + np.array([self.config.tap_step_m, 0.0, 0.0]),
        )

    def test_real_action_includes_wrist_roll(self) -> None:
        mapper = JointRangeMapper(self.model, self.config)

        class FakeRobot:
            is_connected = True

            def __init__(self) -> None:
                self.last_action: dict[str, float] | None = None
                self.disconnected = False

            def get_observation(self) -> dict[str, float]:
                return {
                    **{f"{name}.pos": 0.0 for name in self.config_names},
                    "gripper.pos": 50.0,
                }

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                self.last_action = action
                return action

            def disconnect(self) -> None:
                self.disconnected = True

        fake = FakeRobot()
        fake.config_names = self.config.arm_joint_names
        adapter = RealSO100(self.config, mapper)
        adapter.robot = fake
        measured = adapter.read()
        accepted, was_clamped = adapter.send(measured)
        self.assertFalse(was_clamped)
        self.assertEqual(set(accepted), set(measured))
        self.assertIsNotNone(fake.last_action)
        self.assertIn("wrist_roll.pos", fake.last_action)
        self.assertIn("wrist_flex.pos", fake.last_action)
        adapter.close()
        self.assertTrue(fake.disconnected)

    def test_real_safety_clamp_uses_latest_feedback(self) -> None:
        mapper = JointRangeMapper(self.model, self.config)

        class FakeRobot:
            is_connected = True

            def __init__(self) -> None:
                self.action: dict[str, float] | None = None

            def get_observation(self) -> dict[str, float]:
                return {
                    **{f"{name}.pos": 0.0 for name in self_names},
                    "gripper.pos": 50.0,
                }

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                self.action = action
                return action

            def disconnect(self) -> None:
                pass

        self_names = self.config.arm_joint_names
        fake = FakeRobot()
        adapter = RealSO100(self.config, mapper)
        adapter.robot = fake
        measured = adapter.read()
        far_target = dict(measured)
        far_target["shoulder_lift"] = self.model.joints_by_name["shoulder_lift"].upper
        _, was_clamped = adapter.send(far_target)
        self.assertTrue(was_clamped)
        self.assertIsNotNone(fake.action)
        self.assertLessEqual(
            abs(fake.action["shoulder_lift.pos"]),
            self.config.robot_max_relative_target,
        )

    def test_gain_is_applied_only_to_commanded_arm_motors(self) -> None:
        mapper = JointRangeMapper(self.model, self.config)

        class FakeBus:
            def __init__(self) -> None:
                self.writes: list[tuple[str, str, int]] = []

            def write(self, register: str, motor: str, value: int) -> None:
                self.writes.append((register, motor, value))

        class FakeRobot:
            bus = FakeBus()

        adapter = RealSO100(self.config, mapper)
        adapter.robot = FakeRobot()
        adapter.apply_arm_gain()
        written_motors = {motor for _, motor, _ in adapter.robot.bus.writes}
        self.assertEqual(written_motors, set(self.config.commanded_arm_joint_names))
        written_gains = {
            motor: value
            for register, motor, value in adapter.robot.bus.writes
            if register == "P_Coefficient"
        }
        written_velocities = {
            motor: value
            for register, motor, value in adapter.robot.bus.writes
            if register == "Goal_Velocity"
        }
        self.assertEqual(written_gains, self.config.robot_arm_p_coefficients)
        self.assertEqual(
            written_velocities,
            dict.fromkeys(self.config.commanded_arm_joint_names, 0),
        )
        self.assertIn("wrist_roll", written_motors)
        self.assertNotIn("gripper", written_motors)


if __name__ == "__main__":
    unittest.main()
