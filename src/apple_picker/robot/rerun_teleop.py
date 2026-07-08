"""Control a virtual SO100, or a real SO100 mirrored in Rerun."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "so100_rerun_teleop.yaml"


def _vector(text: str | None, length: int = 3) -> np.ndarray:
    if text is None:
        return np.zeros(length, dtype=float)
    values = np.asarray([float(value) for value in text.split()], dtype=float)
    if values.shape != (length,):
        raise ValueError(f"Expected {length} values, got {text!r}")
    return values


def rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_matrix(axis: Iterable[float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_array)
    if norm == 0:
        return np.eye(3)
    x, y, z = axis_array / norm
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    return np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ],
        dtype=float,
    )


def transform_matrix(translation: Iterable[float], rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float

    @property
    def span(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class LinkVisual:
    link: str
    index: int
    mesh_path: Path
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    rgba: tuple[int, int, int, int] | None


class URDFModel:
    def __init__(self, path: Path, entity_prefix: str = "so100") -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.path}")

        xml = ET.parse(self.path).getroot()
        self.entity_prefix = entity_prefix.strip("/")
        self.links = {link.attrib["name"] for link in xml.findall("link")}
        self.materials = self._read_materials(xml)
        self.joints = self._read_joints(xml)
        self.joints_by_name = {joint.name: joint for joint in self.joints}
        self.children: dict[str, list[JointSpec]] = {link: [] for link in self.links}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)

        roots = sorted(self.links - {joint.child for joint in self.joints})
        if len(roots) != 1:
            raise ValueError(f"Expected one root link, found {roots}")
        self.root_link = roots[0]
        self.link_paths: dict[str, str] = {}
        self._assign_paths(self.root_link, f"{self.entity_prefix}/{self.root_link}")
        self.visuals = self._read_visuals(xml)

    @staticmethod
    def _read_materials(xml: ET.Element) -> dict[str, tuple[int, int, int, int]]:
        result: dict[str, tuple[int, int, int, int]] = {}
        for material in xml.findall("material"):
            color = material.find("color")
            if color is None or "rgba" not in color.attrib:
                continue
            rgba = _vector(color.attrib["rgba"], 4)
            result[material.attrib["name"]] = tuple(
                int(round(np.clip(channel, 0.0, 1.0) * 255)) for channel in rgba
            )
        return result

    @staticmethod
    def _read_joints(xml: ET.Element) -> list[JointSpec]:
        result = []
        for element in xml.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"Joint {element.attrib.get('name')} has no parent or child")
            joint_type = element.attrib.get("type", "fixed")
            origin = element.find("origin")
            axis = element.find("axis")
            limit = element.find("limit")
            if joint_type == "revolute" and limit is None:
                raise ValueError(f"Joint {element.attrib['name']} has no limits")
            result.append(
                JointSpec(
                    name=element.attrib["name"],
                    joint_type=joint_type,
                    parent=parent.attrib["link"],
                    child=child.attrib["link"],
                    origin_xyz=_vector(origin.attrib.get("xyz") if origin is not None else None),
                    origin_rpy=_vector(origin.attrib.get("rpy") if origin is not None else None),
                    axis=_vector(axis.attrib.get("xyz") if axis is not None else "1 0 0"),
                    lower=float(limit.attrib.get("lower", 0.0)) if limit is not None else 0.0,
                    upper=float(limit.attrib.get("upper", 0.0)) if limit is not None else 0.0,
                    velocity=float(limit.attrib.get("velocity", "inf")) if limit is not None else 0.0,
                )
            )
        return result

    def _read_visuals(self, xml: ET.Element) -> list[LinkVisual]:
        result = []
        for link in xml.findall("link"):
            link_name = link.attrib["name"]
            for index, visual in enumerate(link.findall("visual")):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                origin = visual.find("origin")
                material = visual.find("material")
                rgba = None
                if material is not None:
                    inline = material.find("color")
                    if inline is not None and "rgba" in inline.attrib:
                        color = _vector(inline.attrib["rgba"], 4)
                        rgba = tuple(int(round(np.clip(value, 0.0, 1.0) * 255)) for value in color)
                    elif "name" in material.attrib:
                        rgba = self.materials.get(material.attrib["name"])
                result.append(
                    LinkVisual(
                        link=link_name,
                        index=index,
                        mesh_path=(self.path.parent / mesh.attrib["filename"]).resolve(),
                        origin_xyz=_vector(origin.attrib.get("xyz") if origin is not None else None),
                        origin_rpy=_vector(origin.attrib.get("rpy") if origin is not None else None),
                        rgba=rgba,
                    )
                )
        return result

    def _assign_paths(self, link: str, path: str) -> None:
        self.link_paths[link] = path
        for joint in self.children.get(link, []):
            self._assign_paths(joint.child, f"{path}/{joint.child}")

    def validate(self, joint_names: Iterable[str], end_effector: str) -> None:
        missing = set(joint_names) - self.joints_by_name.keys()
        if missing:
            raise ValueError(f"URDF is missing joints: {sorted(missing)}")
        if end_effector not in self.links:
            raise ValueError(f"URDF is missing link {end_effector!r}")

    def midpoint_positions(self) -> dict[str, float]:
        return {
            joint.name: (joint.lower + joint.upper) / 2.0
            for joint in self.joints
            if joint.joint_type == "revolute"
        }

    def local_joint_transform(self, joint: JointSpec, positions: Mapping[str, float]) -> np.ndarray:
        origin = transform_matrix(joint.origin_xyz, rpy_matrix(joint.origin_rpy))
        if joint.joint_type != "revolute":
            return origin
        rotation = transform_matrix((0, 0, 0), axis_angle_matrix(joint.axis, positions[joint.name]))
        return origin @ rotation

    def forward_kinematics(
        self, positions: Mapping[str, float]
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
        transforms = {self.root_link: np.eye(4, dtype=float)}
        joint_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def walk(parent: str) -> None:
            parent_world = transforms[parent]
            for joint in self.children.get(parent, []):
                origin_world = parent_world @ transform_matrix(joint.origin_xyz, rpy_matrix(joint.origin_rpy))
                axis_world = origin_world[:3, :3] @ joint.axis
                axis_world /= np.linalg.norm(axis_world)
                joint_frames[joint.name] = (origin_world[:3, 3].copy(), axis_world)
                transforms[joint.child] = parent_world @ self.local_joint_transform(joint, positions)
                walk(joint.child)

        walk(self.root_link)
        return transforms, joint_frames

    def position_jacobian(
        self,
        positions: Mapping[str, float],
        joint_names: list[str],
        end_effector: str,
        end_effector_offset: Iterable[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        transforms, frames = self.forward_kinematics(positions)
        local_point = np.append(np.asarray(end_effector_offset, dtype=float), 1.0)
        end_position = (transforms[end_effector] @ local_point)[:3]
        columns = []
        for name in joint_names:
            joint_position, axis = frames[name]
            columns.append(np.cross(axis, end_position - joint_position))
        return np.column_stack(columns), end_position

    def point_position(
        self,
        positions: Mapping[str, float],
        link: str,
        offset: Iterable[float],
    ) -> np.ndarray:
        transforms, _ = self.forward_kinematics(positions)
        return (transforms[link] @ np.append(np.asarray(offset, dtype=float), 1.0))[:3]

    @staticmethod
    def pitch_value(positions: Mapping[str, float], pitch_joint_names: Iterable[str]) -> float:
        """Return relative claw pitch; fixed URDF frame rotations cancel in target differences."""
        return sum(positions[name] for name in pitch_joint_names)


@dataclass(frozen=True)
class TeleopConfig:
    urdf_path: Path
    control_rate_hz: float
    viewer_rate_hz: float
    cartesian_speed_m_s: float
    lateral_speed_m_s: float
    vertical_speed_m_s: float
    controller_horizontal_speed_presets_m_s: dict[str, float]
    tap_step_m: float
    max_cartesian_error_m: float
    position_tolerance_m: float
    pitch_speed_degrees_s: float
    pitch_tolerance_degrees: float
    max_pitch_error_degrees: float
    roll_speed_degrees_s: float
    rerun_roll_offset_degrees: float
    pitch_weight_m_per_rad: float
    gripper_speed_rad_s: float
    ik_damping: float
    gamepad_deadzone: float
    max_joint_step_normalized: float
    end_effector_link: str
    end_effector_offset_xyz: list[float]
    arm_joint_names: list[str]
    uncommanded_joint_names: list[str]
    ik_excluded_joint_names: list[str]
    pitch_joint_names: list[str]
    gripper_joint_name: str
    roll_joint_name: str
    joint_directions: dict[str, int]
    robot_port: str
    robot_id: str
    robot_arm_p_coefficients: dict[str, int]
    robot_max_relative_target: float

    @property
    def ik_joint_names(self) -> list[str]:
        excluded = set(self.uncommanded_joint_names) | set(self.ik_excluded_joint_names)
        return [name for name in self.arm_joint_names if name not in excluded]

    @property
    def commanded_arm_joint_names(self) -> list[str]:
        return [name for name in self.arm_joint_names if name not in self.uncommanded_joint_names]

    @property
    def cartesian_speed_xyz_m_s(self) -> np.ndarray:
        return np.array(
            [self.lateral_speed_m_s, self.cartesian_speed_m_s, self.vertical_speed_m_s],
            dtype=float,
        )

    @classmethod
    def load(cls, path: Path) -> "TeleopConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        urdf_path = Path(data["urdf_path"])
        if not urdf_path.is_absolute():
            urdf_path = PROJECT_ROOT / urdf_path
        directions = {name: int(value) for name, value in data["joint_directions"].items()}
        if any(value not in {-1, 1} for value in directions.values()):
            raise ValueError("Joint directions must be either +1 or -1")
        return cls(
            urdf_path=urdf_path,
            control_rate_hz=float(data["control_rate_hz"]),
            viewer_rate_hz=float(data["viewer_rate_hz"]),
            cartesian_speed_m_s=float(data["cartesian_speed_m_s"]),
            lateral_speed_m_s=float(data["lateral_speed_m_s"]),
            vertical_speed_m_s=float(data["vertical_speed_m_s"]),
            controller_horizontal_speed_presets_m_s={
                name: float(value)
                for name, value in data["controller_horizontal_speed_presets_m_s"].items()
            },
            tap_step_m=float(data["tap_step_m"]),
            max_cartesian_error_m=float(data["max_cartesian_error_m"]),
            position_tolerance_m=float(data["position_tolerance_m"]),
            pitch_speed_degrees_s=float(data["pitch_speed_degrees_s"]),
            pitch_tolerance_degrees=float(data["pitch_tolerance_degrees"]),
            max_pitch_error_degrees=float(data["max_pitch_error_degrees"]),
            roll_speed_degrees_s=float(data["roll_speed_degrees_s"]),
            rerun_roll_offset_degrees=float(data["rerun_roll_offset_degrees"]),
            pitch_weight_m_per_rad=float(data["pitch_weight_m_per_rad"]),
            gripper_speed_rad_s=float(data["gripper_speed_rad_s"]),
            ik_damping=float(data["ik_damping"]),
            gamepad_deadzone=float(data["gamepad_deadzone"]),
            max_joint_step_normalized=float(data["max_joint_step_normalized"]),
            end_effector_link=str(data["end_effector_link"]),
            end_effector_offset_xyz=[float(value) for value in data["end_effector_offset_xyz"]],
            arm_joint_names=list(data["arm_joint_names"]),
            uncommanded_joint_names=list(data.get("uncommanded_joint_names", [])),
            ik_excluded_joint_names=list(data.get("ik_excluded_joint_names", [])),
            pitch_joint_names=list(data["pitch_joint_names"]),
            gripper_joint_name=str(data["gripper_joint_name"]),
            roll_joint_name=str(data["roll_joint_name"]),
            joint_directions=directions,
            robot_port=str(data["robot"]["port"]),
            robot_id=str(data["robot"]["id"]),
            robot_arm_p_coefficients={
                name: int(value)
                for name, value in data["robot"]["arm_p_coefficients"].items()
            },
            robot_max_relative_target=float(data["robot"]["max_relative_target"]),
        )


class JointRangeMapper:
    """Convert LeRobot normalized positions to the source URDF's radians."""

    def __init__(self, model: URDFModel, config: TeleopConfig) -> None:
        self.model = model
        self.config = config

    def normalized_to_radians(self, name: str, value: float) -> float:
        joint = self.model.joints_by_name[name]
        ratio = value / 100.0 if name == self.config.gripper_joint_name else (value + 100.0) / 200.0
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if self.config.joint_directions.get(name, 1) == -1:
            ratio = 1.0 - ratio
        return joint.lower + ratio * joint.span

    def radians_to_normalized(self, name: str, value: float) -> float:
        joint = self.model.joints_by_name[name]
        ratio = float(np.clip((value - joint.lower) / joint.span, 0.0, 1.0))
        if self.config.joint_directions.get(name, 1) == -1:
            ratio = 1.0 - ratio
        return ratio * 100.0 if name == self.config.gripper_joint_name else ratio * 200.0 - 100.0

    def observation_to_radians(self, observation: Mapping[str, float]) -> dict[str, float]:
        names = self.config.arm_joint_names + [self.config.gripper_joint_name]
        return {
            name: self.normalized_to_radians(name, float(observation[f"{name}.pos"]))
            for name in names
        }

    def radians_to_action(self, positions: Mapping[str, float]) -> dict[str, float]:
        names = self.config.arm_joint_names + [self.config.gripper_joint_name]
        return {f"{name}.pos": self.radians_to_normalized(name, positions[name]) for name in names}


@dataclass
class InputCommand:
    motion: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    pitch: float = 0.0
    roll: float = 0.0
    gripper: float = 0.0
    go_to_basket: bool = False
    horizontal_speed_m_s: float | None = None
    quit: bool = False
    emergency_stop: bool = False

    @property
    def active(self) -> bool:
        return bool(
            np.linalg.norm(self.motion) > 1e-9
            or abs(self.pitch) > 1e-9
            or abs(self.roll) > 1e-9
            or abs(self.gripper) > 1e-9
            or self.go_to_basket
        )


def merge_commands(*commands: InputCommand) -> InputCommand:
    motion = sum((command.motion for command in commands), start=np.zeros(3, dtype=float))
    norm = np.linalg.norm(motion)
    if norm > 1.0:
        motion /= norm
    return InputCommand(
        motion=motion,
        pitch=float(np.clip(sum(command.pitch for command in commands), -1.0, 1.0)),
        roll=float(np.clip(sum(command.roll for command in commands), -1.0, 1.0)),
        gripper=float(np.clip(sum(command.gripper for command in commands), -1.0, 1.0)),
        go_to_basket=any(command.go_to_basket for command in commands),
        horizontal_speed_m_s=next(
            (
                command.horizontal_speed_m_s
                for command in reversed(commands)
                if command.horizontal_speed_m_s is not None
            ),
            None,
        ),
        quit=any(command.quit for command in commands),
        emergency_stop=any(command.emergency_stop for command in commands),
    )


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


def keyboard_command_from_keys(keys: set[str]) -> InputCommand:
    return InputCommand(
        motion=np.array(
            [
                float("left" in keys) - float("right" in keys),
                float("down" in keys) - float("up" in keys),
                float("shift_r" in keys or "u" in keys) - float("shift_l" in keys or "j" in keys),
            ],
            dtype=float,
        ),
        pitch=float("w" in keys) - float("s" in keys),
        gripper=float("ctrl_r" in keys or "o" in keys) - float("ctrl_l" in keys or "c" in keys),
        quit="quit" in keys or "q" in keys,
        emergency_stop="emergency_stop" in keys,
    )


def xbox_command_from_axes(
    left_x: float,
    left_y: float,
    right_y: float,
    left_trigger: float,
    right_trigger: float,
    quit: bool = False,
    left_bumper: bool = False,
    right_bumper: bool = False,
    go_to_basket: bool = False,
    right_x: float = 0.0,
    horizontal_speed_m_s: float | None = None,
) -> InputCommand:
    return InputCommand(
        motion=np.array([-left_x, -left_y, right_y], dtype=float),
        pitch=float(right_bumper) - float(left_bumper),
        roll=-right_x,
        gripper=float(np.clip(right_trigger - left_trigger, -1.0, 1.0)),
        go_to_basket=go_to_basket,
        horizontal_speed_m_s=horizontal_speed_m_s,
        quit=quit,
    )


def update_cartesian_target(
    target: np.ndarray,
    reference_tip: np.ndarray,
    motion: np.ndarray,
    previous_motion: np.ndarray,
    speed_m_s: float | np.ndarray,
    tap_step_m: float,
    dt: float,
) -> tuple[np.ndarray, bool]:
    """Update a velocity target without queuing unresolved key taps.

    A newly pressed direction (or a direction change) replaces the old target
    with a small step from the visible desired model. Holding the same direction keeps
    extending that target at the configured Cartesian speed.
    """
    magnitude = float(np.linalg.norm(motion))
    if magnitude <= 1e-9:
        return target, False
    previous_magnitude = float(np.linalg.norm(previous_motion))
    direction_changed = previous_magnitude <= 1e-9
    if not direction_changed:
        alignment = float(
            np.dot(motion / magnitude, previous_motion / previous_magnitude)
        )
        direction_changed = alignment < 0.95
    if direction_changed:
        step = motion / magnitude * tap_step_m * min(magnitude, 1.0)
        return reference_tip + step, True
    return target + motion * np.asarray(speed_m_s, dtype=float) * dt, False


class KeyboardInput:
    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self._pending_presses: set[str] = set()
        self._lock = threading.Lock()
        self._listener: Any = None

    def start(self) -> None:
        from pynput import keyboard

        special = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.shift: "shift_l",
            keyboard.Key.shift_l: "shift_l",
            keyboard.Key.shift_r: "shift_r",
            keyboard.Key.ctrl: "ctrl_l",
            keyboard.Key.ctrl_l: "ctrl_l",
            keyboard.Key.ctrl_r: "ctrl_r",
            keyboard.Key.space: "emergency_stop",
            keyboard.Key.esc: "quit",
        }

        def token(key: Any) -> str | None:
            if key in special:
                return special[key]
            char = getattr(key, "char", None)
            return char.lower() if isinstance(char, str) else None

        def pressed(key: Any) -> None:
            value = token(key)
            if value is not None:
                with self._lock:
                    self._pressed.add(value)
                    # Preserve quick taps that begin and end between control ticks.
                    self._pending_presses.add(value)

        def released(key: Any) -> None:
            value = token(key)
            if value is not None:
                with self._lock:
                    self._pressed.discard(value)

        self._listener = keyboard.Listener(on_press=pressed, on_release=released, suppress=False)
        self._listener.start()

    def snapshot(self) -> InputCommand:
        with self._lock:
            keys = self._pressed | self._pending_presses
            self._pending_presses.clear()
            return keyboard_command_from_keys(keys)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class XboxGamepadInput:
    def __init__(self, deadzone: float, speed_presets_m_s: Mapping[str, float]) -> None:
        self.deadzone = deadzone
        self.speed_presets_m_s = dict(speed_presets_m_s)
        self.pygame: Any = None
        self.controller_module: Any = None
        self.controller: Any = None
        self.last_scan = 0.0

    def start(self) -> None:
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="pkg_resources is deprecated as an API.*"
            )
            import pygame
            from pygame._sdl2 import controller

        self.pygame = pygame
        self.controller_module = controller
        pygame.init()
        controller.init()
        try:
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
        except pygame.error:
            pass
        self._scan(force=True)

    def _scan(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_scan < 1.0:
            return
        self.last_scan = now
        if self.controller is not None:
            try:
                if self.controller.attached():
                    return
            except Exception:
                pass
            print("Xbox controller disconnected; keyboard remains available.")
            self.controller = None
        for index in range(self.controller_module.get_count()):
            if self.controller_module.is_controller(index):
                self.controller = self.controller_module.Controller(index)
                name = self.controller.name
                if callable(name):
                    name = name()
                print(f"Xbox/SDL controller connected: {name}")
                return

    def snapshot(self) -> InputCommand:
        if self.pygame is None:
            return InputCommand()
        self.pygame.event.pump()
        self._scan()
        if self.controller is None:
            return InputCommand()
        p = self.pygame
        axis = self.controller.get_axis
        horizontal_speed_m_s = None
        dpad_buttons = (
            ("left", p.CONTROLLER_BUTTON_DPAD_LEFT),
            ("up", p.CONTROLLER_BUTTON_DPAD_UP),
            ("right", p.CONTROLLER_BUTTON_DPAD_RIGHT),
            ("down", p.CONTROLLER_BUTTON_DPAD_DOWN),
        )
        for name, button in dpad_buttons:
            if self.controller.get_button(button):
                horizontal_speed_m_s = self.speed_presets_m_s[name]
        return xbox_command_from_axes(
            left_x=apply_deadzone(axis(p.CONTROLLER_AXIS_LEFTX) / 32768.0, self.deadzone),
            left_y=apply_deadzone(axis(p.CONTROLLER_AXIS_LEFTY) / 32768.0, self.deadzone),
            right_y=apply_deadzone(axis(p.CONTROLLER_AXIS_RIGHTY) / 32768.0, self.deadzone),
            left_trigger=max(0.0, axis(p.CONTROLLER_AXIS_TRIGGERLEFT) / 32767.0),
            right_trigger=max(0.0, axis(p.CONTROLLER_AXIS_TRIGGERRIGHT) / 32767.0),
            quit=bool(self.controller.get_button(p.CONTROLLER_BUTTON_START)),
            left_bumper=bool(
                self.controller.get_button(p.CONTROLLER_BUTTON_LEFTSHOULDER)
            ),
            right_bumper=bool(
                self.controller.get_button(p.CONTROLLER_BUTTON_RIGHTSHOULDER)
            ),
            go_to_basket=bool(self.controller.get_button(p.CONTROLLER_BUTTON_Y)),
            right_x=apply_deadzone(
                axis(p.CONTROLLER_AXIS_RIGHTX) / 32768.0, self.deadzone
            ),
            horizontal_speed_m_s=horizontal_speed_m_s,
        )

    def stop(self) -> None:
        if self.controller is not None:
            self.controller.quit()
            self.controller = None
        if self.controller_module is not None:
            self.controller_module.quit()
        if self.pygame is not None:
            self.pygame.quit()


def differential_ik_step(
    model: URDFModel,
    positions: Mapping[str, float],
    desired_delta: np.ndarray,
    joint_names: list[str],
    end_effector: str,
    end_effector_offset: Iterable[float],
    damping: float,
    dt: float,
    max_joint_step_normalized: float,
    desired_pitch_delta: float = 0.0,
    pitch_joint_names: Iterable[str] = (),
    pitch_weight_m_per_rad: float = 0.08,
) -> tuple[dict[str, float], bool]:
    result = dict(positions)
    if np.linalg.norm(desired_delta) <= 1e-12 and abs(desired_pitch_delta) <= 1e-12:
        return result, True
    jacobian, start = model.position_jacobian(
        positions, joint_names, end_effector, end_effector_offset
    )
    pitch_names = set(pitch_joint_names)
    start_pitch = model.pitch_value(positions, pitch_names)
    desired_task = np.asarray(desired_delta, dtype=float)
    if pitch_names:
        pitch_row = np.asarray(
            [1.0 if name in pitch_names else 0.0 for name in joint_names], dtype=float
        )
        jacobian = np.vstack([jacobian, pitch_weight_m_per_rad * pitch_row])
        desired_task = np.append(
            desired_task, pitch_weight_m_per_rad * desired_pitch_delta
        )
    # Active-set DLS: when a joint is already at a limit and the unconstrained
    # solution asks it to move farther outward, remove that column and solve
    # again. Otherwise one saturated elbow can make every other joint's clipped
    # movement point in the wrong task-space direction and freeze the arm.
    active_indices = list(range(len(joint_names)))
    delta_joints = np.zeros(len(joint_names), dtype=float)
    while active_indices:
        active_jacobian = jacobian[:, active_indices]
        try:
            active_delta = active_jacobian.T @ np.linalg.solve(
                active_jacobian @ active_jacobian.T
                + damping**2 * np.eye(active_jacobian.shape[0]),
                desired_task,
            )
        except np.linalg.LinAlgError:
            return result, False
        delta_joints[:] = 0.0
        delta_joints[active_indices] = active_delta
        blocked_indices = []
        for index in active_indices:
            joint = model.joints_by_name[joint_names[index]]
            value = positions[joint_names[index]]
            at_lower = value <= joint.lower + 1e-6 and delta_joints[index] < 0
            at_upper = value >= joint.upper - 1e-6 and delta_joints[index] > 0
            if at_lower or at_upper:
                blocked_indices.append(index)
        if not blocked_indices:
            break
        active_indices = [index for index in active_indices if index not in blocked_indices]
    if not active_indices:
        return result, False

    caps = []
    for name in joint_names:
        joint = model.joints_by_name[name]
        range_cap = joint.span * max_joint_step_normalized / 200.0
        velocity_cap = joint.velocity * dt if math.isfinite(joint.velocity) else range_cap
        caps.append(min(range_cap, velocity_cap))
    ratio = max((abs(value) / cap for value, cap in zip(delta_joints, caps, strict=True)), default=1.0)
    if ratio > 1.0:
        delta_joints /= ratio

    for index, name in enumerate(joint_names):
        joint = model.joints_by_name[name]
        result[name] = float(np.clip(positions[name] + delta_joints[index], joint.lower, joint.upper))
    transforms, _ = model.forward_kinematics(result)
    local_point = np.append(np.asarray(end_effector_offset, dtype=float), 1.0)
    actual_delta = (transforms[end_effector] @ local_point)[:3] - start
    actual_pitch_delta = model.pitch_value(result, pitch_names) - start_pitch
    task_progress = float(np.dot(actual_delta, desired_delta))
    task_progress += (
        pitch_weight_m_per_rad**2 * actual_pitch_delta * desired_pitch_delta
    )
    progressed = task_progress > 1e-10
    return (result if progressed else dict(positions)), progressed


def apply_joint_step(
    model: URDFModel,
    positions: Mapping[str, float],
    name: str,
    direction: float,
    speed: float,
    dt: float,
) -> dict[str, float]:
    result = dict(positions)
    joint = model.joints_by_name[name]
    result[name] = float(np.clip(positions[name] + direction * speed * dt, joint.lower, joint.upper))
    return result


class RerunViewer:
    def __init__(self, model: URDFModel, config: TeleopConfig) -> None:
        import rerun as rr

        self.rr = rr
        self.model = model
        self.config = config
        self.started = time.monotonic()
        self.last_status: tuple[str, str] | None = None
        rr.init("apple_picker_so100_teleop", spawn=True)
        rr.log(model.entity_prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for visual in model.visuals:
            if not visual.mesh_path.is_file():
                raise FileNotFoundError(f"Mesh not found: {visual.mesh_path}")
            path = f"{model.link_paths[visual.link]}/visual_{visual.index}"
            rr.log(
                path,
                rr.Transform3D(translation=visual.origin_xyz, mat3x3=rpy_matrix(visual.origin_rpy)),
                static=True,
            )
            rr.log(path, rr.Asset3D(path=visual.mesh_path, albedo_factor=visual.rgba), static=True)

    def log(
        self,
        positions: Mapping[str, float],
        status: str,
        level: str,
        measured_positions: Mapping[str, float] | None = None,
    ) -> None:
        rr = self.rr
        rr.set_time("control_time", duration=time.monotonic() - self.started)
        display_positions = dict(positions)
        display_positions[self.config.roll_joint_name] += math.radians(
            self.config.rerun_roll_offset_degrees
        )
        for joint in self.model.joints:
            local = self.model.local_joint_transform(joint, display_positions)
            rr.log(
                self.model.link_paths[joint.child],
                rr.Transform3D(translation=local[:3, 3], mat3x3=local[:3, :3]),
            )
        transforms, _ = self.model.forward_kinematics(display_positions)
        local_tip = np.append(np.asarray(self.config.end_effector_offset_xyz), 1.0)
        tip = (transforms[self.config.end_effector_link] @ local_tip)[:3]
        rr.log(
            f"{self.model.entity_prefix}/claw_position",
            rr.Points3D([tip], radii=0.004, colors=[255, 80, 40]),
        )
        for name, value in positions.items():
            rr.log(f"telemetry/desired_joints/{name}", rr.Scalars([math.degrees(value)]))
            if measured_positions is not None:
                measured = measured_positions[name]
                rr.log(
                    f"telemetry/measured_joints/{name}",
                    rr.Scalars([math.degrees(measured)]),
                )
                rr.log(
                    f"telemetry/tracking_error_degrees/{name}",
                    rr.Scalars([math.degrees(value - measured)]),
                )
        if (status, level) != self.last_status:
            rr.log("telemetry/status", rr.TextLog(status, level=level))
            self.last_status = (status, level)


class RealSO100:
    def __init__(self, config: TeleopConfig, mapper: JointRangeMapper) -> None:
        self.config = config
        self.mapper = mapper
        self.robot: Any = None
        self.last_normalized_positions: dict[str, float] | None = None

    def connect(self) -> None:
        try:
            from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
        except ImportError:
            local_source = PROJECT_ROOT / "lerobot" / "src"
            if not local_source.is_dir():
                raise RuntimeError('Install LeRobot with: python -m pip install -e "./lerobot[feetech]"') from None
            sys.path.insert(0, str(local_source))
            from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

        self.robot = SO100Follower(
            SO100FollowerConfig(
                port=self.config.robot_port,
                id=self.config.robot_id,
                use_degrees=False,
                # The controller applies the same clamp from the fresh feedback
                # read at the start of each tick. Leaving this enabled here would
                # make LeRobot perform a second serial read before every write.
                max_relative_target=None,
                disable_torque_on_disconnect=True,
            )
        )
        self.robot.connect()
        self.apply_arm_gain()

    def apply_arm_gain(self) -> None:
        for motor in self.config.commanded_arm_joint_names:
            gain = self.config.robot_arm_p_coefficients[motor]
            self.robot.bus.write("P_Coefficient", motor, gain)
            # In STS3215 position mode, zero means no velocity cap. This is not
            # continuous-rotation velocity mode; joint limits remain active.
            self.robot.bus.write("Goal_Velocity", motor, 0)
        settings = ", ".join(
            f"{motor}={self.config.robot_arm_p_coefficients[motor]}"
            for motor in self.config.commanded_arm_joint_names
        )
        print(f"Arm position gains set: {settings}. Position velocity cap disabled.")

    def read(self) -> dict[str, float]:
        observation = self.robot.get_observation()
        self.last_normalized_positions = {
            key: float(value)
            for key, value in observation.items()
            if key.endswith(".pos")
        }
        return self.mapper.observation_to_radians(observation)

    def send(self, positions: Mapping[str, float]) -> tuple[dict[str, float], bool]:
        requested = self.mapper.radians_to_action(positions)
        # Never send a target to explicitly feedback-only joints.
        for name in self.config.uncommanded_joint_names:
            requested.pop(f"{name}.pos", None)
        if self.last_normalized_positions is None:
            raise RuntimeError("Cannot send an action before reading current motor positions")
        safe_request: dict[str, float] = {}
        was_clamped = False
        limit = self.config.robot_max_relative_target
        for key, target in requested.items():
            measured = self.last_normalized_positions[key]
            bounded = float(np.clip(target, measured - limit, measured + limit))
            safe_request[key] = bounded
            was_clamped |= abs(bounded - target) > 1e-6
        sent = self.robot.send_action(safe_request)
        accepted_positions = dict(positions)
        for key, value in sent.items():
            name = key.removesuffix(".pos")
            accepted_positions[name] = self.mapper.normalized_to_radians(name, float(value))
        was_clamped |= any(
            abs(float(sent[name]) - value) > 1e-6 for name, value in safe_request.items()
        )
        return accepted_positions, was_clamped

    def close(self) -> None:
        if self.robot is not None:
            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
            finally:
                self.robot = None


CONTROL_HELP = """
Arrow keys: forward/back/left/right
Right/Left Shift: up/down (fallback U/J)
W/S: pitch claw up/down
Right/Left Ctrl: open/close (fallback O/C)
Xbox left stick: forward/back/left/right
Xbox right stick vertical: up/down
Xbox right stick horizontal: wrist roll
Xbox RB/LB: claw pitch up/down
Xbox RT/LT: open/close
Xbox Y: move to basket preset
Xbox D-pad left/up/right/down: horizontal speed 0.04/0.10/0.16/0.20 m/s
Esc, Q, or Xbox Menu: quit
SPACE: EMERGENCY STOP, disable torque, and exit
""".strip()


POSITION_PRESETS: dict[str, tuple[float, float, float]] = {
    "default": (0.00, -0.30, 0.22),
    "middle-up": (0.00, -0.28, 0.36),
    "forward": (0.00, -0.42, 0.22),
    "left": (0.15, -0.30, 0.25),
    "basket": (0.20, -0.25, 0.20),
}
POSITION_PRESET_PITCH_DEGREES: dict[str, float] = {
    "basket": -30.0,
}


def run(
    mode: str,
    config: TeleopConfig,
    initial_target: np.ndarray | None = None,
    initial_pitch_degrees: float | None = None,
) -> int:
    model = URDFModel(config.urdf_path)
    model.validate(config.arm_joint_names + [config.gripper_joint_name], config.end_effector_link)
    mapper = JointRangeMapper(model, config)
    keyboard = KeyboardInput()
    gamepad = XboxGamepadInput(
        config.gamepad_deadzone,
        config.controller_horizontal_speed_presets_m_s,
    )
    real_robot: RealSO100 | None = None
    positions = model.midpoint_positions()
    planned_positions = dict(positions)
    emergency_stop = False

    if mode == "real":
        print(
            "\nREAL HARDWARE MODE\n"
            f"Connecting to {config.robot_id} on {config.robot_port}.\n"
            "Motor 5 wrist-roll is independently controlled by the right stick."
        )
        real_robot = RealSO100(config, mapper)

    try:
        if real_robot is not None:
            real_robot.connect()
            positions = real_robot.read()
            planned_positions = dict(positions)
        viewer = RerunViewer(model, config)
        keyboard.start()
        gamepad.start()
        print(CONTROL_HELP)
        print(f"Running in {mode.upper()} mode.")

        period = 1.0 / config.control_rate_hz
        viewer_period = 1.0 / config.viewer_rate_hz
        last_viewer_log = float("-inf")
        previous_tick = time.monotonic()
        cartesian_target = model.point_position(
            planned_positions, config.end_effector_link, config.end_effector_offset_xyz
        )
        goal_active = initial_target is not None
        if initial_target is not None:
            cartesian_target = np.asarray(initial_target, dtype=float)
            print(
                "Moving claw to robot-base target "
                f"X={cartesian_target[0]:.3f}, "
                f"Y={cartesian_target[1]:.3f}, Z={cartesian_target[2]:.3f} m"
            )
        pitch_control_active = initial_pitch_degrees is not None
        pitch_target = (
            math.radians(initial_pitch_degrees)
            if initial_pitch_degrees is not None
            else model.pitch_value(planned_positions, config.pitch_joint_names)
        )
        forward_speed_m_s = config.cartesian_speed_m_s
        lateral_speed_m_s = config.lateral_speed_m_s
        previous_motion = np.zeros(3, dtype=float)
        while True:
            tick = time.monotonic()
            dt = min(max(tick - previous_tick, period * 0.25), period * 2.0)
            previous_tick = tick
            command = merge_commands(keyboard.snapshot(), gamepad.snapshot())
            if command.horizontal_speed_m_s is not None:
                selected_speed = command.horizontal_speed_m_s
                if (
                    abs(forward_speed_m_s - selected_speed) > 1e-9
                    or abs(lateral_speed_m_s - selected_speed) > 1e-9
                ):
                    forward_speed_m_s = selected_speed
                    lateral_speed_m_s = selected_speed
                    print(f"Horizontal speed set to {selected_speed:.2f} m/s")
            if command.emergency_stop:
                emergency_stop = True
                print("EMERGENCY STOP: disabling torque and exiting...")
                break
            if command.quit:
                break
            if real_robot is not None:
                positions = real_robot.read()

            motion_is_active = np.linalg.norm(command.motion) > 1e-9
            control_positions = planned_positions if real_robot is not None else positions
            requested = dict(control_positions)
            ik_ok = True
            current_tip = model.point_position(
                control_positions, config.end_effector_link, config.end_effector_offset_xyz
            )
            current_pitch = model.pitch_value(
                control_positions, config.pitch_joint_names
            )
            if command.go_to_basket:
                cartesian_target = np.asarray(POSITION_PRESETS["basket"], dtype=float)
                pitch_target = math.radians(POSITION_PRESET_PITCH_DEGREES["basket"])
                pitch_control_active = True
                goal_active = True
            elif motion_is_active:
                goal_active = True
                cartesian_target, _ = update_cartesian_target(
                    cartesian_target,
                    current_tip,
                    command.motion,
                    previous_motion,
                    np.array(
                        [lateral_speed_m_s, forward_speed_m_s, config.vertical_speed_m_s],
                        dtype=float,
                    ),
                    config.tap_step_m,
                    dt,
                )

            if abs(command.pitch) > 1e-9:
                if not pitch_control_active:
                    pitch_target = current_pitch
                    pitch_control_active = True
                pitch_target += (
                    command.pitch
                    * math.radians(config.pitch_speed_degrees_s)
                    * dt
                )
                goal_active = True

            if goal_active:
                cartesian_error = cartesian_target - current_tip
                error_norm = np.linalg.norm(cartesian_error)
                pitch_error = (
                    pitch_target - current_pitch if pitch_control_active else 0.0
                )
                pitch_error = float(
                    np.clip(
                        pitch_error,
                        -math.radians(config.max_pitch_error_degrees),
                        math.radians(config.max_pitch_error_degrees),
                    )
                )
                position_needs_work = error_norm > config.position_tolerance_m
                pitch_needs_work = (
                    pitch_control_active
                    and abs(pitch_error) > math.radians(config.pitch_tolerance_degrees)
                )
                if position_needs_work or pitch_needs_work:
                    if error_norm > config.max_cartesian_error_m:
                        cartesian_error *= config.max_cartesian_error_m / error_norm
                    requested, ik_ok = differential_ik_step(
                        model,
                        requested,
                        cartesian_error,
                        config.ik_joint_names,
                        config.end_effector_link,
                        config.end_effector_offset_xyz,
                        config.ik_damping,
                        dt,
                        config.max_joint_step_normalized,
                        desired_pitch_delta=pitch_error,
                        pitch_joint_names=config.pitch_joint_names,
                        pitch_weight_m_per_rad=config.pitch_weight_m_per_rad,
                    )
                    # If the combined XYZ+pitch task is blocked at a joint
                    # limit, keep making useful progress instead of freezing.
                    # Position-only motion can move away from the singular pose;
                    # the persistent pitch target is retried on following ticks.
                    if not ik_ok and position_needs_work:
                        requested, ik_ok = differential_ik_step(
                            model,
                            control_positions,
                            cartesian_error,
                            config.ik_joint_names,
                            config.end_effector_link,
                            config.end_effector_offset_xyz,
                            config.ik_damping,
                            dt,
                            config.max_joint_step_normalized,
                        )
                    if not ik_ok and pitch_needs_work:
                        requested, ik_ok = differential_ik_step(
                            model,
                            control_positions,
                            np.zeros(3, dtype=float),
                            config.ik_joint_names,
                            config.end_effector_link,
                            config.end_effector_offset_xyz,
                            config.ik_damping,
                            dt,
                            config.max_joint_step_normalized,
                            desired_pitch_delta=pitch_error,
                            pitch_joint_names=config.pitch_joint_names,
                            pitch_weight_m_per_rad=config.pitch_weight_m_per_rad,
                        )
            if abs(command.roll) > 1e-9:
                requested = apply_joint_step(
                    model,
                    requested,
                    config.roll_joint_name,
                    command.roll,
                    math.radians(config.roll_speed_degrees_s),
                    dt,
                )
            if abs(command.gripper) > 1e-9:
                requested = apply_joint_step(
                    model,
                    requested,
                    config.gripper_joint_name,
                    command.gripper,
                    config.gripper_speed_rad_s,
                    dt,
                )

            clamped = False
            if real_robot is not None:
                # The simulated solution remains authoritative. Send it every
                # control tick so physical correction never stops while an input
                # is held. Send the exact simulated joint solution; recomputing
                # IK from measured feedback can select a different arm posture.
                planned_positions = dict(requested)
                _, clamped = real_robot.send(planned_positions)
            else:
                positions = requested

            planned_tip = model.point_position(
                planned_positions if real_robot is not None else positions,
                config.end_effector_link,
                config.end_effector_offset_xyz,
            )
            plan_reached = (
                np.linalg.norm(cartesian_target - planned_tip)
                <= config.position_tolerance_m
            )
            if pitch_control_active:
                planned_pitch = model.pitch_value(
                    planned_positions if real_robot is not None else positions,
                    config.pitch_joint_names,
                )
                plan_reached &= (
                    abs(pitch_target - planned_pitch)
                    <= math.radians(config.pitch_tolerance_degrees)
                )
            measured_reached = True
            if real_robot is not None:
                measured_tip = model.point_position(
                    positions, config.end_effector_link, config.end_effector_offset_xyz
                )
                measured_reached = (
                    np.linalg.norm(cartesian_target - measured_tip)
                    <= config.position_tolerance_m
                )
                if pitch_control_active:
                    measured_pitch = model.pitch_value(
                        positions, config.pitch_joint_names
                    )
                    measured_reached &= (
                        abs(pitch_target - measured_pitch)
                        <= math.radians(config.pitch_tolerance_degrees)
                    )
            if goal_active and not motion_is_active and plan_reached and measured_reached:
                goal_active = False
            previous_motion = command.motion.copy()

            if not ik_ok:
                status, level = "Direction blocked by pose or joint limit", "WARN"
            elif clamped:
                status, level = "LeRobot safety clamp limited the command", "WARN"
            elif command.active or goal_active:
                status, level = "Moving", "INFO"
            else:
                status, level = "Ready", "INFO"
            # In real mode Rerun shows the full desired simulation, not measured
            # feedback and not the intermediate safety-clamped motor target.
            viewer_positions = planned_positions if real_robot is not None else positions
            if tick - last_viewer_log >= viewer_period:
                viewer.log(
                    viewer_positions,
                    status,
                    level,
                    measured_positions=positions if real_robot is not None else None,
                )
                last_viewer_log = tick
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        # Hardware is released first so the emergency path is not delayed by
        # viewer/input cleanup.
        if real_robot is not None:
            if not emergency_stop:
                print("Disconnecting and disabling torque...")
            real_robot.close()
        keyboard.stop()
        gamepad.stop()
    print("SO100 teleoperation stopped.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("virtual", "real"), default="virtual")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Move the claw to absolute robot-base coordinates in meters",
    )
    target_group.add_argument(
        "--preset",
        choices=tuple(POSITION_PRESETS),
        help="Move to a named starting position",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print named coordinate templates and exit",
    )
    args = parser.parse_args()
    if args.list_presets:
        print("Coordinates are robot-base meters: +X left, -Y forward, +Z up.")
        for name, (x, y, z) in POSITION_PRESETS.items():
            pitch = POSITION_PRESET_PITCH_DEGREES.get(name)
            pitch_text = "" if pitch is None else f"  Pitch={pitch: .1f} deg"
            print(f"{name:10s}  X={x: .3f}  Y={y: .3f}  Z={z: .3f}{pitch_text}")
        return 0
    initial_target = None
    initial_pitch_degrees = None
    if args.target is not None:
        initial_target = np.asarray(args.target, dtype=float)
        if not np.all(np.isfinite(initial_target)):
            parser.error("--target coordinates must be finite numbers")
    elif args.preset is not None:
        initial_target = np.asarray(POSITION_PRESETS[args.preset], dtype=float)
        initial_pitch_degrees = POSITION_PRESET_PITCH_DEGREES.get(args.preset)
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    try:
        return run(
            args.mode,
            TeleopConfig.load(config_path.resolve()),
            initial_target=initial_target,
            initial_pitch_degrees=initial_pitch_degrees,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
