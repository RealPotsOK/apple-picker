# SO100 Cartesian Claw Control

Goal:

```text
move_to_pose(x, y, z, roll, pitch, yaw)
```

The code should convert a desired claw pose into SO100 joint angles, then command the arm through LeRobot.

## Pieces Needed

1. A URDF model of the SO100.
2. Your custom claw added to that URDF.
3. The claw tip marked as the end-effector link.
4. An inverse kinematics solver.
5. A LeRobot arm driver that sends solved joint positions to the real motors.

## Recommended Control Flow

```text
target claw pose
  -> check workspace limits
  -> solve inverse kinematics
  -> reject if unreachable or outside joint limits
  -> slowly interpolate joints
  -> send joint targets to SO100
  -> stop and hold if anything fails
```

## Best Failure Behavior

Do not just keep trying forever. If the target is unreachable:

```text
stop arm
hold current position
report why it failed
choose a safer nearby approach pose
```

For apple picking, a good fallback is to move to a point farther away from the apple and let the claw camera re-center before trying again.

## Why The Custom Claw Matters

IK does not care where the stock gripper used to be. It cares where your final useful point is.

If your custom claw is longer than the stock gripper, the end-effector link must be at the new claw tip. Otherwise the arm will move the wrist to the right place, but the claw tip will be offset.
