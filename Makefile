PYTHON ?= .venv/bin/python
PYTHONPATH := src

CAMERA ?= 0
MODEL ?= yolo11s.mlpackage
CONFIDENCE ?= 0.35
YOLO_ARGS ?=
ROBOT_ARGS ?=

.PHONY: help install install-lerobot yolo arm-virtual arm-real arm-presets check-venv

help:
	@echo "Apple Picker commands"
	@echo "  make install         Install project Python dependencies"
	@echo "  make install-lerobot Install the local LeRobot clone with Feetech support"
	@echo "  make yolo            Run Core ML apple detection and tracking"
	@echo "  make arm-virtual     Control the virtual SO100 in Rerun"
	@echo "  make arm-real        Control the calibrated physical SO100"
	@echo "  make arm-presets     List named claw-position templates"
	@echo
	@echo "Examples"
	@echo "  make yolo CAMERA=1 CONFIDENCE=0.20"
	@echo "  make yolo YOLO_ARGS='--width 1920 --height 1080'"
	@echo "  make arm-real ROBOT_ARGS='--config config/so100_rerun_teleop.yaml'"
	@echo "  make arm-virtual ROBOT_ARGS='--preset middle-up'"
	@echo "  make arm-real ROBOT_ARGS='--target 0.00 -0.30 0.22'"

check-venv:
	@test -x "$(PYTHON)" || (echo "Missing $(PYTHON). Create the virtual environment first." && exit 1)

install: check-venv
	$(PYTHON) -m pip install -r requirements.txt

install-lerobot: check-venv
	$(PYTHON) -m pip install -e "./lerobot[feetech]"

yolo: check-venv
	PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m apple_picker.vision.webcam_yolo \
		--camera "$(CAMERA)" \
		--model "$(MODEL)" \
		--confidence "$(CONFIDENCE)" \
		$(YOLO_ARGS)

arm-virtual: check-venv
	PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m apple_picker.robot.rerun_teleop \
		--mode virtual $(ROBOT_ARGS)

arm-real: check-venv
	PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m apple_picker.robot.rerun_teleop \
		--mode real $(ROBOT_ARGS)

arm-presets: check-venv
	PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m apple_picker.robot.rerun_teleop \
		--list-presets
