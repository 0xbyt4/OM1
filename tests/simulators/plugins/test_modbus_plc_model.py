import random

import pytest

from simulators.plugins.modbus_plc_model import (
    COIL_ALARM,
    COIL_MOTOR_ON,
    HOLDING_REGISTER_COUNT,
    REG_FAN_SPEED,
    REG_MOTOR_RPM,
    REG_PRESSURE,
    REG_TEMPERATURE,
    PLCPhysicsConfig,
    PLCPhysicsModel,
)


@pytest.fixture
def default_model():
    """Create a PLC physics model with default configuration."""
    return PLCPhysicsModel()


@pytest.fixture
def custom_model():
    """Create a PLC physics model with custom configuration for faster testing."""
    config = PLCPhysicsConfig(
        initial_temp=25.0,
        ambient_temp=22.0,
        initial_pressure=3.2,
        motor_heat_rate=1.0,
        max_fan_cooling=2.0,
        ambient_drift_factor=0.05,
        motor_ramp_up=100,
        motor_ramp_down=200,
        motor_target_rpm=1500,
        pressure_volatility=0.0,  # Disable randomness for deterministic tests
        alarm_high_temp=80.0,
        alarm_low_temp=75.0,
    )
    return PLCPhysicsModel(config=config)


class TestInitialization:
    """Tests for PLCPhysicsModel initialization."""

    def test_default_initialization(self, default_model):
        """Test model initializes with default values."""
        assert default_model.temperature == 22.0
        assert default_model.pressure == 3.0
        assert default_model.fan_speed == 0
        assert default_model.motor_rpm == 0
        assert default_model.motor_on is False
        assert default_model.alarm is False
        assert default_model.tick_count == 0

    def test_custom_initialization(self, custom_model):
        """Test model initializes with custom config values."""
        assert custom_model.temperature == 25.0
        assert custom_model.pressure == 3.2
        assert custom_model.fan_speed == 0
        assert custom_model.motor_rpm == 0

    def test_config_defaults(self):
        """Test PLCPhysicsConfig default values."""
        config = PLCPhysicsConfig()
        assert config.initial_temp == 22.0
        assert config.ambient_temp == 22.0
        assert config.motor_heat_rate == 0.5
        assert config.alarm_high_temp == 80.0
        assert config.alarm_low_temp == 75.0
        assert config.fan_speed_max == 3000


class TestMotorRPM:
    """Tests for motor RPM ramping behavior."""

    def test_motor_ramps_up_when_on(self, custom_model):
        """Test motor RPM increases when motor is ON."""
        custom_model.set_coil(COIL_MOTOR_ON, True)

        custom_model.tick()
        assert custom_model.motor_rpm == 100  # motor_ramp_up = 100

        custom_model.tick()
        assert custom_model.motor_rpm == 200

    def test_motor_caps_at_target(self, custom_model):
        """Test motor RPM does not exceed target."""
        custom_model.set_coil(COIL_MOTOR_ON, True)

        for _ in range(20):
            custom_model.tick()

        assert custom_model.motor_rpm == 1500  # motor_target_rpm

    def test_motor_ramps_down_when_off(self, custom_model):
        """Test motor RPM decreases when motor is OFF."""
        custom_model.set_coil(COIL_MOTOR_ON, True)
        for _ in range(5):
            custom_model.tick()
        assert custom_model.motor_rpm == 500

        custom_model.set_coil(COIL_MOTOR_ON, False)
        custom_model.tick()
        assert custom_model.motor_rpm == 300  # motor_ramp_down = 200

    def test_motor_rpm_does_not_go_negative(self, custom_model):
        """Test motor RPM stays at 0 when already stopped."""
        assert custom_model.motor_rpm == 0

        custom_model.tick()
        assert custom_model.motor_rpm == 0

    def test_motor_ramps_down_to_zero(self, custom_model):
        """Test motor RPM reaches zero after turning off."""
        custom_model.set_coil(COIL_MOTOR_ON, True)
        for _ in range(3):
            custom_model.tick()
        assert custom_model.motor_rpm == 300

        custom_model.set_coil(COIL_MOTOR_ON, False)
        for _ in range(5):
            custom_model.tick()
        assert custom_model.motor_rpm == 0


class TestTemperature:
    """Tests for temperature dynamics."""

    def test_motor_heats_up_temperature(self, custom_model):
        """Test temperature rises when motor is ON over multiple ticks."""
        initial_temp = custom_model.temperature
        custom_model.set_coil(COIL_MOTOR_ON, True)

        # First few ticks: RPM is low, ambient drift dominates.
        # After RPM ramps up, heating overcomes drift.
        for _ in range(15):
            custom_model.tick()

        assert custom_model.temperature > initial_temp

    def test_fan_cools_temperature(self, custom_model):
        """Test fan reduces temperature."""
        # Set high temp so cooling is visible against ambient drift
        custom_model._temperature = 50.0
        custom_model.set_holding_register(REG_FAN_SPEED, 3000)

        custom_model.tick()
        # Cooling = 2.0 * (3000/3000) = 2.0
        # Ambient drift toward 22: (22 - 50) * 0.05 = -1.4 (away from 50)
        # Net should be below 50
        assert custom_model.temperature < 50.0

    def test_ambient_drift_toward_ambient(self, custom_model):
        """Test temperature drifts toward ambient when no heat/cooling."""
        custom_model._temperature = 40.0

        custom_model.tick()
        # Drift = (22 - 40) * 0.05 = -0.9
        # So temp should be ~39.1
        assert custom_model.temperature < 40.0
        assert custom_model.temperature > 22.0

    def test_temperature_clamped_at_max(self, custom_model):
        """Test temperature does not exceed max."""
        custom_model._temperature = 119.5
        custom_model.set_coil(COIL_MOTOR_ON, True)
        custom_model._motor_rpm = 1500  # Full RPM

        for _ in range(10):
            custom_model.tick()

        assert custom_model.temperature <= custom_model.config.temp_max

    def test_temperature_clamped_at_min(self, custom_model):
        """Test temperature does not go below min."""
        custom_model._temperature = -9.0
        custom_model.set_holding_register(REG_FAN_SPEED, 3000)

        for _ in range(10):
            custom_model.tick()

        assert custom_model.temperature >= custom_model.config.temp_min

    def test_heating_proportional_to_rpm(self, custom_model):
        """Test heating is proportional to motor RPM fraction."""
        custom_model._temperature = 22.0
        custom_model.set_coil(COIL_MOTOR_ON, True)
        custom_model._motor_rpm = 750  # Half of target

        initial = custom_model.temperature
        custom_model.tick()

        # Heat = 1.0 * (750/1500) = 0.5 at half RPM
        # Full RPM heat = 1.0 * (1500/1500) = 1.0
        heat_at_half = custom_model.temperature - initial

        custom_model._temperature = 22.0
        custom_model._motor_rpm = 1500  # Full RPM
        custom_model.tick()
        heat_at_full = custom_model.temperature - initial

        assert heat_at_full > heat_at_half


class TestPressure:
    """Tests for pressure random walk."""

    def test_pressure_stays_in_bounds(self, default_model):
        """Test pressure stays within configured bounds over many ticks."""
        random.seed(42)
        for _ in range(1000):
            default_model.tick()

        assert default_model.pressure >= default_model.config.pressure_min
        assert default_model.pressure <= default_model.config.pressure_max

    def test_pressure_deterministic_zero_volatility(self, custom_model):
        """Test pressure stays constant with zero volatility."""
        initial = custom_model.pressure
        custom_model.tick()
        assert custom_model.pressure == initial

    def test_pressure_changes_with_volatility(self):
        """Test pressure actually changes when volatility is non-zero."""
        random.seed(42)
        config = PLCPhysicsConfig(pressure_volatility=0.1)
        model = PLCPhysicsModel(config=config)

        initial = model.pressure
        model.tick()
        # With non-zero volatility, pressure should change (very unlikely to stay same)
        # Use multiple ticks to ensure change
        changed = False
        for _ in range(10):
            model.tick()
            if model.pressure != initial:
                changed = True
                break
        assert changed


class TestAlarm:
    """Tests for alarm hysteresis logic.

    Uses a zero-drift model to test alarm thresholds precisely,
    since ambient drift would shift temperature during tick().
    """

    @pytest.fixture
    def no_drift_model(self):
        """Model with no ambient drift for precise threshold testing."""
        config = PLCPhysicsConfig(
            initial_temp=22.0,
            ambient_drift_factor=0.0,
            pressure_volatility=0.0,
        )
        return PLCPhysicsModel(config=config)

    def test_alarm_triggers_above_threshold(self, no_drift_model):
        """Test alarm activates when temperature exceeds high threshold."""
        no_drift_model._temperature = 81.0
        no_drift_model.tick()
        assert no_drift_model.alarm is True

    def test_alarm_clears_below_low_threshold(self, no_drift_model):
        """Test alarm deactivates when temperature drops below low threshold."""
        no_drift_model._alarm = True
        no_drift_model._temperature = 74.0
        no_drift_model.tick()
        assert no_drift_model.alarm is False

    def test_alarm_stays_on_in_hysteresis_band(self, no_drift_model):
        """Test alarm stays ON when temperature is between low and high thresholds."""
        no_drift_model._alarm = True
        no_drift_model._temperature = 77.0  # Between 75 and 80
        no_drift_model.tick()
        assert no_drift_model.alarm is True

    def test_alarm_stays_off_in_hysteresis_band(self, no_drift_model):
        """Test alarm stays OFF when temperature is between low and high thresholds."""
        no_drift_model._alarm = False
        no_drift_model._temperature = 77.0  # Between 75 and 80
        no_drift_model.tick()
        assert no_drift_model.alarm is False

    def test_alarm_at_exact_threshold(self, no_drift_model):
        """Test alarm behavior at exact threshold values."""
        no_drift_model._temperature = 80.0  # Exactly at high threshold
        no_drift_model.tick()
        assert no_drift_model.alarm is False  # Must be > 80, not >=

        no_drift_model._temperature = 80.1
        no_drift_model.tick()
        assert no_drift_model.alarm is True


class TestRegisterInterface:
    """Tests for Modbus register read/write interface."""

    def test_get_holding_registers(self, custom_model):
        """Test getting all holding registers as raw values."""
        regs = custom_model.get_holding_registers()

        assert len(regs) == HOLDING_REGISTER_COUNT
        assert regs[REG_TEMPERATURE] == 250  # 25.0 * 10
        assert regs[REG_PRESSURE] == 320  # 3.2 * 100
        assert regs[REG_FAN_SPEED] == 0
        assert regs[REG_MOTOR_RPM] == 0

    def test_get_coils(self, custom_model):
        """Test getting all coil values."""
        coils = custom_model.get_coils()

        assert len(coils) == 2
        assert coils[COIL_MOTOR_ON] is False
        assert coils[COIL_ALARM] is False

    def test_set_fan_speed(self, custom_model):
        """Test setting fan speed via register write."""
        result = custom_model.set_holding_register(REG_FAN_SPEED, 1500)
        assert result is True
        assert custom_model.fan_speed == 1500

    def test_set_fan_speed_clamped(self, custom_model):
        """Test fan speed is clamped to max value."""
        custom_model.set_holding_register(REG_FAN_SPEED, 5000)
        assert custom_model.fan_speed == 3000  # fan_speed_max

    def test_set_fan_speed_negative_clamped(self, custom_model):
        """Test negative fan speed is clamped to zero."""
        custom_model.set_holding_register(REG_FAN_SPEED, -100)
        assert custom_model.fan_speed == 0

    def test_read_only_registers_rejected(self, custom_model):
        """Test that writing to read-only registers is rejected."""
        assert custom_model.set_holding_register(REG_TEMPERATURE, 999) is False
        assert custom_model.set_holding_register(REG_PRESSURE, 999) is False
        assert custom_model.set_holding_register(REG_MOTOR_RPM, 999) is False

    def test_unknown_register_rejected(self, custom_model):
        """Test that writing to unknown register address is rejected."""
        assert custom_model.set_holding_register(99, 100) is False

    def test_set_motor_coil(self, custom_model):
        """Test setting motor ON/OFF via coil write."""
        result = custom_model.set_coil(COIL_MOTOR_ON, True)
        assert result is True
        assert custom_model.motor_on is True

        result = custom_model.set_coil(COIL_MOTOR_ON, False)
        assert result is True
        assert custom_model.motor_on is False

    def test_alarm_coil_read_only(self, custom_model):
        """Test that alarm coil cannot be written directly."""
        assert custom_model.set_coil(COIL_ALARM, True) is False

    def test_unknown_coil_rejected(self, custom_model):
        """Test that writing to unknown coil address is rejected."""
        assert custom_model.set_coil(99, True) is False


class TestRawConversion:
    """Tests for raw value conversion helpers."""

    def test_temp_to_raw(self):
        """Test temperature to raw conversion."""
        assert PLCPhysicsModel._temp_to_raw(25.0) == 250
        assert PLCPhysicsModel._temp_to_raw(0.0) == 0
        assert PLCPhysicsModel._temp_to_raw(100.5) == 1005

    def test_temp_to_raw_clamped(self):
        """Test temperature raw value is clamped to uint16 range."""
        assert PLCPhysicsModel._temp_to_raw(-50.0) == 0
        assert PLCPhysicsModel._temp_to_raw(70000.0) == 65535

    def test_pressure_to_raw(self):
        """Test pressure to raw conversion."""
        assert PLCPhysicsModel._pressure_to_raw(3.21) == 321
        assert PLCPhysicsModel._pressure_to_raw(0.0) == 0
        assert PLCPhysicsModel._pressure_to_raw(10.0) == 1000

    def test_pressure_to_raw_clamped(self):
        """Test pressure raw value is clamped to uint16 range."""
        assert PLCPhysicsModel._pressure_to_raw(-1.0) == 0
        assert PLCPhysicsModel._pressure_to_raw(700.0) == 65535

    def test_raw_to_temp(self):
        """Test raw to temperature conversion."""
        assert PLCPhysicsModel.raw_to_temp(250) == 25.0
        assert PLCPhysicsModel.raw_to_temp(0) == 0.0

    def test_raw_to_pressure(self):
        """Test raw to pressure conversion."""
        assert PLCPhysicsModel.raw_to_pressure(321) == 3.21
        assert PLCPhysicsModel.raw_to_pressure(0) == 0.0

    def test_roundtrip_conversion(self):
        """Test that temperature/pressure roundtrip through raw format."""
        temp = 25.3
        raw_temp = PLCPhysicsModel._temp_to_raw(temp)
        assert PLCPhysicsModel.raw_to_temp(raw_temp) == pytest.approx(temp, abs=0.1)

        pressure = 3.21
        raw_pressure = PLCPhysicsModel._pressure_to_raw(pressure)
        assert PLCPhysicsModel.raw_to_pressure(raw_pressure) == pytest.approx(
            pressure, abs=0.01
        )


class TestStateSummary:
    """Tests for state summary output."""

    def test_state_summary_format(self, custom_model):
        """Test state summary contains all variables."""
        summary = custom_model.get_state_summary()

        assert "T=25.0C" in summary
        assert "P=3.20bar" in summary
        assert "Fan=0rpm" in summary
        assert "Motor=OFF(0rpm)" in summary
        assert "OK" in summary

    def test_state_summary_motor_on(self, custom_model):
        """Test state summary reflects motor ON state."""
        custom_model.set_coil(COIL_MOTOR_ON, True)
        custom_model._motor_rpm = 1500

        summary = custom_model.get_state_summary()
        assert "Motor=ON(1500rpm)" in summary

    def test_state_summary_alarm(self, custom_model):
        """Test state summary reflects alarm state."""
        custom_model._alarm = True
        summary = custom_model.get_state_summary()
        assert "ALARM" in summary


class TestReset:
    """Tests for model reset."""

    def test_reset_restores_initial_state(self, custom_model):
        """Test reset returns model to initial configuration state."""
        custom_model.set_coil(COIL_MOTOR_ON, True)
        custom_model.set_holding_register(REG_FAN_SPEED, 2000)
        for _ in range(10):
            custom_model.tick()

        custom_model.reset()

        assert custom_model.temperature == 25.0  # initial_temp from config
        assert custom_model.pressure == 3.2  # initial_pressure from config
        assert custom_model.fan_speed == 0
        assert custom_model.motor_rpm == 0
        assert custom_model.motor_on is False
        assert custom_model.alarm is False
        assert custom_model.tick_count == 0


class TestTickCount:
    """Tests for tick counter."""

    def test_tick_increments_count(self, custom_model):
        """Test tick count increments each tick."""
        assert custom_model.tick_count == 0

        custom_model.tick()
        assert custom_model.tick_count == 1

        custom_model.tick()
        assert custom_model.tick_count == 2

    def test_tick_count_after_reset(self, custom_model):
        """Test tick count resets to zero after reset."""
        for _ in range(5):
            custom_model.tick()
        assert custom_model.tick_count == 5

        custom_model.reset()
        assert custom_model.tick_count == 0


class TestFeedbackLoop:
    """Tests for the full feedback loop (agent write -> physics -> register change)."""

    def test_motor_on_raises_temperature(self, custom_model):
        """Test turning motor ON causes temperature to increase over time."""
        initial_temp = custom_model.temperature
        custom_model.set_coil(COIL_MOTOR_ON, True)

        for _ in range(50):
            custom_model.tick()

        assert custom_model.temperature > initial_temp + 5.0

    def test_fan_counteracts_heating(self, custom_model):
        """Test fan cooling counteracts motor heating."""
        # Run motor for a while
        custom_model.set_coil(COIL_MOTOR_ON, True)
        for _ in range(20):
            custom_model.tick()
        temp_without_fan = custom_model.temperature

        # Reset and run with fan
        custom_model.reset()
        custom_model.set_coil(COIL_MOTOR_ON, True)
        custom_model.set_holding_register(REG_FAN_SPEED, 3000)
        for _ in range(20):
            custom_model.tick()
        temp_with_fan = custom_model.temperature

        assert temp_with_fan < temp_without_fan

    def test_alarm_triggers_from_motor_heat(self):
        """Test alarm triggers naturally from sustained motor operation."""
        config = PLCPhysicsConfig(
            initial_temp=70.0,
            motor_heat_rate=2.0,
            ambient_drift_factor=0.0,  # No ambient drift
            pressure_volatility=0.0,
            motor_ramp_up=1500,  # Instant ramp
        )
        model = PLCPhysicsModel(config=config)
        model.set_coil(COIL_MOTOR_ON, True)

        # Run until alarm triggers or max ticks
        alarm_triggered = False
        for _ in range(100):
            model.tick()
            if model.alarm:
                alarm_triggered = True
                break

        assert alarm_triggered
        assert model.temperature > config.alarm_high_temp

    def test_register_values_reflect_physics(self, custom_model):
        """Test that register values update after physics tick."""
        custom_model.set_coil(COIL_MOTOR_ON, True)
        custom_model.set_holding_register(REG_FAN_SPEED, 500)

        for _ in range(10):
            custom_model.tick()

        regs = custom_model.get_holding_registers()
        coils = custom_model.get_coils()

        # Motor RPM should have ramped up
        assert regs[REG_MOTOR_RPM] > 0

        # Fan speed should be what we set
        assert regs[REG_FAN_SPEED] == 500

        # Temperature should have changed from initial
        assert regs[REG_TEMPERATURE] != 250  # Was 25.0 * 10

        # Motor coil should be ON
        assert coils[COIL_MOTOR_ON] is True
