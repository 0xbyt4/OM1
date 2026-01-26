"""
PLC Physics Model for Modbus Simulator.

Models a simple industrial process with thermal dynamics, motor control,
and pressure monitoring. Designed to be used with a Modbus TCP server
to create a realistic PLC simulation with feedback loop.

Register Layout
---------------
Holding Registers:
    0 : Temperature (scale 0.1, unit C) - e.g. 250 = 25.0 C
    1 : Pressure (scale 0.01, unit bar) - e.g. 321 = 3.21 bar
    2 : Fan Speed (raw RPM, 0-3000) - writable by agent
    3 : Motor RPM (raw RPM, 0-1500) - read-only output

Coils:
    0 : Motor ON/OFF - writable by agent
    1 : Alarm - auto-set at >80 C, auto-clear at <75 C (hysteresis)
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PLCPhysicsConfig:
    """
    Configuration for the PLC physics simulation.

    Parameters
    ----------
    initial_temp : float
        Starting temperature in Celsius.
    ambient_temp : float
        Ambient (room) temperature for thermal drift.
    initial_pressure : float
        Starting pressure in bar.
    motor_heat_rate : float
        Temperature increase per tick when motor is ON (Celsius).
    max_fan_cooling : float
        Maximum cooling rate per tick at full fan speed (Celsius).
    ambient_drift_factor : float
        Rate of temperature drift toward ambient (0-1, fraction per tick).
    motor_ramp_up : int
        RPM increase per tick when motor is ON.
    motor_ramp_down : int
        RPM decrease per tick when motor is OFF.
    motor_target_rpm : int
        Target RPM when motor is fully ramped up.
    pressure_min : float
        Minimum pressure for random walk (bar).
    pressure_max : float
        Maximum pressure for random walk (bar).
    pressure_volatility : float
        Maximum pressure change per tick (bar).
    alarm_high_temp : float
        Temperature threshold to trigger alarm (Celsius).
    alarm_low_temp : float
        Temperature threshold to clear alarm (Celsius).
    temp_min : float
        Minimum allowed temperature (Celsius).
    temp_max : float
        Maximum allowed temperature (Celsius).
    fan_speed_max : int
        Maximum fan speed (RPM).
    """

    initial_temp: float = 22.0
    ambient_temp: float = 22.0
    initial_pressure: float = 3.0
    motor_heat_rate: float = 0.5
    max_fan_cooling: float = 0.8
    ambient_drift_factor: float = 0.02
    motor_ramp_up: int = 50
    motor_ramp_down: int = 100
    motor_target_rpm: int = 1500
    pressure_min: float = 2.8
    pressure_max: float = 3.5
    pressure_volatility: float = 0.05
    alarm_high_temp: float = 80.0
    alarm_low_temp: float = 75.0
    temp_min: float = -10.0
    temp_max: float = 120.0
    fan_speed_max: int = 3000


# Register address constants
REG_TEMPERATURE: int = 0
REG_PRESSURE: int = 1
REG_FAN_SPEED: int = 2
REG_MOTOR_RPM: int = 3

# Coil address constants
COIL_MOTOR_ON: int = 0
COIL_ALARM: int = 1

# Total counts for datastore allocation
HOLDING_REGISTER_COUNT: int = 4
COIL_COUNT: int = 2


@dataclass
class PLCPhysicsModel:
    """
    Physics model for a simple industrial process.

    Simulates temperature dynamics, motor control, pressure monitoring,
    and alarm logic. State is stored as physical values internally and
    converted to raw Modbus register values on demand.

    Parameters
    ----------
    config : PLCPhysicsConfig
        Physics configuration parameters.
    """

    config: PLCPhysicsConfig = field(default_factory=PLCPhysicsConfig)

    # Internal physical state
    _temperature: float = field(init=False)
    _pressure: float = field(init=False)
    _fan_speed: int = field(init=False, default=0)
    _motor_rpm: int = field(init=False, default=0)
    _motor_on: bool = field(init=False, default=False)
    _alarm: bool = field(init=False, default=False)
    _tick_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize physical state from configuration."""
        self._temperature = self.config.initial_temp
        self._pressure = self.config.initial_pressure

    def tick(self) -> None:
        """
        Advance the physics simulation by one time step.

        Updates motor RPM ramping, temperature dynamics (heating from motor,
        cooling from fan, ambient drift), pressure random walk, and alarm
        hysteresis logic.
        """
        self._tick_count += 1

        self._update_motor_rpm()
        self._update_temperature()
        self._update_pressure()
        self._update_alarm()

    def _update_motor_rpm(self) -> None:
        """
        Update motor RPM based on ON/OFF state.

        Motor ramps up gradually when ON and ramps down faster when OFF.
        """
        if self._motor_on:
            self._motor_rpm = min(
                self.config.motor_target_rpm,
                self._motor_rpm + self.config.motor_ramp_up,
            )
        else:
            self._motor_rpm = max(
                0,
                self._motor_rpm - self.config.motor_ramp_down,
            )

    def _update_temperature(self) -> None:
        """
        Update temperature based on heat sources and cooling.

        Heat sources: motor operation.
        Cooling: fan (proportional to speed), ambient drift.
        """
        # Heat from motor (proportional to actual RPM fraction)
        if self._motor_on and self._motor_rpm > 0:
            rpm_fraction = self._motor_rpm / self.config.motor_target_rpm
            self._temperature += self.config.motor_heat_rate * rpm_fraction

        # Cooling from fan (proportional to fan speed)
        if self._fan_speed > 0:
            fan_fraction = self._fan_speed / self.config.fan_speed_max
            cooling = self.config.max_fan_cooling * fan_fraction
            self._temperature -= cooling

        # Ambient drift (Newton's law of cooling approximation)
        temp_diff = self.config.ambient_temp - self._temperature
        self._temperature += temp_diff * self.config.ambient_drift_factor

        # Clamp temperature
        self._temperature = max(
            self.config.temp_min,
            min(self.config.temp_max, self._temperature),
        )

    def _update_pressure(self) -> None:
        """
        Update pressure with bounded random walk.

        Pressure fluctuates randomly within configured bounds, independent
        of other process variables.
        """
        delta = random.uniform(
            -self.config.pressure_volatility,
            self.config.pressure_volatility,
        )
        self._pressure += delta
        self._pressure = max(
            self.config.pressure_min,
            min(self.config.pressure_max, self._pressure),
        )

    def _update_alarm(self) -> None:
        """
        Update alarm state with hysteresis.

        Alarm activates when temperature exceeds high threshold and
        deactivates when temperature drops below low threshold.
        """
        if self._temperature > self.config.alarm_high_temp:
            if not self._alarm:
                logging.info(
                    f"ALARM ON: Temperature {self._temperature:.1f}C "
                    f"> {self.config.alarm_high_temp}C"
                )
            self._alarm = True
        elif self._temperature < self.config.alarm_low_temp:
            if self._alarm:
                logging.info(
                    f"ALARM OFF: Temperature {self._temperature:.1f}C "
                    f"< {self.config.alarm_low_temp}C"
                )
            self._alarm = False

    # --- State access (physical values) ---

    @property
    def temperature(self) -> float:
        """Current temperature in Celsius."""
        return self._temperature

    @property
    def pressure(self) -> float:
        """Current pressure in bar."""
        return self._pressure

    @property
    def fan_speed(self) -> int:
        """Current fan speed in RPM (0-3000)."""
        return self._fan_speed

    @property
    def motor_rpm(self) -> int:
        """Current motor RPM (0-1500)."""
        return self._motor_rpm

    @property
    def motor_on(self) -> bool:
        """Whether the motor is ON."""
        return self._motor_on

    @property
    def alarm(self) -> bool:
        """Whether the alarm is active."""
        return self._alarm

    @property
    def tick_count(self) -> int:
        """Number of ticks elapsed."""
        return self._tick_count

    # --- Register interface (raw Modbus values) ---

    def get_holding_registers(self) -> Dict[int, int]:
        """
        Get all holding register values in raw Modbus format.

        Returns
        -------
        Dict[int, int]
            Mapping of register address to raw uint16 value.
            Temperature: actual * 10 (25.0C = 250)
            Pressure: actual * 100 (3.21 bar = 321)
            Fan speed: raw RPM
            Motor RPM: raw RPM
        """
        return {
            REG_TEMPERATURE: self._temp_to_raw(self._temperature),
            REG_PRESSURE: self._pressure_to_raw(self._pressure),
            REG_FAN_SPEED: self._fan_speed,
            REG_MOTOR_RPM: self._motor_rpm,
        }

    def get_coils(self) -> Dict[int, bool]:
        """
        Get all coil values.

        Returns
        -------
        Dict[int, bool]
            Mapping of coil address to boolean state.
        """
        return {
            COIL_MOTOR_ON: self._motor_on,
            COIL_ALARM: self._alarm,
        }

    def set_holding_register(self, address: int, value: int) -> bool:
        """
        Set a holding register value (agent write).

        Only writable registers (fan_speed) are accepted. Other addresses
        are rejected since they are read-only process outputs.

        Parameters
        ----------
        address : int
            Register address.
        value : int
            Raw register value.

        Returns
        -------
        bool
            True if the write was accepted, False if the address is
            read-only or invalid.
        """
        if address == REG_FAN_SPEED:
            self._fan_speed = max(0, min(self.config.fan_speed_max, value))
            logging.info(f"Fan speed set to {self._fan_speed} RPM")
            return True

        if address in (REG_TEMPERATURE, REG_PRESSURE, REG_MOTOR_RPM):
            logging.warning(f"Register {address} is read-only (process output)")
            return False

        logging.warning(f"Unknown register address: {address}")
        return False

    def set_coil(self, address: int, value: bool) -> bool:
        """
        Set a coil value (agent write).

        Only writable coils (motor_on) are accepted. Alarm coil is
        auto-managed by the physics model.

        Parameters
        ----------
        address : int
            Coil address.
        value : bool
            Coil state.

        Returns
        -------
        bool
            True if the write was accepted, False if the address is
            read-only or invalid.
        """
        if address == COIL_MOTOR_ON:
            self._motor_on = value
            state = "ON" if value else "OFF"
            logging.info(f"Motor turned {state}")
            return True

        if address == COIL_ALARM:
            logging.warning("Alarm coil is read-only (auto-managed)")
            return False

        logging.warning(f"Unknown coil address: {address}")
        return False

    def get_state_summary(self) -> str:
        """
        Get a human-readable summary of the current process state.

        Returns
        -------
        str
            Formatted string with all process variables.
        """
        motor_state = "ON" if self._motor_on else "OFF"
        alarm_state = "ALARM" if self._alarm else "OK"
        return (
            f"T={self._temperature:.1f}C "
            f"P={self._pressure:.2f}bar "
            f"Fan={self._fan_speed}rpm "
            f"Motor={motor_state}({self._motor_rpm}rpm) "
            f"{alarm_state}"
        )

    def reset(self) -> None:
        """Reset the physics model to initial state."""
        self._temperature = self.config.initial_temp
        self._pressure = self.config.initial_pressure
        self._fan_speed = 0
        self._motor_rpm = 0
        self._motor_on = False
        self._alarm = False
        self._tick_count = 0

    # --- Internal conversion helpers ---

    @staticmethod
    def _temp_to_raw(temp: float) -> int:
        """
        Convert temperature to raw register value.

        Parameters
        ----------
        temp : float
            Temperature in Celsius.

        Returns
        -------
        int
            Raw uint16 value (temp * 10), clamped to 0-65535.
        """
        raw = int(round(temp * 10))
        return max(0, min(65535, raw))

    @staticmethod
    def _pressure_to_raw(pressure: float) -> int:
        """
        Convert pressure to raw register value.

        Parameters
        ----------
        pressure : float
            Pressure in bar.

        Returns
        -------
        int
            Raw uint16 value (pressure * 100), clamped to 0-65535.
        """
        raw = int(round(pressure * 100))
        return max(0, min(65535, raw))

    @staticmethod
    def raw_to_temp(raw: int) -> float:
        """
        Convert raw register value to temperature.

        Parameters
        ----------
        raw : int
            Raw uint16 register value.

        Returns
        -------
        float
            Temperature in Celsius.
        """
        return raw / 10.0

    @staticmethod
    def raw_to_pressure(raw: int) -> float:
        """
        Convert raw register value to pressure.

        Parameters
        ----------
        raw : int
            Raw uint16 register value.

        Returns
        -------
        float
            Pressure in bar.
        """
        return raw / 100.0
