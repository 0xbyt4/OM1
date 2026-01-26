"""
Integration tests for the Modbus PLC simulator feedback loop.

These tests verify the full feedback cycle:
Agent writes → Physics updates → Sensor reads updated values.

Uses the PLC physics model with a real pymodbus TCP server to validate
that the simulator correctly closes the control loop.

Run with: uv run pytest -m "integration" tests/integration/test_modbus_feedback_loop.py -v
"""

import asyncio

import pytest
import pytest_asyncio
from pymodbus.datastore import (
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusServerContext,
)
from pymodbus.server import ServerAsyncStop, StartAsyncTcpServer

from actions.modbus_write.connector.modbus import (
    ModbusWriteConfig,
    ModbusWriteConnector,
)
from actions.modbus_write.interface import ModbusWriteInput
from providers.modbus_provider import ModbusProvider
from simulators.plugins.modbus_plc_model import (
    COIL_ALARM,
    COIL_COUNT,
    COIL_MOTOR_ON,
    HOLDING_REGISTER_COUNT,
    REG_FAN_SPEED,
    REG_MOTOR_RPM,
    REG_PRESSURE,
    REG_TEMPERATURE,
    PLCPhysicsConfig,
    PLCPhysicsModel,
)

TEST_HOST = "127.0.0.1"
TEST_PORT = 15520  # Unique port for feedback loop tests

# Block offset: pymodbus wire address N = block index N+1
BLOCK_OFFSET = 1


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset ModbusProvider singleton between tests."""
    ModbusProvider.reset()  # type: ignore
    yield
    ModbusProvider.reset()  # type: ignore


def create_datastore(model: PLCPhysicsModel) -> ModbusServerContext:
    """
    Create a Modbus server datastore from physics model state.

    Parameters
    ----------
    model : PLCPhysicsModel
        Physics model with initial values.

    Returns
    -------
    ModbusServerContext
        Server context with initialized datastore.
    """
    initial_regs = model.get_holding_registers()
    initial_coils = model.get_coils()

    hr_size = HOLDING_REGISTER_COUNT + BLOCK_OFFSET + 10
    hr_values = [0] * hr_size
    for addr, value in initial_regs.items():
        hr_values[addr + BLOCK_OFFSET] = value

    co_size = COIL_COUNT + BLOCK_OFFSET + 10
    co_values = [0] * co_size
    for addr, value in initial_coils.items():
        co_values[addr + BLOCK_OFFSET] = int(value)

    device_context = ModbusDeviceContext(
        di=ModbusSequentialDataBlock(0, [0] * co_size),
        co=ModbusSequentialDataBlock(0, co_values),
        hr=ModbusSequentialDataBlock(0, hr_values),
        ir=ModbusSequentialDataBlock(0, [0] * hr_size),
    )
    return ModbusServerContext(devices=device_context, single=True)


def read_agent_inputs(context: ModbusServerContext, model: PLCPhysicsModel) -> None:
    """
    Sync agent-written values from datastore to physics model.

    Parameters
    ----------
    context : ModbusServerContext
        The Modbus server datastore.
    model : PLCPhysicsModel
        The physics model to update.
    """
    device = context[0]

    fan_values = device.getValues(3, REG_FAN_SPEED, count=1)
    if isinstance(fan_values, list) and len(fan_values) > 0:
        if fan_values[0] != model.fan_speed:
            model.set_holding_register(REG_FAN_SPEED, fan_values[0])

    motor_values = device.getValues(1, COIL_MOTOR_ON, count=1)
    if isinstance(motor_values, list) and len(motor_values) > 0:
        new_motor = bool(motor_values[0])
        if new_motor != model.motor_on:
            model.set_coil(COIL_MOTOR_ON, new_motor)


def write_physics_outputs(context: ModbusServerContext, model: PLCPhysicsModel) -> None:
    """
    Write physics model outputs to the Modbus datastore.

    Parameters
    ----------
    context : ModbusServerContext
        The Modbus server datastore.
    model : PLCPhysicsModel
        The physics model with updated values.
    """
    device = context[0]
    regs = model.get_holding_registers()
    coils = model.get_coils()

    device.setValues(3, REG_TEMPERATURE, [regs[REG_TEMPERATURE]])
    device.setValues(3, REG_PRESSURE, [regs[REG_PRESSURE]])
    device.setValues(3, REG_MOTOR_RPM, [regs[REG_MOTOR_RPM]])
    device.setValues(1, COIL_ALARM, [int(coils[COIL_ALARM])])


async def run_physics_ticks(
    context: ModbusServerContext,
    model: PLCPhysicsModel,
    num_ticks: int,
    tick_interval: float = 0.05,
) -> None:
    """
    Run physics ticks, syncing with the datastore each tick.

    Parameters
    ----------
    context : ModbusServerContext
        The Modbus server datastore.
    model : PLCPhysicsModel
        The physics model.
    num_ticks : int
        Number of ticks to run.
    tick_interval : float
        Seconds between ticks.
    """
    for _ in range(num_ticks):
        read_agent_inputs(context, model)
        model.tick()
        write_physics_outputs(context, model)
        await asyncio.sleep(tick_interval)


@pytest_asyncio.fixture
async def simulator():
    """
    Start a PLC simulator with Modbus TCP server.

    Yields a tuple of (server_context, physics_model) for test use.
    The physics model uses deterministic settings (no pressure randomness).
    """
    config = PLCPhysicsConfig(
        initial_temp=25.0,
        ambient_temp=22.0,
        motor_heat_rate=1.0,
        max_fan_cooling=2.0,
        ambient_drift_factor=0.01,
        motor_ramp_up=300,  # Fast ramp for tests
        motor_ramp_down=500,
        pressure_volatility=0.0,  # Deterministic pressure
    )
    model = PLCPhysicsModel(config=config)
    context = create_datastore(model)

    server_task = asyncio.create_task(
        StartAsyncTcpServer(
            context=context,
            address=(TEST_HOST, TEST_PORT),
        )
    )
    await asyncio.sleep(0.5)

    yield context, model

    await ServerAsyncStop()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.integration
@pytest.mark.asyncio
class TestFeedbackLoop:
    """Integration tests for the PLC simulator feedback loop."""

    async def test_read_initial_values(self, simulator):
        """Test that ModbusProvider reads correct initial register values."""
        context, model = simulator
        provider = ModbusProvider()

        result = await provider.read_holding_registers(
            TEST_HOST, TEST_PORT, address=0, count=4
        )
        assert not result.is_error
        assert result.values[0] == 250  # 25.0C * 10
        assert result.values[1] == 300  # 3.0 bar * 100 (deterministic)
        assert result.values[2] == 0  # Fan speed
        assert result.values[3] == 0  # Motor RPM

    async def test_write_motor_on_changes_rpm(self, simulator):
        """Test that writing motor ON causes RPM to ramp up after physics ticks."""
        context, model = simulator
        provider = ModbusProvider()

        # Write motor ON via Modbus
        success = await provider.write_coil(TEST_HOST, TEST_PORT, address=0, value=True)
        assert success

        # Run physics ticks
        await run_physics_ticks(context, model, num_ticks=5)

        # Read motor RPM - should have ramped up
        result = await provider.read_holding_registers(
            TEST_HOST, TEST_PORT, address=3, count=1
        )
        assert not result.is_error
        assert result.values[0] > 0  # Motor RPM > 0

    async def test_motor_heats_temperature(self, simulator):
        """Test that motor operation increases temperature over time."""
        context, model = simulator
        provider = ModbusProvider()

        # Read initial temperature
        initial = await provider.read_holding_registers(
            TEST_HOST, TEST_PORT, address=0, count=1
        )
        initial_temp_raw = initial.values[0]

        # Turn motor ON
        await provider.write_coil(TEST_HOST, TEST_PORT, address=0, value=True)

        # Run many ticks for temperature to accumulate
        await run_physics_ticks(context, model, num_ticks=30)

        # Read updated temperature
        updated = await provider.read_holding_registers(
            TEST_HOST, TEST_PORT, address=0, count=1
        )
        assert updated.values[0] > initial_temp_raw

    async def test_fan_counteracts_heating(self, simulator):
        """Test that fan speed reduces temperature rise from motor."""
        context, model = simulator
        provider = ModbusProvider()

        # Run with motor ON only
        await provider.write_coil(TEST_HOST, TEST_PORT, address=0, value=True)
        await run_physics_ticks(context, model, num_ticks=20)
        temp_no_fan = (
            await provider.read_holding_registers(
                TEST_HOST, TEST_PORT, address=0, count=1
            )
        ).values[0]

        # Reset model state
        model.reset()
        write_physics_outputs(context, model)

        # Reset provider for fresh connection
        ModbusProvider.reset()  # type: ignore
        provider = ModbusProvider()

        # Run with motor ON + fan at max
        await provider.write_coil(TEST_HOST, TEST_PORT, address=0, value=True)
        await provider.write_register(TEST_HOST, TEST_PORT, address=2, value=3000)
        await run_physics_ticks(context, model, num_ticks=20)
        temp_with_fan = (
            await provider.read_holding_registers(
                TEST_HOST, TEST_PORT, address=0, count=1
            )
        ).values[0]

        # Temperature with fan should be lower than without
        assert temp_with_fan < temp_no_fan

    async def test_alarm_triggers_from_overheating(self, simulator):
        """Test that alarm coil activates when temperature exceeds threshold."""
        context, model = simulator
        provider = ModbusProvider()

        # Force high temperature via direct model manipulation
        model._temperature = 85.0
        model.tick()
        write_physics_outputs(context, model)

        # Read alarm coil
        coil_result = await provider.read_coils(
            TEST_HOST, TEST_PORT, address=1, count=1
        )
        assert not coil_result.is_error
        assert coil_result.values[0] is True  # Alarm ON

    async def test_alarm_clears_after_cooling(self, simulator):
        """Test that alarm clears when temperature drops below low threshold."""
        context, model = simulator
        provider = ModbusProvider()

        # Trigger alarm
        model._temperature = 85.0
        model.tick()
        write_physics_outputs(context, model)

        # Verify alarm is ON
        coil_result = await provider.read_coils(
            TEST_HOST, TEST_PORT, address=1, count=1
        )
        assert coil_result.values[0] is True

        # Cool down below hysteresis low threshold
        model._temperature = 70.0
        model.tick()
        write_physics_outputs(context, model)

        # Verify alarm cleared
        coil_result = await provider.read_coils(
            TEST_HOST, TEST_PORT, address=1, count=1
        )
        assert coil_result.values[0] is False

    async def test_write_connector_sets_fan_speed(self, simulator):
        """Test ModbusWriteConnector correctly writes fan speed."""
        context, model = simulator

        config = ModbusWriteConfig(host=TEST_HOST, port=TEST_PORT, device_id=1)
        connector = ModbusWriteConnector(config)

        # Use connector to write fan speed
        action = ModbusWriteInput(action="register:2:2000")
        await connector.connect(action)

        # Run physics tick to sync
        await run_physics_ticks(context, model, num_ticks=1)

        # Verify fan speed in physics model
        assert model.fan_speed == 2000

    async def test_write_connector_sets_motor(self, simulator):
        """Test ModbusWriteConnector correctly toggles motor coil."""
        context, model = simulator

        config = ModbusWriteConfig(host=TEST_HOST, port=TEST_PORT, device_id=1)
        connector = ModbusWriteConnector(config)

        # Turn motor ON via connector
        action = ModbusWriteInput(action="coil:0:true")
        await connector.connect(action)

        # Run physics tick to sync
        await run_physics_ticks(context, model, num_ticks=1)

        assert model.motor_on is True
        assert model.motor_rpm > 0

    async def test_full_control_cycle(self, simulator):
        """
        Test complete control cycle:
        1. Agent reads initial values (cold, motor off)
        2. Agent turns motor ON
        3. Temperature rises
        4. Agent sets fan to cool
        5. Temperature stabilizes lower
        """
        context, model = simulator
        provider = ModbusProvider()

        # Step 1: Read initial state
        regs = await provider.read_holding_registers(
            TEST_HOST, TEST_PORT, address=0, count=4
        )
        assert regs.values[0] == 250  # 25.0C
        assert regs.values[3] == 0  # Motor off

        # Step 2: Turn motor ON
        await provider.write_coil(TEST_HOST, TEST_PORT, address=0, value=True)
        await run_physics_ticks(context, model, num_ticks=15)

        # Step 3: Verify temperature rose
        regs_heated = await provider.read_holding_registers(
            TEST_HOST, TEST_PORT, address=0, count=4
        )
        assert regs_heated.values[0] > 250  # Temperature increased
        assert regs_heated.values[3] > 0  # Motor RPM > 0
        temp_before_fan = regs_heated.values[0]

        # Step 4: Set fan speed to cool down
        await provider.write_register(TEST_HOST, TEST_PORT, address=2, value=3000)
        await run_physics_ticks(context, model, num_ticks=20)

        # Step 5: Verify temperature dropped or stabilized lower
        regs_cooled = await provider.read_holding_registers(
            TEST_HOST, TEST_PORT, address=0, count=4
        )
        assert regs_cooled.values[0] < temp_before_fan  # Temperature decreased
        assert regs_cooled.values[2] == 3000  # Fan speed preserved
