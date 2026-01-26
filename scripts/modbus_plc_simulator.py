#!/usr/bin/env python3
"""
Standalone Modbus PLC Simulator with Physics Engine.

Runs a pymodbus TCP server backed by a thermal physics model that creates
a realistic feedback loop: agent writes (motor ON, fan speed) affect the
simulated process (temperature, pressure, RPM), and the updated values
are readable by the agent through standard Modbus reads.

Usage:
    uv run python scripts/modbus_plc_simulator.py
    uv run python scripts/modbus_plc_simulator.py --port 5020 --tick-rate 1.0
    uv run python scripts/modbus_plc_simulator.py --initial-temp 30.0

Register Layout:
    Holding Registers:
        0 : Temperature (scale 0.1 C) - read-only output
        1 : Pressure (scale 0.01 bar) - read-only output
        2 : Fan Speed (RPM, 0-3000) - writable by agent
        3 : Motor RPM (0-1500) - read-only output

    Coils:
        0 : Motor ON/OFF - writable by agent
        1 : Alarm - auto-managed (>80C ON, <75C OFF)
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import cast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pymodbus.datastore import (
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusServerContext,
)
from pymodbus.server import ServerAsyncStop, StartAsyncTcpServer

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# pymodbus 3.x: wire address N maps to block index N+1
# when using ModbusSequentialDataBlock(0, values).
BLOCK_OFFSET = 1


def create_datastore(model: PLCPhysicsModel) -> ModbusServerContext:
    """
    Create a Modbus server datastore initialized from the physics model.

    Parameters
    ----------
    model : PLCPhysicsModel
        The physics model providing initial register values.

    Returns
    -------
    ModbusServerContext
        Configured server context with holding registers and coils.
    """
    # Get initial values from physics model
    initial_regs = model.get_holding_registers()
    initial_coils = model.get_coils()

    # Build holding register block: dummy at index 0, then register values
    # Extra registers for safety margin
    hr_size = HOLDING_REGISTER_COUNT + BLOCK_OFFSET + 10
    hr_values = [0] * hr_size
    for addr, value in initial_regs.items():
        hr_values[addr + BLOCK_OFFSET] = value

    # Build coil block: dummy at index 0, then coil values
    co_size = COIL_COUNT + BLOCK_OFFSET + 10
    co_values = [0] * co_size
    for addr, value in initial_coils.items():
        co_values[addr + BLOCK_OFFSET] = int(value)

    # Input registers and discrete inputs (not used but required by context)
    ir_values = [0] * hr_size
    di_values = [0] * co_size

    holding_registers = ModbusSequentialDataBlock(0, hr_values)
    coils = ModbusSequentialDataBlock(0, co_values)
    input_registers = ModbusSequentialDataBlock(0, ir_values)
    discrete_inputs = ModbusSequentialDataBlock(0, di_values)

    device_context = ModbusDeviceContext(
        di=discrete_inputs,
        co=coils,
        hr=holding_registers,
        ir=input_registers,
    )

    return ModbusServerContext(devices=device_context, single=True)


def read_agent_inputs(
    context: ModbusServerContext,
    model: PLCPhysicsModel,
) -> None:
    """
    Read agent-written values from the datastore and apply to physics model.

    The agent may have written to fan_speed (holding register 2) or
    motor_on (coil 0) via Modbus write commands. This function reads
    those values from the shared datastore and updates the physics model.

    Parameters
    ----------
    context : ModbusServerContext
        The Modbus server datastore.
    model : PLCPhysicsModel
        The physics model to update.
    """
    device = cast(ModbusDeviceContext, context[0])  # single=True, device 0

    # Read fan speed from holding register
    # DeviceContext.getValues uses wire addresses (no block offset needed)
    fan_values = device.getValues(3, REG_FAN_SPEED, count=1)
    if isinstance(fan_values, list) and len(fan_values) > 0:
        new_fan_speed = fan_values[0]
        if new_fan_speed != model.fan_speed:
            model.set_holding_register(REG_FAN_SPEED, new_fan_speed)

    # Read motor ON/OFF from coil
    motor_values = device.getValues(1, COIL_MOTOR_ON, count=1)
    if isinstance(motor_values, list) and len(motor_values) > 0:
        new_motor_on = bool(motor_values[0])
        if new_motor_on != model.motor_on:
            model.set_coil(COIL_MOTOR_ON, new_motor_on)


def write_physics_outputs(
    context: ModbusServerContext,
    model: PLCPhysicsModel,
) -> None:
    """
    Write physics model outputs to the Modbus datastore.

    After a physics tick, this function updates the shared datastore
    with the new temperature, pressure, motor RPM, and alarm values
    so the agent can read them via Modbus.

    Parameters
    ----------
    context : ModbusServerContext
        The Modbus server datastore.
    model : PLCPhysicsModel
        The physics model with updated values.
    """
    device = cast(ModbusDeviceContext, context[0])
    regs = model.get_holding_registers()
    coils = model.get_coils()

    # Write holding registers (func_code 3 = holding registers)
    # DeviceContext.setValues uses wire addresses (no block offset needed)
    device.setValues(3, REG_TEMPERATURE, [regs[REG_TEMPERATURE]])
    device.setValues(3, REG_PRESSURE, [regs[REG_PRESSURE]])
    device.setValues(3, REG_MOTOR_RPM, [regs[REG_MOTOR_RPM]])
    # Do NOT overwrite REG_FAN_SPEED - that's agent-controlled

    # Write coils (func_code 1 = coils)
    device.setValues(1, COIL_ALARM, [int(coils[COIL_ALARM])])
    # Do NOT overwrite COIL_MOTOR_ON - that's agent-controlled


async def physics_loop(
    context: ModbusServerContext,
    model: PLCPhysicsModel,
    tick_rate: float,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Main physics simulation loop.

    Runs continuously, reading agent inputs from the datastore, advancing
    the physics model, and writing updated values back to the datastore.

    Parameters
    ----------
    context : ModbusServerContext
        The Modbus server datastore (shared with TCP server).
    model : PLCPhysicsModel
        The physics model to advance.
    tick_rate : float
        Seconds between physics ticks.
    shutdown_event : asyncio.Event
        Event to signal graceful shutdown.
    """
    logger.info(
        f"Physics loop started (tick_rate={tick_rate}s, "
        f"initial_temp={model.temperature:.1f}C)"
    )

    last_summary = ""

    while not shutdown_event.is_set():
        # 1. Read agent inputs from datastore
        read_agent_inputs(context, model)

        # 2. Advance physics
        model.tick()

        # 3. Write updated values back to datastore
        write_physics_outputs(context, model)

        # 4. Log state (only when changed)
        summary = model.get_state_summary()
        if summary != last_summary:
            logger.info(f"[Tick {model.tick_count:>5}] {summary}")
            last_summary = summary

        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=tick_rate,
            )
        except asyncio.TimeoutError:
            pass


async def run_simulator(
    host: str,
    port: int,
    tick_rate: float,
    initial_temp: float,
) -> None:
    """
    Run the PLC simulator with Modbus TCP server and physics engine.

    Parameters
    ----------
    host : str
        Hostname or IP address to bind the server.
    port : int
        TCP port number for the Modbus server.
    tick_rate : float
        Seconds between physics ticks.
    initial_temp : float
        Starting temperature in Celsius.
    """
    # Create physics model
    config = PLCPhysicsConfig(initial_temp=initial_temp)
    model = PLCPhysicsModel(config=config)

    # Create datastore
    context = create_datastore(model)

    # Setup shutdown event
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()
        asyncio.ensure_future(ServerAsyncStop())

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start physics loop
    physics_task = asyncio.create_task(
        physics_loop(context, model, tick_rate, shutdown_event)
    )

    logger.info(f"PLC Simulator running at {host}:{port}")
    logger.info(
        "Register layout: "
        "HR0=Temperature(x0.1C) HR1=Pressure(x0.01bar) "
        "HR2=FanSpeed(RPM) HR3=MotorRPM | "
        "COIL0=MotorON COIL1=Alarm"
    )

    try:
        # Start Modbus TCP server (blocking)
        await StartAsyncTcpServer(
            context=context,
            address=(host, port),
        )
    except asyncio.CancelledError:
        pass
    finally:
        shutdown_event.set()
        physics_task.cancel()
        try:
            await physics_task
        except asyncio.CancelledError:
            pass
        logger.info(
            f"Simulator stopped after {model.tick_count} ticks. "
            f"Final state: {model.get_state_summary()}"
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with host, port, tick_rate, and initial_temp.
    """
    parser = argparse.ArgumentParser(
        description="Modbus PLC Simulator with Physics Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Hostname or IP address to bind the Modbus TCP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5020,
        help="TCP port number for the Modbus server",
    )
    parser.add_argument(
        "--tick-rate",
        type=float,
        default=0.5,
        help="Seconds between physics simulation ticks",
    )
    parser.add_argument(
        "--initial-temp",
        type=float,
        default=22.0,
        help="Starting temperature in Celsius",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the PLC simulator."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Modbus PLC Simulator")
    logger.info("=" * 60)
    logger.info(f"  Host:         {args.host}")
    logger.info(f"  Port:         {args.port}")
    logger.info(f"  Tick Rate:    {args.tick_rate}s")
    logger.info(f"  Initial Temp: {args.initial_temp}C")
    logger.info("=" * 60)

    try:
        asyncio.run(
            run_simulator(
                host=args.host,
                port=args.port,
                tick_rate=args.tick_rate,
                initial_temp=args.initial_temp,
            )
        )
    except KeyboardInterrupt:
        logger.info("Simulator stopped by user")


if __name__ == "__main__":
    main()
