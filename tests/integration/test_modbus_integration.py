"""
Integration tests for the Modbus plugin using a local pymodbus TCP server.

These tests verify the full Modbus communication flow:
Provider → Sensor (input) → Action (write) against a real TCP server.

Run with: uv run pytest -m "integration" tests/integration/test_modbus_integration.py -v
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

from providers.modbus_provider import ModbusProvider

TEST_HOST = "127.0.0.1"
TEST_PORT = 15502  # Non-standard port to avoid conflicts


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset ModbusProvider singleton between tests."""
    ModbusProvider.reset()  # type: ignore
    yield
    ModbusProvider.reset()  # type: ignore


@pytest.fixture
def modbus_datastore():
    """Create a Modbus datastore with test values."""
    # pymodbus 3.x: wire address N maps to datastore index N+1,
    # so prepend a dummy value at index 0.
    # Holding registers: address 0-9
    holding_registers = ModbusSequentialDataBlock(
        0, [0, 250, 321, 1500, 0, 42] + [0] * 5
    )

    # Input registers: address 0-9
    input_registers = ModbusSequentialDataBlock(0, [0, 100, 200, 300] + [0] * 7)

    # Coils: address 0-9
    coils = ModbusSequentialDataBlock(0, [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # Discrete inputs: address 0-9
    discrete_inputs = ModbusSequentialDataBlock(0, [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    device_context = ModbusDeviceContext(
        di=discrete_inputs,
        co=coils,
        hr=holding_registers,
        ir=input_registers,
    )
    return ModbusServerContext(devices=device_context, single=True)


@pytest_asyncio.fixture
async def modbus_server(modbus_datastore):
    """Start a local Modbus TCP server for testing."""
    server_task = asyncio.create_task(
        StartAsyncTcpServer(
            context=modbus_datastore,
            address=(TEST_HOST, TEST_PORT),
        )
    )

    # Wait for server to start
    await asyncio.sleep(0.5)

    yield modbus_datastore

    await ServerAsyncStop()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_holding_registers(modbus_server):
    """Test reading holding registers from local Modbus server."""
    provider = ModbusProvider()

    result = await provider.read_holding_registers(
        TEST_HOST, TEST_PORT, address=0, count=3
    )

    assert not result.is_error
    assert result.values == [250, 321, 1500]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_input_registers(modbus_server):
    """Test reading input registers from local Modbus server."""
    provider = ModbusProvider()

    result = await provider.read_input_registers(
        TEST_HOST, TEST_PORT, address=0, count=3
    )

    assert not result.is_error
    assert result.values == [100, 200, 300]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_coils(modbus_server):
    """Test reading coils from local Modbus server."""
    provider = ModbusProvider()

    result = await provider.read_coils(TEST_HOST, TEST_PORT, address=0, count=4)

    assert not result.is_error
    assert result.values == [True, False, True, False]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_discrete_inputs(modbus_server):
    """Test reading discrete inputs from local Modbus server."""
    provider = ModbusProvider()

    result = await provider.read_discrete_inputs(
        TEST_HOST, TEST_PORT, address=0, count=4
    )

    assert not result.is_error
    assert result.values == [True, True, False, False]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_and_read_register(modbus_server):
    """Test writing a register and reading it back."""
    provider = ModbusProvider()

    # Write value 9999 to register address 5
    success = await provider.write_register(TEST_HOST, TEST_PORT, address=5, value=9999)
    assert success

    # Read it back
    result = await provider.read_holding_registers(
        TEST_HOST, TEST_PORT, address=5, count=1
    )
    assert not result.is_error
    assert result.values == [9999]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_and_read_coil(modbus_server):
    """Test writing a coil and reading it back."""
    provider = ModbusProvider()

    # Coil 1 is initially False, write True
    success = await provider.write_coil(TEST_HOST, TEST_PORT, address=1, value=True)
    assert success

    # Read it back
    result = await provider.read_coils(TEST_HOST, TEST_PORT, address=1, count=1)
    assert not result.is_error
    assert result.values == [True]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connection_reuse(modbus_server):
    """Test that multiple reads reuse the same connection."""
    provider = ModbusProvider()

    result1 = await provider.read_holding_registers(
        TEST_HOST, TEST_PORT, address=0, count=1
    )
    result2 = await provider.read_holding_registers(
        TEST_HOST, TEST_PORT, address=1, count=1
    )

    assert not result1.is_error
    assert not result2.is_error
    assert result1.values == [250]
    assert result2.values == [321]

    # Verify only one client was created
    assert len(provider._clients) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_close_all(modbus_server):
    """Test closing all connections."""
    provider = ModbusProvider()

    await provider.read_holding_registers(TEST_HOST, TEST_PORT, address=0, count=1)

    assert len(provider._clients) == 1

    await provider.close_all()

    assert len(provider._clients) == 0
