from dataclasses import dataclass

from actions.base import Interface


@dataclass
class ModbusWriteInput:
    """
    Input interface for the Modbus write action.

    Parameters
    ----------
    action : str
        Write command in format "register:<address>:<value>" for holding
        registers or "coil:<address>:<true|false>" for coils.
    """

    action: str


@dataclass
class ModbusWrite(Interface[ModbusWriteInput, ModbusWriteInput]):
    """
    Write a value to a Modbus PLC register or coil.
    Format: "register:<address>:<value>" for holding registers,
    "coil:<address>:<true|false>" for coils.
    Example: "register:100:1500" writes value 1500 to holding register 100.
    Example: "coil:200:true" sets coil 200 to ON.
    """

    input: ModbusWriteInput
    output: ModbusWriteInput
