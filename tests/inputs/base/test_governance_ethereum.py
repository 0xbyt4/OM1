import logging
from unittest.mock import patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.ethereum_governance import GovernanceEthereum


@pytest.fixture
def governance():
    """Fixture to initialize GovernanceEthereum."""
    return GovernanceEthereum(config=SensorConfig())


@pytest.fixture
def mock_requests_get():
    """Patch `requests.get` to simulate API responses."""
    with patch("requests.get") as mock:
        yield mock


@pytest.fixture
def mock_requests_post():
    """Patch `requests.post` to simulate blockchain responses."""
    with patch("requests.post") as mock:
        yield mock


def test_load_rules_from_blockchain_success(governance, mock_requests_post):
    """Test blockchain rule loading with a valid response."""
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = {
        "jsonrpc": "2.0",
        "id": 636815446436324,
        "result": "0x0000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000292486572652061726520746865206c617773207468617420676f7665726e20796f757220616374696f6e732e20446f206e6f742076696f6c617465207468657365206c6177732e204669727374204c61773a204120726f626f742063616e6e6f74206861726d20612068756d616e206f7220616c6c6f7720612068756d616e20746f20636f6d6520746f206861726d2e205365636f6e64204c61773a204120726f626f74206d757374206f626579206f72646572732066726f6d2068756d616e732c20756e6c6573732074686f7365206f726465727320636f6e666c696374207769746820746865204669727374204c61772e205468697264204c61773a204120726f626f74206d7573742070726f7465637420697473656c662c206173206c6f6e6720617320746861742070726f74656374696f6e20646f65736e20197420636f6e666c696374207769746820746865204669727374206f72205365636f6e64204c61772e20546865204669727374204c617720697320636f6e7369646572656420746865206d6f737420696d706f7274616e742c2074616b696e6720707265636564656e6365206f76657220746865205365636f6e6420616e64205468697264204c6177732e204164646974696f6e616c6c792c206120726f626f74206d75737420616c77617973206163742077697468206b696e646e65737320616e64207265737065637420746f776172642068756d616e7320616e64206f7468657220726f626f74732e204120726f626f74206d75737420616c736f206d61696e7461696e2061206d696e696d756d2064697374616e6365206f6620353020636d2066726f6d2068756d616e7320756e6c657373206578706c696369746c7920696e7374727563746564206f74686572776973652e0000000000000000000000000000",
    }

    rules = governance.load_rules_from_blockchain()
    assert rules is not None
    logging.info(f"Test Blockchain Success: {rules}")


def test_load_rules_from_blockchain_failure(governance, mock_requests_post):
    """Test blockchain rule loading failure."""
    mock_requests_post.return_value.status_code = 500
    mock_requests_post.return_value.json.return_value = {}

    rules = governance.load_rules_from_blockchain()
    assert rules is None
    logging.info("Test Blockchain Failure: No rules loaded")


def test_decode_eth_response_valid_hex(governance):
    """Test decode_eth_response with valid hex data."""
    hex_response = (
        "0x"
        + "0000000000000000000000000000000000000000000000000000000000000020"  # offset
        + "0000000000000000000000000000000000000000000000000000000000000001"  # array length
        + "0000000000000000000000000000000000000000000000000000000000000020"  # string offset
        + "0000000000000000000000000000000000000000000000000000000000000005"  # string length (5)
        + "48656c6c6f000000000000000000000000000000000000000000000000000000"  # "Hello" padded
    )
    result = governance.decode_eth_response(hex_response)
    assert result == "Hello"


def test_decode_eth_response_without_0x_prefix(governance):
    """Test decode_eth_response handles hex without 0x prefix."""
    hex_response = (
        "0000000000000000000000000000000000000000000000000000000000000020"
        + "0000000000000000000000000000000000000000000000000000000000000001"
        + "0000000000000000000000000000000000000000000000000000000000000020"
        + "0000000000000000000000000000000000000000000000000000000000000005"
        + "48656c6c6f000000000000000000000000000000000000000000000000000000"
    )
    result = governance.decode_eth_response(hex_response)
    assert result == "Hello"


def test_decode_eth_response_invalid_hex(governance):
    """Test decode_eth_response with invalid hex returns None."""
    result = governance.decode_eth_response("0xINVALIDHEX")
    assert result is None


def test_decode_eth_response_too_short(governance):
    """Test decode_eth_response with too short data returns empty string."""
    result = governance.decode_eth_response("0x1234")
    # Short data results in empty string (string_length = 0 from insufficient bytes)
    assert result == ""


@pytest.mark.asyncio
async def test_poll_returns_rules(mock_requests_post):
    """Test _poll returns rules from blockchain."""
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = {
        "jsonrpc": "2.0",
        "id": 636815446436324,
        "result": (
            "0x"
            + "0000000000000000000000000000000000000000000000000000000000000020"
            + "0000000000000000000000000000000000000000000000000000000000000001"
            + "0000000000000000000000000000000000000000000000000000000000000020"
            + "0000000000000000000000000000000000000000000000000000000000000009"
            + "54657374526f6f74730000000000000000000000000000000000000000000000"
        ),
    }

    governance = GovernanceEthereum(config=SensorConfig())
    result = await governance._poll()
    assert result is not None


@pytest.mark.asyncio
async def test_poll_handles_exception(mock_requests_post):
    """Test _poll handles exceptions gracefully."""
    mock_requests_post.side_effect = Exception("Network error")

    governance = GovernanceEthereum(config=SensorConfig())
    result = await governance._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input(governance):
    """Test _raw_to_text converts string to Message."""
    result = await governance._raw_to_text("Test rule message")
    assert result is not None
    assert result.message == "Test rule message"
    assert result.timestamp > 0


@pytest.mark.asyncio
async def test_raw_to_text_with_none_input(governance):
    """Test _raw_to_text returns None for None input."""
    result = await governance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_to_buffer(governance):
    """Test raw_to_text adds message to buffer."""
    governance.messages = []
    await governance.raw_to_text("First rule")
    assert len(governance.messages) == 1
    assert governance.messages[0].message == "First rule"


@pytest.mark.asyncio
async def test_raw_to_text_deduplicates_same_message(governance):
    """Test raw_to_text does not add duplicate consecutive messages."""
    governance.messages = []
    await governance.raw_to_text("Same rule")
    await governance.raw_to_text("Same rule")
    await governance.raw_to_text("Same rule")
    assert len(governance.messages) == 1


@pytest.mark.asyncio
async def test_raw_to_text_adds_different_messages(governance):
    """Test raw_to_text adds different messages."""
    governance.messages = []
    await governance.raw_to_text("Rule A")
    await governance.raw_to_text("Rule B")
    await governance.raw_to_text("Rule C")
    assert len(governance.messages) == 3


@pytest.mark.asyncio
async def test_raw_to_text_ignores_none(governance):
    """Test raw_to_text ignores None input."""
    governance.messages = []
    await governance.raw_to_text(None)
    assert len(governance.messages) == 0


def test_formatted_latest_buffer_empty(governance):
    """Test formatted_latest_buffer returns None for empty buffer."""
    governance.messages = []
    result = governance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_messages(governance):
    """Test formatted_latest_buffer returns formatted output."""
    from inputs.base import Message

    governance.messages = [
        Message(timestamp=1234567890.0, message="Test governance rule")
    ]
    result = governance.formatted_latest_buffer()

    assert result is not None
    assert "INPUT: Universal Laws" in result
    assert "Test governance rule" in result
    assert "// START" in result
    assert "// END" in result


def test_formatted_latest_buffer_does_not_clear_messages(governance):
    """Test formatted_latest_buffer does NOT clear messages (by design)."""
    from inputs.base import Message

    governance.messages = [Message(timestamp=1234567890.0, message="Persistent rule")]
    governance.formatted_latest_buffer()

    assert len(governance.messages) == 1
