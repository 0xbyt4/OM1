from cli import (
    _check_action_exists,
    _check_background_exists,
    _check_input_exists,
    _check_llm_exists,
    _check_simulator_exists,
)


class TestCheckInputExists:

    def test_existing_input_found(self):
        assert _check_input_exists("VLMVila") is True

    def test_nonexistent_input_not_found(self):
        assert _check_input_exists("NonExistentInput12345") is False

    def test_case_sensitive(self):
        assert _check_input_exists("VLMVila") is True
        assert _check_input_exists("vlmvila") is False


class TestCheckLLMExists:

    def test_existing_llm_found(self):
        assert _check_llm_exists("OpenAILLM") is True

    def test_nonexistent_llm_not_found(self):
        assert _check_llm_exists("NonExistentLLM12345") is False


class TestCheckSimulatorExists:

    def test_existing_simulator_found(self):
        assert _check_simulator_exists("WebSim") is True

    def test_nonexistent_simulator_not_found(self):
        assert _check_simulator_exists("NonExistentSim12345") is False


class TestCheckActionExists:

    def test_existing_action_found(self):
        assert _check_action_exists("speak") is True

    def test_nonexistent_action_not_found(self):
        assert _check_action_exists("nonexistent_action_12345") is False

    def test_action_needs_interface_file(self):
        assert _check_action_exists("move") is True


class TestCheckBackgroundExists:

    def test_existing_background_found(self):
        assert _check_background_exists("Gps") is True

    def test_nonexistent_background_not_found(self):
        assert _check_background_exists("NonExistentBg12345") is False
