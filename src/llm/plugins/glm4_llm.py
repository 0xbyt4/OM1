"""GLM-4 LLM plugin for ZhiPu AI models with function calling support."""

import logging
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)

# GLM-4 default configuration
DEFAULT_GLM4_MODEL = "glm-4.7"
DEFAULT_GLM4_BASE_URL = "https://api.openmind.org/api/core/glm4"


class GLM4LLM(LLM[R]):
    """
    ZhiPu AI GLM-4 Language Model implementation using OpenAI-compatible API.

    GLM-4.7 is ZhiPu AI's flagship model featuring:
    - 358B parameters (Mixture-of-Experts architecture)
    - 200K context window
    - 128K max output tokens
    - Advanced tool/function calling support
    - Optimized for coding, reasoning, and agentic tasks

    This class implements the LLM interface for GLM-4 models, handling
    configuration, authentication, and async API communication.

    Parameters
    ----------
    config : LLMConfig
        Configuration object containing API settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation. If provided,
        the LLM will use function calls instead of structured JSON output.

    Examples
    --------
    Configuration in JSON5:
    ```json5
    {
        "cortex_llm": {
            "type": "GLM4LLM",
            "config": {
                "api_key": "openmind_free",
                "model": "glm-4.7",
                "agent_name": "MyRobot",
                "history_length": 3
            }
        }
    }
    ```

    For direct ZhiPu API access (requires your own API key):
    ```json5
    {
        "cortex_llm": {
            "type": "GLM4LLM",
            "config": {
                "api_key": "your-zhipu-api-key",
                "base_url": "https://api.z.ai/api/paas/v4/",
                "model": "glm-4.7"
            }
        }
    }
    ```
    """

    def __init__(
        self,
        config: LLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the GLM-4 LLM instance.

        Parameters
        ----------
        config : LLMConfig
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.

        Raises
        ------
        ValueError
            If api_key is not provided in config.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key for GLM-4")

        if not config.model:
            self._config.model = DEFAULT_GLM4_MODEL

        self._client = openai.AsyncOpenAI(
            base_url=config.base_url or DEFAULT_GLM4_BASE_URL,
            api_key=config.api_key,
        )

        self.history_manager = LLMHistoryManager(self._config, self._client)

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
        """
        Send a prompt to the GLM-4 API and get a structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]]
            List of message dictionaries containing conversation history.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails or an error occurs.

        Notes
        -----
        This method supports GLM-4's function calling capability. When
        available_actions are provided, the model will use tool calls
        to determine actions.
        """
        try:
            logging.debug(f"GLM-4 LLM input: {prompt}")
            logging.debug(f"GLM-4 LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self._config.model or DEFAULT_GLM4_MODEL,
                messages=T.cast(T.Any, formatted_messages),
                tools=T.cast(T.Any, self.function_schemas),
                tool_choice="auto",
                timeout=self._config.timeout,
            )

            message = response.choices[0].message
            self.io_provider.llm_end_time = time.time()

            if message.tool_calls:
                logging.info(f"GLM-4 received {len(message.tool_calls)} function calls")
                logging.debug(f"GLM-4 function calls: {message.tool_calls}")

                function_call_data = [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]

                actions = convert_function_calls_to_actions(function_call_data)

                result = CortexOutputModel(actions=actions)
                logging.info(f"GLM-4 LLM function call output: {result}")
                return T.cast(R, result)

            return None
        except openai.APIConnectionError as e:
            logging.error(f"GLM-4 API connection error: {e}")
            return None
        except openai.RateLimitError as e:
            logging.error(f"GLM-4 API rate limit exceeded: {e}")
            return None
        except openai.APIStatusError as e:
            logging.error(f"GLM-4 API status error: {e.status_code} - {e.message}")
            return None
        except Exception as e:
            logging.error(f"GLM-4 API error: {e}")
            return None
