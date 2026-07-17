import asyncio
import fcntl
import functools
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, TypedDict

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.completion import Completion
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 120  # seconds
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class InvalidResponseError(Exception):
    pass


class ScoredTokens(TypedDict):
    prompt_token_ids: list[int]
    score_positions: list[int]
    token_logprobs: list[float]
    hidden_states_path: str | None
    hidden_states_deleted: bool


def _handle_retry_error(
    error: Exception, attempt: int, total_attempts: int
) -> float | None:
    """Handle a retry-eligible error.

    Returns backoff seconds if the caller should retry, or ``None`` on the
    final attempt.  Raises ``InvalidResponseError`` immediately.
    """
    if isinstance(error, InvalidResponseError):
        raise error
    if attempt < total_attempts:
        backoff = RETRY_BACKOFF_BASE**attempt
        logger.warning(
            "Request aborted (attempt %d/%d): %s. Retrying in %ds...",
            attempt,
            total_attempts,
            error,
            backoff,
        )
        return backoff
    logger.error("Request timed out after %d attempts: %s", total_attempts, error)
    return None


def with_retries(fn):
    """Decorator that adds retry logic with exponential backoff.

    The decorated function gains a ``max_retries`` keyword argument
    (default ``DEFAULT_MAX_RETRIES``). ``InvalidResponseError`` is never
    retried. Works for both sync and async functions.
    """
    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args, max_retries=DEFAULT_MAX_RETRIES, **kwargs):
            total_attempts = max_retries + 1
            last_error: Exception | None = None
            for attempt in range(1, total_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    backoff = _handle_retry_error(e, attempt, total_attempts)
                    if backoff is not None:
                        await asyncio.sleep(backoff)
            raise last_error  # type: ignore[misc]

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args, max_retries=DEFAULT_MAX_RETRIES, **kwargs):
        total_attempts = max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                backoff = _handle_retry_error(e, attempt, total_attempts)
                if backoff is not None:
                    time.sleep(backoff)
        raise last_error  # type: ignore[misc]

    return sync_wrapper


def _prompt_token_ids(response: Completion | ChatCompletion) -> list[int] | None:
    if isinstance(response, Completion):
        prompt_token_ids = getattr(response.choices[0], "prompt_token_ids", None)
    else:
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
    return None if prompt_token_ids is None else list(prompt_token_ids)


def _kv_hidden_states_path(response: Completion | ChatCompletion) -> str | None:
    kv_transfer_params = getattr(response, "kv_transfer_params", None)
    if isinstance(kv_transfer_params, dict):
        return kv_transfer_params.get("hidden_states_path")
    return None


def _prompt_logprobs(response: Completion | ChatCompletion) -> Any:
    if isinstance(response, Completion):
        prompt_logprobs = getattr(response.choices[0], "prompt_logprobs", None)
        if prompt_logprobs is not None:
            return prompt_logprobs

    prompt_logprobs = getattr(response, "prompt_logprobs", None)
    if prompt_logprobs is not None:
        return prompt_logprobs

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if dumped.get("choices"):
            return dumped["choices"][0].get("prompt_logprobs")
        return dumped.get("prompt_logprobs")

    return None


def _logprob_value(value: Any) -> float:
    if isinstance(value, dict):
        if "logprob" not in value:
            raise InvalidResponseError(f"Logprob entry missing 'logprob': {value}")
        return float(value["logprob"])
    logprob = getattr(value, "logprob", None)
    if logprob is None:
        raise InvalidResponseError(f"Logprob entry missing 'logprob': {value}")
    return float(logprob)


def _extract_token_logprob(prompt_logprobs: Any, position: int, token_id: int) -> float:
    try:
        position_logprobs = prompt_logprobs[position]
    except (IndexError, TypeError) as e:
        raise InvalidResponseError(
            f"Missing prompt logprobs for position {position}"
        ) from e

    if position_logprobs is None:
        raise InvalidResponseError(f"Prompt logprobs for position {position} are None")

    if isinstance(position_logprobs, dict):
        for key in (token_id, str(token_id)):
            if key in position_logprobs:
                return _logprob_value(position_logprobs[key])
        raise InvalidResponseError(
            f"Token {token_id} not found in prompt logprobs at position {position}"
        )

    if hasattr(position_logprobs, "get"):
        for key in (token_id, str(token_id)):
            value = position_logprobs.get(key)
            if value is not None:
                return _logprob_value(value)

    raise InvalidResponseError(
        f"Unsupported prompt logprobs entry at position {position}: "
        f"{type(position_logprobs).__name__}"
    )


def _delete_hidden_states_file(path_value: str | None, timeout: float = 30.0) -> bool:
    if path_value is None:
        return False

    path = Path(path_value)
    lock_path = Path(str(path) + ".lock")
    deadline = time.monotonic() + timeout
    if lock_path.exists():
        wait_for_lock(str(lock_path), timeout=timeout)
    while lock_path.exists() or not path.exists():
        if time.monotonic() >= deadline:
            break
        if lock_path.exists():
            wait_for_lock(str(lock_path), timeout=max(deadline - time.monotonic(), 0.1))
            continue
        time.sleep(0.05)

    path.unlink(missing_ok=True)
    lock_path.unlink(missing_ok=True)
    return True


def extract_output(
    response: Completion | ChatCompletion,
    token_ids: list[int],
) -> str:
    prompt_token_ids = _prompt_token_ids(response)
    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    if prompt_token_ids != token_ids:
        raise InvalidResponseError(
            f"Prompt token IDs mismatch: expected {token_ids}, got {prompt_token_ids}"
        )

    hidden_states_path = _kv_hidden_states_path(response)
    if hidden_states_path is None:
        raise InvalidResponseError("Response missing kv_transfer_params")

    return hidden_states_path


class ClientItem(TypedDict):
    input_ids: list[int]
    """The input token IDs."""

    messages: NotRequired[list[ChatCompletionMessageParam]]
    """If provided, pass `messages` to Chat Completions API
    instead of passing `token_ids` to Completions API."""


async def _poll_lock_async(fd, poll_interval):
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            await asyncio.sleep(poll_interval)


async def wait_for_lock_async(lock_path, timeout=10.0, poll_interval=0.1):
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        await asyncio.wait_for(_poll_lock_async(fd, poll_interval), timeout=timeout)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


def wait_for_lock(lock_path, timeout=10.0, poll_interval=0.1):
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


@with_retries
async def generate_hidden_states_async(
    client: openai.AsyncClient,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.

    Args:
        client: The async OpenAI client.
        model: The model ID.
        client_item: Inputs to send via the client.
        timeout: Timeout in seconds for each request attempt. None for no timeout.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    coro: Coroutine[Any, Any, Completion | ChatCompletion]
    if messages is None:
        coro = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        coro = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True},
            timeout=timeout,
        )

    res: Completion | ChatCompletion
    if timeout is not None:
        res = await asyncio.wait_for(coro, timeout=timeout)
    else:
        res = await coro

    return extract_output(res, token_ids)


@with_retries
def generate_hidden_states(
    client: openai.Client,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    res: Completion | ChatCompletion
    if messages is None:
        res = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        res = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True},
            timeout=timeout,
        )

    return extract_output(res, token_ids)


@with_retries
def score_sampled_tokens(
    client: openai.Client,
    model: str,
    prefix_token_ids: list[int],
    sampled_token_ids: list[int],
    *,
    prompt_logprobs: int = 1,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
    cleanup_hidden_states: bool = True,
    hidden_states_file_timeout: float = 30.0,
) -> ScoredTokens:
    """Score sampled tokens with the target model without reading hidden states.

    The request uses vLLM prompt logprobs over ``prefix + sampled``.  If the
    server is configured with the hidden-state connector, vLLM may still write a
    hidden-state file for this request; this function deliberately ignores that
    file and deletes it by default.
    """
    if not prefix_token_ids:
        raise ValueError("prefix_token_ids must not be empty")
    if not sampled_token_ids:
        raise ValueError("sampled_token_ids must not be empty")

    prompt = [*prefix_token_ids, *sampled_token_ids]
    score_positions = list(range(len(prefix_token_ids), len(prompt)))

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1,
        extra_body={
            "return_token_ids": True,
            "prompt_logprobs": prompt_logprobs,
        },
        timeout=timeout,
    )

    response_prompt_ids = _prompt_token_ids(response)
    if response_prompt_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")
    if response_prompt_ids != prompt:
        raise InvalidResponseError(
            "Prompt token IDs mismatch while scoring sampled tokens: "
            f"expected {prompt}, got {response_prompt_ids}"
        )

    response_prompt_logprobs = _prompt_logprobs(response)
    if response_prompt_logprobs is None:
        raise InvalidResponseError("Response missing prompt_logprobs")

    token_logprobs = [
        _extract_token_logprob(response_prompt_logprobs, pos, prompt[pos])
        for pos in score_positions
    ]

    hidden_states_path = _kv_hidden_states_path(response)
    hidden_states_deleted = False
    if cleanup_hidden_states:
        hidden_states_deleted = _delete_hidden_states_file(
            hidden_states_path, timeout=hidden_states_file_timeout
        )

    return {
        "prompt_token_ids": response_prompt_ids,
        "score_positions": score_positions,
        "token_logprobs": token_logprobs,
        "hidden_states_path": hidden_states_path,
        "hidden_states_deleted": hidden_states_deleted,
    }
