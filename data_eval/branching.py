"""Shared, strict step counting and prompt-prefix extraction.

A completed assistant+tool step is one assistant message with tool_calls,
followed by the tool responses for all those calls. This is one *turn*, not
one message. The normal mini-SWE-agent configuration makes one tool call per
turn, but multiple calls in a single assistant turn still count as one step.

We preserve entire message dictionaries for comparisons (including thinking,
timestamps and verbatim tool observations). Never normalize messages here.
"""
from typing import Any


def completed_step_endpoints(messages: Any) -> tuple[list[int], list[str]]:
    """Return end-exclusive message offsets for each completed assistant/tool turn.

    Also return errors that make step counting ambiguous. Does not mutate data.
    """
    if not isinstance(messages, list):
        return [], ["messages: expected a list"]
    endpoints: list[int] = []
    problems: list[str] = []
    outstanding: list[str | None] | None = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            problems.append(f"messages[{i}]: expected object")
            continue
        role = msg.get("role")
        if role == "assistant":
            if outstanding is not None:
                problems.append(f"messages[{i}]: assistant action before previous tool replies completed")
                outstanding = None
            calls = msg.get("tool_calls")
            if isinstance(calls, list) and calls:
                outstanding = []
                for j, call in enumerate(calls):
                    if not isinstance(call, dict):
                        problems.append(f"messages[{i}].tool_calls[{j}]: expected object")
                        continue
                    call_id = call.get("id")
                    outstanding.append(call_id if isinstance(call_id, str) and call_id else None)
                if not outstanding:
                    outstanding = None
        elif role == "tool":
            if outstanding is None:
                problems.append(f"messages[{i}]: tool response without pending assistant call")
                continue
            returned = msg.get("tool_call_id")
            if isinstance(returned, str) and returned and any(x is not None for x in outstanding):
                if returned not in outstanding:
                    problems.append(f"messages[{i}]: tool_call_id does not match pending tool call")
                    continue
                outstanding.remove(returned)
            else:
                # Some raw agent logs omit call IDs. Count responses by order.
                outstanding.pop(0)
            if not outstanding:
                endpoints.append(i + 1)
                outstanding = None
        elif role not in ("user", "system", "developer", "exit", "assistant"):
            # Other roles can be present, but do not count as assistant+tool steps.
            pass
    # A final outstanding call may reflect a crashed/incomplete rollout;
    # do not count it as a completed step.
    return endpoints, problems


def shared_prefix(messages: Any, branch_step: int) -> tuple[list[dict] | None, str | None]:
    """Get exact messages through *branch_step* completed assistant/tool turns.

    Step 0 includes initial context messages up to (but not including) the
    first assistant tool-call turn. Does not include later actions.
    """
    if type(branch_step) is not int or branch_step < 0:
        return None, "branch_step: expected nonnegative integer"
    endpoints, problems = completed_step_endpoints(messages)
    if problems:
        return None, "; ".join(problems)
    if branch_step > len(endpoints):
        return None, (f"branch_step={branch_step} exceeds {len(endpoints)} "
                      "completed assistant+tool pairs")
    if branch_step == 0:
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                return messages[:i], None
        return messages[:], None
    return messages[:endpoints[branch_step - 1]], None
