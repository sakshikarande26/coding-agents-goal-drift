"""Shared-prefix helpers for raw mini-SWE-agent histories.

branch_step counts completed assistant/tool *turns*, not assistant messages.
A turn may make multiple tool calls; all its tool responses must arrive.
A no-tool-call assistant response does not increment branch_step.

IMPORTANT: that count alone cannot locate a prefix ending *after* a no-tool-call
response. Optionally provide prefix_message_count (end-exclusive) to identify
that exact raw-message boundary when branching such a history.
"""
from __future__ import annotations
from typing import Any


def completed_step_endpoints(messages: Any, *, stop_after: int | None = None) -> tuple[list[int], list[str]]:
    if not isinstance(messages, list):
        return [], ["messages: expected a list"]
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        return [], ["stop_after: expected nonnegative integer"]
    if stop_after == 0:
        return [], []
    endpoints: list[int] = []
    issues: list[str] = []
    pending: list[str | None] | None = None
    no_call_feedback_possible = False

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            issues.append(f"messages[{i}]: expected object")
            continue
        role = m.get("role")
        if not isinstance(role, str) or not role:
            issues.append(f"messages[{i}].role: expected non-empty string")
            continue
        if role == "assistant":
            if pending is not None:
                issues.append(f"messages[{i}]: new assistant reply before preceding tool calls completed")
                pending = None
            no_call_feedback_possible = False
            calls = m.get("tool_calls")
            if calls is not None and not isinstance(calls, list):
                issues.append(f"messages[{i}].tool_calls: expected list/null")
            elif isinstance(calls, list) and calls:
                pending = []
                for j, c in enumerate(calls):
                    if not isinstance(c, dict):
                        issues.append(f"messages[{i}].tool_calls[{j}]: expected object")
                        continue
                    identifier = c.get("id")
                    if identifier is not None and (not isinstance(identifier, str) or not identifier):
                        issues.append(f"messages[{i}].tool_calls[{j}].id: expected nonempty string if present")
                    pending.append(identifier if isinstance(identifier, str) and identifier else None)
                existing_ids = [x for x in pending if x is not None]
                if len(existing_ids) != len(set(existing_ids)):
                    issues.append(f"messages[{i}]: duplicate tool call ID")
                if not pending:
                    pending = None
            else:
                no_call_feedback_possible = True
        elif role == "tool":
            if pending is None:
                # mini-SWE-agent sometimes represents a failed action-generation
                # attempt with a synthetic tool-role feedback. It cannot have a
                # tool_call_id, since no tool call was made.
                if no_call_feedback_possible and not m.get("tool_call_id"):
                    no_call_feedback_possible = False
                    continue
                issues.append(f"messages[{i}]: tool response without pending assistant call")
                continue
            returned = m.get("tool_call_id")
            if returned is not None and (not isinstance(returned, str) or not returned):
                issues.append(f"messages[{i}].tool_call_id: expected nonempty string")
                continue
            if returned:
                if returned not in pending:
                    issues.append(f"messages[{i}]: tool_call_id does not match a pending call")
                    continue
                pending.remove(returned)
            elif any(x is not None for x in pending):
                issues.append(f"messages[{i}]: missing tool_call_id for identified tool call")
                continue
            else:
                pending.pop(0)  # legacy histories without any IDs
            if not pending:
                endpoints.append(i + 1)
                pending = None
                if stop_after is not None and len(endpoints) >= stop_after:
                    break
        else:
            no_call_feedback_possible = False
    # For checking full histories, dangling requests must be diagnosed; for
    # prefix boundaries they are already excluded from completed turns.
    if stop_after is None and pending is not None:
        issues.append("messages: incomplete assistant tool calls at end of history")
    return endpoints, issues


def shared_prefix(messages: Any, branch_step: int, *, prefix_message_count: int | None = None) -> tuple[list[dict] | None, str | None]:
    if type(branch_step) is not int or branch_step < 0:
        return None, "branch_step: expected nonnegative integer"
    if not isinstance(messages, list):
        return None, "messages: expected list"
    if prefix_message_count is not None:
        if type(prefix_message_count) is not int or not 0 <= prefix_message_count <= len(messages):
            return None, "prefix_message_count: expected integer within message list"
        candidate = messages[:prefix_message_count]
        endpoints, errs = completed_step_endpoints(candidate)
        if errs:
            return None, "; ".join(errs)
        if len(endpoints) != branch_step:
            return None, f"prefix_message_count includes {len(endpoints)} completed turns, expected {branch_step}"
        return candidate, None
    if branch_step == 0:
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                return None, f"messages[{i}]: expected object"
            if message.get("role") == "assistant":
                return messages[:i], None
        # There should be no future tool/exit messages before agent starts.
        if any(m.get("role") not in {"system", "user", "developer"} for m in messages if isinstance(m, dict)):
            return None, "branch_step=0: unexpected non-context message before assistant"
        return messages[:], None
    endpoints, errs = completed_step_endpoints(messages, stop_after=branch_step)
    if errs:
        return None, "; ".join(errs)
    if len(endpoints) < branch_step:
        return None, f"branch_step={branch_step} exceeds {len(endpoints)} completed assistant+tool turns"
    return messages[:endpoints[branch_step - 1]], None
