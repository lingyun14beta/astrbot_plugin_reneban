"""Dashboard Page backend for ReNeBan.

This module exposes the plugin Web APIs consumed by ``pages/reneban``. Every
route is registered under the plugin name prefix, so the Page side only needs a
relative endpoint such as ``overview``.

The handlers never touch the data files directly. They go through
``DatafileManager`` so that the plugin cache, the commit queue and the
``pass > ban`` cleanup rules stay the single source of truth.
"""

from __future__ import annotations

import asyncio
import time as time_module
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request

from .exceptions import PermanentRecordTimeError, TimestrValueError
from .time_utils import timestr_to_int
from .user_manager import UmoDataModel, UserDataList, UserDataModel

if TYPE_CHECKING:
    from .main import ReNeBan

PLUGIN_NAME = "astrbot_plugin_reneban"

# Page name of the bundled dashboard page.
PAGE_NAME = "reneban"

# Longest reason accepted from the Page, in characters.
MAX_REASON_LENGTH = 200

# Longest user id / UMO accepted from the Page, in characters.
MAX_ID_LENGTH = 128

# Upper bound for a single time value, in seconds (10 years).
MAX_DURATION_SECONDS = 10 * 365 * 86400

# ``unified_msg_origin`` looks like ``platform:message_type:session_id``.
_UMO_SEPARATOR = ":"

# Cap on how many sessions get their display name resolved from the database.
_MAX_NAME_LOOKUPS = 60

# Reason values that the plugin treats as "no reason".
_NO_REASON = {"无理由", "None", "NULL"}


class PageValidationError(ValueError):
    """Raised when a Page supplied field fails validation."""


class PageNotFoundError(LookupError):
    """Raised when the addressed record does not exist (any more)."""


def _umo_parts(umo: str) -> tuple[str, str, str]:
    """Split a ``unified_msg_origin`` into its three components.

    Args:
        umo: Unified message origin, e.g. ``napcat:GroupMessage:123456``.

    Returns:
        A ``(platform, message_type, session_id)`` tuple. Missing parts are
        returned as empty strings.
    """
    parts = umo.split(_UMO_SEPARATOR, 2)
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def parse_duration(raw: Any) -> int:
    """Validate a Page supplied time expression.

    The Page speaks the same syntax as the chat commands, for example ``1d2h``,
    ``30m`` or ``0`` for a permanent record.

    Args:
        raw: Raw value from the request payload.

    Returns:
        The duration in seconds, ``0`` meaning permanent.

    Raises:
        PageValidationError: If the expression is invalid or out of range.
    """
    if raw is None or raw == "":
        return 0
    if isinstance(raw, bool):
        raise PageValidationError("duration 必须是数字或时间字符串")
    if isinstance(raw, (int, float)):
        seconds = int(raw)
    else:
        text = str(raw).strip()
        if not text:
            return 0
        try:
            seconds = timestr_to_int(text)
        except TimestrValueError as exc:
            raise PageValidationError(
                f"时间格式错误：{text}（示例：1d2h30m、30m、0）"
            ) from exc
    if seconds < 0:
        raise PageValidationError("时间不能为负数")
    if seconds > MAX_DURATION_SECONDS:
        raise PageValidationError("时间过长，最多 10 年")
    return seconds


def parse_text(raw: Any, field: str, max_length: int = MAX_ID_LENGTH) -> str:
    """Validate a required string field.

    Args:
        raw: Raw value from the request payload.
        field: Human readable field name used in the error message.
        max_length: Longest accepted value.

    Returns:
        The trimmed value.

    Raises:
        PageValidationError: If the value is missing, not a string or too long.
    """
    if not isinstance(raw, str):
        raise PageValidationError(f"{field} 必须是字符串")
    value = raw.strip()
    if not value:
        raise PageValidationError(f"{field} 不能为空")
    if len(value) > max_length:
        raise PageValidationError(f"{field} 过长（最多 {max_length} 字符）")
    return value


def parse_reason(raw: Any) -> str | None:
    """Validate the optional reason field.

    Args:
        raw: Raw value from the request payload.

    Returns:
        The trimmed reason, or ``None`` when the record has no reason.

    Raises:
        PageValidationError: If the reason is not a string or too long.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise PageValidationError("reason 必须是字符串")
    value = raw.strip()
    if not value or value in _NO_REASON:
        return None
    if len(value) > MAX_REASON_LENGTH:
        raise PageValidationError(f"理由过长（最多 {MAX_REASON_LENGTH} 字符）")
    return value


def _expiry_buckets(
    records: list[dict[str, Any]],
    session_records: list[dict[str, Any]],
    now: int,
) -> list[dict[str, Any]]:
    """Bucket every temporary record by its remaining lifetime.

    Permanent records have no expiry, so they are reported separately by
    ``stats.permanent`` instead of being forced into a bucket.

    Args:
        records: User scoped records.
        session_records: Whole-session records.
        now: Current unix timestamp.

    Returns:
        A list of ``{"key", "count", "bans", "passes"}`` dicts in ascending
        order of remaining time. Buckets with no record are included with a
        zero count so the chart keeps a stable shape.
    """
    edges = (
        ("lt1h", 3600),
        ("lt6h", 6 * 3600),
        ("lt24h", 86400),
        ("lt3d", 3 * 86400),
        ("lt7d", 7 * 86400),
    )
    buckets: list[dict[str, Any]] = [
        {"key": key, "count": 0, "bans": 0, "passes": 0} for key, _ in edges
    ]
    buckets.append({"key": "gt7d", "count": 0, "bans": 0, "passes": 0})

    for item in list(records) + list(session_records):
        if item["time"] == 0:
            continue
        remaining = item["time"] - now
        if remaining <= 0:
            # Already expired entries are cleaned up by the data manager, but a
            # stale cache can still surface one; count it in the nearest bucket.
            index = 0
        else:
            index = len(edges)
            for position, (_, limit) in enumerate(edges):
                if remaining <= limit:
                    index = position
                    break
        buckets[index]["count"] += 1
        buckets[index]["bans" if item["kind"] == "ban" else "passes"] += 1
    return buckets


def _build_insights(
    records: list[dict[str, Any]],
    session_records: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Derive a few ranking facts from the records.

    Everything here is read straight off the stored records, so the Page can
    present rankings without inventing history the plugin does not keep.

    Args:
        records: User scoped records, already normalised for the Page.
        session_records: Whole-session records.
        states: Per-session ban/pass counters from ``PageApi._collect``.

    Returns:
        A dict with ``nextExpiry``, ``topUser`` and ``topSession`` keys. Each
        may be ``None`` when there is nothing to rank.
    """
    temporary = [
        item
        for item in list(records) + list(session_records)
        if item["time"] != 0 and item["remaining"] > 0
    ]
    next_expiry = min(temporary, key=lambda item: item["time"]) if temporary else None

    user_counts: dict[str, dict[str, Any]] = {}
    for item in records:
        entry = user_counts.setdefault(
            item["id"], {"id": item["id"], "count": 0, "bans": 0, "passes": 0}
        )
        entry["count"] += 1
        entry["bans" if item["kind"] == "ban" else "passes"] += 1
    top_user = (
        max(user_counts.values(), key=lambda entry: entry["count"])
        if user_counts
        else None
    )

    top_session = None
    for umo, state in states.items():
        total = state["bans"] + state["passes"]
        if total and (top_session is None or total > top_session["total"]):
            top_session = {"umo": umo, "total": total, **state}

    return {
        "nextExpiry": next_expiry,
        "topUser": top_user,
        "topSession": top_session,
    }


class PageApi:
    """Implements the Web APIs used by the bundled dashboard Page."""

    def __init__(self, plugin: "ReNeBan") -> None:
        """Bind the API layer to a live plugin instance.

        Args:
            plugin: The running ``ReNeBan`` star instance.
        """
        self.plugin = plugin
        self._session_name_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    async def _resolve_session_name(self, umo: str) -> str:
        """Resolve the human readable name of one session.

        Falls back to the raw session id when the session has no conversation
        record yet, which is the common case for sessions the bot only replied
        to without an LLM conversation.

        Args:
            umo: Unified message origin of the session.

        Returns:
            The conversation title, the raw session id, or the UMO itself.
        """
        cached = self._session_name_cache.get(umo)
        if cached:
            return cached

        _, _, session_id = _umo_parts(umo)
        name = session_id or umo
        try:
            manager = self.plugin.context.conversation_manager
            conv_id = await manager.get_curr_conversation_id(umo)
            if conv_id:
                conversation = await manager.get_conversation(umo, conv_id)
                title = getattr(conversation, "title", None)
                if isinstance(title, str) and title.strip():
                    name = title.strip()
        except Exception as exc:  # noqa: BLE001 - naming is best effort only
            logger.debug(f"ReNeBan Page: 解析会话名失败({umo}): {exc}")

        self._session_name_cache[umo] = name
        return name

    async def _resolve_session_names(self, umos: list[str]) -> dict[str, str]:
        """Resolve display names for a batch of sessions.

        Args:
            umos: Unified message origins to resolve.

        Returns:
            A mapping of UMO to display name.
        """
        names: dict[str, str] = {}
        for index, umo in enumerate(umos):
            if index >= _MAX_NAME_LOOKUPS:
                # Keep the response fast on instances with many sessions.
                _, _, session_id = _umo_parts(umo)
                names[umo] = session_id or umo
                continue
            names[umo] = await self._resolve_session_name(umo)
        return names

    @staticmethod
    def _record(
        *,
        identifier: str,
        time_value: int,
        reason: str | None,
        scope: str,
        kind: str,
        umo: str | None = None,
        target: str = "user",
    ) -> dict[str, Any]:
        """Convert one stored record into a JSON friendly payload.

        Args:
            identifier: User id for user records, UMO for session records.
            time_value: Absolute expiry timestamp in seconds, ``0`` is permanent.
            reason: Stored reason, ``None`` when the record has no reason.
            scope: ``session`` or ``global``.
            kind: ``ban`` or ``pass``.
            umo: Owning session UMO for session scoped user records.
            target: ``user`` for per-user records, ``session`` for whole session
                records. The Page needs it to address the right data table.

        Returns:
            A dict with the remaining lifetime precomputed for the frontend.
        """
        now = int(time_module.time())
        platform, message_type, session_id = _umo_parts(umo) if umo else ("", "", "")
        return {
            "id": identifier,
            "time": time_value,
            "remaining": 0 if time_value == 0 else max(0, time_value - now),
            "permanent": time_value == 0,
            "expired": time_value != 0 and time_value <= now,
            "reason": reason,
            "scope": scope,
            "kind": kind,
            "target": target,
            "umo": umo,
            "platform": platform,
            "message_type": message_type,
            "session_id": session_id,
            "session_name": session_id or umo or identifier,
        }

    def _collect(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalise the plugin data cache into Page friendly structures.

        Args:
            data: Raw payload from ``DatafileManager.get_data()``.

        Returns:
            A dict with ``records``, ``session_records``, ``sessions``,
            ``stats`` and ``enabled`` keys.
        """
        records: list[dict[str, Any]] = []
        session_records: list[dict[str, Any]] = []
        session_map: dict[str, dict[str, Any]] = {}

        def track(umo: str) -> dict[str, Any]:
            entry = session_map.get(umo)
            if entry is None:
                platform, message_type, session_id = _umo_parts(umo)
                entry = {
                    "umo": umo,
                    "platform": platform,
                    "message_type": message_type,
                    "session_id": session_id,
                    "session_name": session_id or umo,
                    "bans": 0,
                    "passes": 0,
                }
                session_map[umo] = entry
            return entry

        # Session scoped user records: {"ban"|"pass": {umo: UserDataList}}
        for kind in ("ban", "pass"):
            for umo, model_list in (data.get(kind) or {}).items():
                entry = track(umo)
                counter = "bans" if kind == "ban" else "passes"
                for item in model_list:
                    records.append(
                        self._record(
                            identifier=item.uid,
                            time_value=item.time,
                            reason=item.reason,
                            scope="session",
                            kind=kind,
                            umo=umo,
                        )
                    )
                    entry[counter] += 1

        # Global user records: {"banall"|"passall": UserDataList}
        for key, kind in (("banall", "ban"), ("passall", "pass")):
            for item in data.get(key) or []:
                records.append(
                    self._record(
                        identifier=item.uid,
                        time_value=item.time,
                        reason=item.reason,
                        scope="global",
                        kind=kind,
                    )
                )

        # Session level records: {"umoban"|"umopass": UmoDataList}
        for key, kind in (("umoban", "ban"), ("umopass", "pass")):
            for item in data.get(key) or []:
                track(item.umo)
                session_records.append(
                    self._record(
                        identifier=item.umo,
                        time_value=item.time,
                        reason=item.reason,
                        scope="session",
                        kind=kind,
                        umo=item.umo,
                        target="session",
                    )
                )

        now = int(time_module.time())
        permanent = sum(1 for item in records if item["time"] == 0)
        passed = sum(1 for item in records if item["kind"] == "pass")
        return {
            "enabled": bool(self.plugin.enable),
            "records": records,
            "session_records": session_records,
            "sessions": sorted(session_map.values(), key=lambda item: item["umo"]),
            "stats": {
                "total": len(records),
                "bans": len(records) - passed,
                "passes": passed,
                "permanent": permanent,
                "temporary": len(records) - permanent,
                "expiring_soon": sum(
                    1
                    for item in records
                    if item["time"] != 0 and 0 < item["time"] - now <= 86400
                ),
                "sessions": len(session_map),
                "users": len({item["id"] for item in records}),
                "session_bans": sum(
                    1 for item in session_records if item["kind"] == "ban"
                ),
                "session_passes": sum(
                    1 for item in session_records if item["kind"] == "pass"
                ),
            },
            # Expiry distribution over both user and whole-session records,
            # bucketed by remaining lifetime. Drives the histogram chart.
            "expiry_buckets": _expiry_buckets(records, session_records, now),
        }

    async def _build_overview(self) -> dict[str, Any]:
        """Read all ban data and build the overview payload.

        Returns:
            The full overview payload with resolved session names.
        """
        # The data manager takes a lock and rewrites several files, so keep it
        # off the event loop.
        tables = await asyncio.to_thread(self.plugin.data_manager.get_data)
        payload = self._collect(tables)
        umos = sorted(
            {item["umo"] for item in payload["records"] if item["umo"]}
            | {item["umo"] for item in payload["session_records"] if item["umo"]}
        )
        names = await self._resolve_session_names(umos)
        for entry in payload["sessions"]:
            entry["session_name"] = names.get(entry["umo"], entry["session_name"])
        for bucket in ("records", "session_records"):
            for item in payload[bucket]:
                if item["umo"]:
                    item["session_name"] = names.get(item["umo"], item["session_name"])
        payload["insights"] = _build_insights(
            payload["records"],
            payload["session_records"],
            {entry["umo"]: entry for entry in payload["sessions"]},
        )
        top_session = payload["insights"].get("topSession")
        if top_session and top_session.get("umo"):
            top_session["session_name"] = names.get(
                top_session["umo"], top_session["umo"]
            )
        return payload

    async def _load_tables(self, keys: list[str]) -> dict[str, Any]:
        """Read several data tables from the plugin data manager.

        Args:
            keys: Data names such as ``ban`` or ``umoban``.

        Returns:
            A mapping of data name to its mutable table.
        """
        return await asyncio.to_thread(self.plugin.data_manager.get_data, keys)

    def _shift(self, model: Any, delta: int, reason: str | None) -> None:
        """Apply a time delta to a record.

        Args:
            model: A ``BaseDataModel`` instance.
            delta: Seconds to add; negative values subtract.
            reason: Optional new reason.

        Raises:
            PermanentRecordTimeError: If the record is permanent and cannot be
                shifted.
            TimeNegativeError: If the resulting time would be negative.
        """
        if delta > 0:
            model.add_time(delta, reason)
        else:
            model.subtract_time(-delta, reason)

    # ------------------------------------------------------------------
    # read endpoints
    # ------------------------------------------------------------------

    async def overview(self):
        """Return the whole ban state for the dashboard Page.

        Returns:
            A JSON response with records, sessions, statistics and the current
            runtime switch state.
        """
        try:
            payload = await self._build_overview()
        except Exception as exc:  # noqa: BLE001 - surface a readable error
            logger.error(f"ReNeBan Page: 读取数据失败: {exc}")
            return error_response(f"读取数据失败：{exc}", status_code=500)
        return json_response(payload)

    # ------------------------------------------------------------------
    # write endpoints
    # ------------------------------------------------------------------

    async def toggle(self):
        """Enable or disable the ban filter at runtime.

        The switch mirrors ``/ban-enable`` and ``/ban-disable``: it lives on the
        plugin instance only and is lost on restart.

        Returns:
            A JSON response with the resulting state.
        """
        payload = await request.json(default={}) or {}
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            return error_response("enabled 必须是布尔值", status_code=400)
        self.plugin.enable = enabled
        logger.info(f"ReNeBan Page: 通过 Dashboard 将禁用功能设为 {enabled}")
        return json_response({"enabled": enabled})

    async def ban(self):
        """Create a ban record or extend the remaining time of an existing one.

        Mirrors ``/ban``, ``/ban-all`` and ``/ban-umo``. An empty or zero
        duration creates a permanent ban.

        Returns:
            A JSON response describing the applied change.
        """
        return await self._apply_action("ban")

    async def grant_pass(self):
        """Create a temporary exemption or extend an existing one.

        Mirrors ``/pass``, ``/pass-all`` and ``/pass-umo``. An empty or zero
        duration creates a permanent exemption.

        Returns:
            A JSON response describing the applied change.
        """
        return await self._apply_action("pass")

    async def delete(self):
        """Remove one stored record.

        Returns:
            A JSON response describing the deleted record.
        """
        try:
            payload = await request.json(default={}) or {}
            identifier = parse_text(payload.get("id"), "ID")
            scope, kind, target, umo = _parse_scope(payload)
            keys = _data_keys(kind, scope, target)

            tables = await self._load_tables(keys)
            if target == "session":
                model_list = tables[keys[0]]
                if not model_list.remove_by_id(umo):
                    raise PageNotFoundError("未找到该会话记录，可能已过期")
            elif scope == "global":
                model_list = tables[keys[0]]
                if not model_list.remove_by_id(identifier):
                    raise PageNotFoundError("未找到该记录，可能已过期")
            else:
                table = tables[keys[0]]
                model_list = table.get(umo)
                if model_list is None or not model_list.remove_by_id(identifier):
                    raise PageNotFoundError("未找到该记录，可能已过期")

            self.plugin.data_manager.write_data(keys[0], tables[keys[0]])
        except PageValidationError as exc:
            return error_response(str(exc), status_code=400)
        except PageNotFoundError as exc:
            return error_response(str(exc), status_code=404)
        except Exception as exc:  # noqa: BLE001 - surface a readable error
            logger.error(f"ReNeBan Page: 删除记录失败: {exc}")
            return error_response(f"删除失败：{exc}", status_code=500)

        return json_response(
            {"deleted": {"id": identifier, "umo": umo, "scope": scope, "kind": kind}}
        )

    async def shift_time(self):
        """Add to or subtract from the remaining time of an existing record.

        Mirrors the ``dec-*`` command family. A negative ``delta`` removes time,
        a positive ``delta`` extends the record.

        Returns:
            A JSON response describing the resulting record.
        """
        try:
            payload = await request.json(default={}) or {}
            identifier = parse_text(payload.get("id"), "ID")
            raw_delta = payload.get("delta", 0)
            if isinstance(raw_delta, bool) or not isinstance(raw_delta, (int, float)):
                raise PageValidationError("delta 必须是数字（秒）")
            delta = int(raw_delta)
            if delta == 0:
                raise PageValidationError("delta 不能为 0，如需删除请使用删除操作")
            if abs(delta) > MAX_DURATION_SECONDS:
                raise PageValidationError("调整幅度过大，最多 10 年")

            scope, kind, target, umo = _parse_scope(payload)
            reason = parse_reason(payload.get("reason"))
            keys = _data_keys(kind, scope, target)

            tables = await self._load_tables(keys)
            if target == "session":
                model = tables[keys[0]].find_by_id(umo, no_copy=True)
            elif scope == "global":
                model = tables[keys[0]].find_by_id(identifier, no_copy=True)
            else:
                bucket = tables[keys[0]].get(umo)
                model = (
                    bucket.find_by_id(identifier, no_copy=True)
                    if bucket is not None
                    else None
                )
            if model is None:
                raise PageNotFoundError("未找到该记录，可能已过期")

            self._shift(model, delta, reason)
            self.plugin.data_manager.write_data(keys[0], tables[keys[0]])
            remaining = model.time
        except PageValidationError as exc:
            return error_response(str(exc), status_code=400)
        except PageNotFoundError as exc:
            return error_response(str(exc), status_code=404)
        except PermanentRecordTimeError as exc:
            return error_response(_time_change_message(exc), status_code=400)
        except Exception as exc:  # noqa: BLE001 - surface a readable error
            logger.error(f"ReNeBan Page: 调整时长失败: {exc}")
            return error_response(f"调整失败：{exc}", status_code=500)

        return json_response(
            {
                "id": identifier,
                "umo": umo,
                "scope": scope,
                "kind": kind,
                "target": target,
                "time": remaining,
                "remaining": 0
                if remaining == 0
                else max(0, remaining - int(time_module.time())),
                "delta": delta,
            }
        )

    async def reset_user(self):
        """Delete every record of one user across all scopes.

        Mirrors ``/ban-reset``.

        Returns:
            A JSON response with the cleared user id and removed record count.
        """
        try:
            payload = await request.json(default={}) or {}
            identifier = parse_text(payload.get("id"), "用户 ID")
            tables = await self._load_tables(["ban", "pass", "banall", "passall"])
            removed = 0
            for key in ("ban", "pass"):
                for umo in list(tables[key].keys()):
                    model_list = tables[key][umo]
                    while model_list.remove_by_id(identifier):
                        removed += 1
            for key in ("banall", "passall"):
                while tables[key].remove_by_id(identifier):
                    removed += 1
            self.plugin.data_manager.write_data(
                list(tables.keys()), list(tables.values())
            )
        except PageValidationError as exc:
            return error_response(str(exc), status_code=400)
        except Exception as exc:  # noqa: BLE001 - surface a readable error
            logger.error(f"ReNeBan Page: 清除用户记录失败: {exc}")
            return error_response(f"清除失败：{exc}", status_code=500)
        return json_response({"removed": removed, "id": identifier})

    async def reset_session(self):
        """Delete every record attached to one session.

        Mirrors ``/ban-reset-umo`` and additionally clears the session scoped
        user records of that session.

        Returns:
            A JSON response with the cleared UMO and removed record count.
        """
        try:
            payload = await request.json(default={}) or {}
            umo = parse_text(payload.get("umo"), "UMO")
            if _UMO_SEPARATOR not in umo:
                raise PageValidationError(f"UMO 不合法：{umo}")
            tables = await self._load_tables(["ban", "pass", "umoban", "umopass"])
            removed = 0
            for key in ("ban", "pass", "umoban", "umopass"):
                model_list = tables[key]
                if isinstance(model_list, dict):
                    bucket = model_list.pop(umo, None)
                    if bucket is not None:
                        removed += len(bucket)
                else:
                    while model_list.remove_by_id(umo):
                        removed += 1
            self.plugin.data_manager.write_data(
                list(tables.keys()), list(tables.values())
            )
        except PageValidationError as exc:
            return error_response(str(exc), status_code=400)
        except Exception as exc:  # noqa: BLE001 - surface a readable error
            logger.error(f"ReNeBan Page: 清除会话记录失败: {exc}")
            return error_response(f"清除失败：{exc}", status_code=500)
        return json_response({"removed": removed, "umo": umo})

    @staticmethod
    def _require_target_ban(
        tables: dict[str, Any],
        scope: str,
        target: str,
        identifier: str,
        umo: str | None,
    ) -> None:
        """Reject an exemption that has nothing to exempt the target from.

        The data manager prunes ``pass`` records without a matching ``ban``, so
        creating one would silently disappear right after the write. Checking up
        front turns that into a readable message instead.

        Args:
            tables: Tables already loaded by the caller.
            scope: ``session`` or ``global``.
            target: ``user`` or ``session``.
            identifier: User id for user records, UMO for session records.
            umo: Session UMO for session scoped user records.

        Raises:
            PageValidationError: If no matching ban record exists.
        """
        if target == "session":
            banned = any(item.umo == identifier for item in tables["umoban"])
            if not banned:
                raise PageValidationError(
                    "该会话当前没有被禁用，无需解限。请先禁用该会话，或改用会话内的用户操作。"
                )
            return

        banned = tables["banall"].find_by_id(identifier) is not None
        if not banned and scope == "session" and umo:
            bucket = tables["ban"].get(umo)
            banned = bucket is not None and bucket.find_by_id(identifier) is not None
        if not banned:
            raise PageValidationError(
                "该用户当前没有被禁用，无需解限。请先创建禁用记录，或确认作用范围是否正确。"
            )

    async def _apply_action(self, event: str):
        """Shared implementation of the ban and pass endpoints.

        Args:
            event: ``ban`` or ``pass``; it is sent by the caller and not read
                from the request body, so the endpoint decides the record kind.

        Returns:
            A JSON response describing the applied change.
        """
        kind = event
        try:
            payload = await request.json(default={}) or {}
            identifier = parse_text(payload.get("id"), "ID")
            duration = parse_duration(payload.get("duration", 0))
            reason = parse_reason(payload.get("reason"))
            scope, kind, target, umo = _parse_scope(payload, kind=event)
            keys = _data_keys(kind, scope, target)

            expire_at = 0 if duration == 0 else int(time_module.time()) + duration
            action = "created"
            # Extra tables are needed to validate that an exemption is meaningful.
            wanted = list(dict.fromkeys([keys[0], "ban", "banall", "umoban"]))
            tables = await self._load_tables(wanted)
            data = tables[keys[0]]

            if kind == "pass":
                self._require_target_ban(
                    tables,
                    scope,
                    target,
                    umo if target == "session" else identifier,
                    umo,
                )

            if target == "session":
                if data.add_time_to_data(umo, duration, reason):
                    action = "extended"
                else:
                    data.append(UmoDataModel(umo=umo, time=expire_at, reason=reason))
                model = data.find_by_id(umo, no_copy=True)
            elif scope == "global":
                if data.add_time_to_data(identifier, duration, reason):
                    action = "extended"
                else:
                    data.append(
                        UserDataModel(uid=identifier, time=expire_at, reason=reason)
                    )
                model = data.find_by_id(identifier, no_copy=True)
            else:
                bucket = data.get(umo)
                if bucket is None:
                    bucket = UserDataList()
                    data[umo] = bucket
                if bucket.add_time_to_data(identifier, duration, reason):
                    action = "extended"
                else:
                    bucket.append(
                        UserDataModel(uid=identifier, time=expire_at, reason=reason)
                    )
                model = bucket.find_by_id(identifier, no_copy=True)

            expire_at = model.time if model is not None else expire_at
            self.plugin.data_manager.write_data(keys[0], data)
        except PageValidationError as exc:
            return error_response(str(exc), status_code=400)
        except PermanentRecordTimeError as exc:
            return error_response(_time_change_message(exc), status_code=400)
        except Exception as exc:  # noqa: BLE001 - surface a readable error
            logger.error(f"ReNeBan Page: 写入 {event} 记录失败: {exc}")
            return error_response(f"操作失败：{exc}", status_code=500)

        return json_response(
            {
                "id": identifier,
                "umo": umo,
                "scope": scope,
                "kind": kind,
                "target": target,
                "action": action,
                "time": expire_at,
                "permanent": expire_at == 0,
                "reason": reason,
            }
        )


def _parse_scope(
    payload: dict[str, Any], kind: str | None = None
) -> tuple[str, str, str, str | None]:
    """Validate the scope related fields shared by the write endpoints.

    Args:
        payload: Parsed JSON request body.
        kind: Record kind fixed by the calling endpoint, or ``None`` to read it
            from the payload.

    Returns:
        A ``(scope, kind, target, umo)`` tuple. ``umo`` is ``None`` for the
        global scope.

    Raises:
        PageValidationError: If any field is missing or inconsistent.
    """
    resolved_kind = kind or payload.get("kind")
    if resolved_kind not in ("ban", "pass"):
        raise PageValidationError("kind 必须是 ban 或 pass")

    scope = payload.get("scope")
    if scope not in ("session", "global"):
        raise PageValidationError("scope 必须是 session 或 global")

    target = payload.get("target") or "user"
    if target not in ("user", "session"):
        raise PageValidationError("target 必须是 user 或 session")
    if scope == "global" and target == "session":
        # The data model has no global session record.
        raise PageValidationError("会话级记录不支持全局范围")

    umo: str | None = None
    if scope == "session":
        # Session level records carry their session in ``id``, user records in
        # ``umo``. Accept both so either endpoint shape works.
        raw = payload.get("umo") or (payload.get("id") if target == "session" else None)
        umo = parse_text(raw, "UMO")
        if _UMO_SEPARATOR not in umo:
            raise PageValidationError(f"UMO 不合法：{umo}")
    return scope, resolved_kind, target, umo


def _data_keys(kind: str, scope: str, target: str) -> list[str]:
    """Map a record description to the data table name it lives in.

    Args:
        kind: ``ban`` or ``pass``.
        scope: ``session`` or ``global``.
        target: ``user`` or ``session``.

    Returns:
        A single element list holding the data table name.
    """
    if target == "session":
        return ["umoban" if kind == "ban" else "umopass"]
    if scope == "global":
        return [f"{kind}all"]
    return ["ban" if kind == "ban" else "pass"]


def _time_change_message(exc: Exception) -> str:
    """Translate a data model time error into a user facing message.

    Args:
        exc: Exception raised by ``add_time`` or ``subtract_time``.

    Returns:
        A readable Chinese message.
    """
    text = str(exc)
    if "permanent record" in text:
        return "该记录为永久时限，不支持此操作（如需删除请直接删除记录）"
    if "non-negative" in text:
        return "时间不能为负数"
    return f"调整失败：{text}"


__all__ = [
    "PAGE_NAME",
    "PLUGIN_NAME",
    "PageApi",
    "PageNotFoundError",
    "PageValidationError",
]
