# -*- coding: utf-8 -*-

import json
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional, Sequence, Tuple

PRIMARY_CUSTOM_MODEL_API_KEY_ENV_NAME = "CUSTOM_MODEL_API_KEY"
LEGACY_CUSTOM_MODEL_API_KEY_ENV_NAME = "CUSTON_MODEL_API_KEY"
CUSTOM_MODEL_API_KEY_FILE_GLOBS = ("/app/data/*.json", "/data/*.json")
CUSTOM_MODEL_API_KEY_FILE_EXAMPLE = "/app/data/CUSTOM_MODEL_API_KEY.json"
CUSTOM_MODEL_API_KEY_JSON_EXAMPLE = '{"api_keys": ["vck_xxx", "vck_yyy"]}'


@dataclass(frozen=True)
class ResolvedCustomModelApiKeys:
    raw_value: str
    source_type: str
    api_keys: Tuple[str, ...]
    file_path: Optional[str] = None

    @property
    def key_count(self) -> int:
        return len(self.api_keys)

    @property
    def serialized_keys(self) -> str:
        return serialize_custom_model_api_keys(self.api_keys)


def get_custom_model_api_key_raw_value(
    *,
    primary_env_name: str = PRIMARY_CUSTOM_MODEL_API_KEY_ENV_NAME,
    legacy_env_name: str = LEGACY_CUSTOM_MODEL_API_KEY_ENV_NAME,
) -> str:
    return (
        str(os.getenv(primary_env_name, "") or "").strip()
        or str(os.getenv(legacy_env_name, "") or "").strip()
    )


def split_custom_model_inline_api_keys(raw_api_keys: Optional[str]) -> Tuple[str, ...]:
    raw = str(raw_api_keys or "").strip()
    if not raw:
        return ()

    parts = re.split(r"[,\n\r\uFF0C]+", raw)
    normalized = []
    seen = set()

    for part in parts:
        key = part.strip().strip('"').strip("'").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)

    return tuple(normalized)


def serialize_custom_model_api_keys(api_keys: Sequence[str]) -> str:
    return ",".join(str(key).strip() for key in api_keys if str(key).strip())


def persist_custom_model_api_keys_to_file(
    file_path: str, api_keys: Sequence[str]
) -> None:
    normalized_keys = list(split_custom_model_inline_api_keys(serialize_custom_model_api_keys(api_keys)))
    if not normalized_keys:
        raise ValueError("写回 CUSTOM_MODEL_API_KEY 文件时至少需要 1 个有效 key。")

    normalized_file_path = _validate_custom_model_api_key_file_path(file_path)
    directory = os.path.dirname(normalized_file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {"api_keys": normalized_keys}
    with open(normalized_file_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def build_custom_model_api_key_format_note() -> str:
    supported_globs = " 或 ".join(f"`{item}`" for item in CUSTOM_MODEL_API_KEY_FILE_GLOBS)
    return (
        f"文件路径仅支持 {supported_globs}，"
        "其中以 `/data/` 开头的路径会自动映射到 `/app/data/`；"
        f"例如 `{CUSTOM_MODEL_API_KEY_FILE_EXAMPLE}`；"
        f"JSON 格式：`{CUSTOM_MODEL_API_KEY_JSON_EXAMPLE}`。"
    )


def is_custom_model_api_key_file_reference(raw_value: Optional[str]) -> bool:
    raw = str(raw_value or "").strip()
    if not raw:
        return False
    return (
        raw.startswith("/")
        or raw.startswith("\\")
        or raw.lower().endswith(".json")
        or ":\\" in raw
    )


def _build_format_error(detail: str) -> ValueError:
    return ValueError(f"{detail} {build_custom_model_api_key_format_note()}")


def _validate_custom_model_api_key_file_path(raw_value: str) -> str:
    file_path = str(raw_value or "").strip()
    normalized_posix_path = file_path.replace("\\", "/")
    allowed_parent_paths = (PurePosixPath("/app/data"), PurePosixPath("/data"))

    if not normalized_posix_path.lower().endswith(".json"):
        raise _build_format_error(
            "CUSTOM_MODEL_API_KEY 文件路径必须是 `/app/data/*.json` 或 `/data/*.json`。"
        )

    posix_path = PurePosixPath(normalized_posix_path)
    if posix_path.parent not in allowed_parent_paths or not posix_path.name:
        raise _build_format_error(
            "CUSTOM_MODEL_API_KEY 文件路径必须直接位于 `/app/data` 或 `/data` 目录下。"
        )

    if normalized_posix_path.startswith("/data/"):
        return f"/app{normalized_posix_path}"

    return file_path


def _resolve_custom_model_api_keys_from_file(file_path: str) -> ResolvedCustomModelApiKeys:
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except FileNotFoundError as exc:
        raise _build_format_error(
            f"CUSTOM_MODEL_API_KEY 文件不存在：`{file_path}`。"
        ) from exc
    except json.JSONDecodeError as exc:
        raise _build_format_error(
            f"CUSTOM_MODEL_API_KEY 文件不是合法 JSON：`{file_path}`。"
        ) from exc
    except OSError as exc:
        raise _build_format_error(
            f"CUSTOM_MODEL_API_KEY 文件无法读取：`{file_path}`。"
        ) from exc

    if not isinstance(payload, dict):
        raise _build_format_error(
            f"CUSTOM_MODEL_API_KEY 文件顶层必须是对象：`{file_path}`。"
        )

    raw_api_keys = payload.get("api_keys")
    if not isinstance(raw_api_keys, list):
        raise _build_format_error(
            f"CUSTOM_MODEL_API_KEY 文件必须包含数组字段 `api_keys`：`{file_path}`。"
        )

    normalized = []
    seen = set()
    for index, item in enumerate(raw_api_keys):
        if not isinstance(item, str):
            raise _build_format_error(
                f"CUSTOM_MODEL_API_KEY 文件的 `api_keys[{index}]` 必须是非空字符串：`{file_path}`。"
            )
        key = item.strip().strip('"').strip("'").strip()
        if not key:
            raise _build_format_error(
                f"CUSTOM_MODEL_API_KEY 文件的 `api_keys[{index}]` 必须是非空字符串：`{file_path}`。"
            )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)

    if not normalized:
        raise _build_format_error(
            f"CUSTOM_MODEL_API_KEY 文件中至少需要 1 个有效 key：`{file_path}`。"
        )

    return ResolvedCustomModelApiKeys(
        raw_value=file_path,
        source_type="file",
        api_keys=tuple(normalized),
        file_path=file_path,
    )


def resolve_custom_model_api_keys(raw_value: Optional[str]) -> ResolvedCustomModelApiKeys:
    raw = str(raw_value or "").strip()
    if not raw:
        return ResolvedCustomModelApiKeys(
            raw_value="",
            source_type="inline",
            api_keys=(),
            file_path=None,
        )

    if is_custom_model_api_key_file_reference(raw):
        file_path = _validate_custom_model_api_key_file_path(raw)
        return _resolve_custom_model_api_keys_from_file(file_path)

    return ResolvedCustomModelApiKeys(
        raw_value=raw,
        source_type="inline",
        api_keys=split_custom_model_inline_api_keys(raw),
        file_path=None,
    )
