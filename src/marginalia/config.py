"""Configuration loading and scope profiles.

A profile is a named scope. The corpus this was built against wants at least two: the
narrow set of documents the operator actually wrote, and the wider working library they
collected. Those answer different questions and deserve different numbers, so they are
configuration rather than a flag someone has to remember.

Each profile gets its own index directory. Two profiles sharing one index would give
each other's results, which is the kind of bug that looks like a retrieval-quality
problem for a long time before anyone checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .documents import TrustPolicy
from .scope import ScopePolicy

DEFAULT_CONFIG_NAMES = ("config.local.yaml", "config.local.yml", "config.yaml", "config.yml")

# Keys whose lists accumulate across defaults and profile rather than being replaced.
# Exclusions are safety rules: a profile should be able to add one without silently
# dropping the shared ones.
ADDITIVE_KEYS = frozenset({"exclude"})

IMPLICIT_PROFILE = "default"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    """A parsed configuration file."""

    path: Path
    data: Mapping[str, Any]

    # ---- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None, *, start: Path | None = None) -> Config:
        """Load an explicit path, or discover one by name in `start` (default: cwd).

        The discovery order prefers `config.local.*` over `config.*` so a machine's real
        configuration wins over any committed sample without needing a flag.
        """
        if path is not None:
            p = Path(path).expanduser()
            if not p.is_file():
                raise ConfigError(f"config file not found: {p}")
            return cls._read(p)

        base = start or Path.cwd()
        for name in DEFAULT_CONFIG_NAMES:
            candidate = base / name
            if candidate.is_file():
                return cls._read(candidate)
        raise ConfigError(
            f"no configuration found in {base}. Looked for: {', '.join(DEFAULT_CONFIG_NAMES)}. "
            "Copy config.example.yaml to config.local.yaml and edit it."
        )

    @classmethod
    def _read(cls, p: Path) -> Config:
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{p}: invalid YAML: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{p}: top level must be a mapping")
        return cls(path=p.resolve(), data=raw)

    # ---- profiles -----------------------------------------------------------

    @property
    def _scope_section(self) -> Mapping[str, Any]:
        section = self.data.get("scope") or {}
        if not isinstance(section, Mapping):
            raise ConfigError(f"{self.path}: 'scope' must be a mapping")
        return section

    @property
    def profile_names(self) -> tuple[str, ...]:
        """Declared profiles, or the single implicit one for a flat config."""
        profiles = self._scope_section.get("profiles")
        if isinstance(profiles, Mapping) and profiles:
            return tuple(profiles.keys())
        if self._scope_section.get("roots"):
            return (IMPLICIT_PROFILE,)
        return ()

    @property
    def default_profile(self) -> str:
        """The profile used when none is named."""
        declared = self._scope_section.get("default_profile")
        names = self.profile_names
        if not names:
            raise ConfigError(
                f"{self.path}: no scope profiles and no 'scope.roots'. Nothing can be indexed."
            )
        if declared is not None:
            if declared not in names:
                raise ConfigError(
                    f"{self.path}: default_profile '{declared}' is not defined. "
                    f"Available: {', '.join(names)}"
                )
            return str(declared)
        return names[0]

    def _profile_data(self, name: str) -> Mapping[str, Any]:
        section = self._scope_section
        profiles = section.get("profiles")

        if isinstance(profiles, Mapping) and profiles:
            if name not in profiles:
                raise ConfigError(
                    f"unknown profile '{name}'. Available: {', '.join(self.profile_names)}"
                )
            entry = profiles[name]
            if not isinstance(entry, Mapping):
                raise ConfigError(f"{self.path}: profile '{name}' must be a mapping")
            defaults = section.get("defaults") or {}
            if not isinstance(defaults, Mapping):
                raise ConfigError(f"{self.path}: 'scope.defaults' must be a mapping")
            return self._merge(defaults, entry)

        # Flat config: the scope section *is* the profile.
        if name != IMPLICIT_PROFILE:
            raise ConfigError(
                f"unknown profile '{name}': this config declares no profiles, only 'scope.roots'"
            )
        reserved = ("profiles", "defaults", "default_profile")
        return {k: v for k, v in section.items() if k not in reserved}

    @staticmethod
    def _merge(defaults: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
        """Profile values win, except additive list keys which accumulate."""
        merged: dict[str, Any] = dict(defaults)
        for key, value in profile.items():
            if key in ADDITIVE_KEYS:
                base = defaults.get(key) or []
                base_list = [base] if isinstance(base, str) else list(base)
                add_list = [value] if isinstance(value, str) else list(value)
                merged[key] = list(dict.fromkeys([*base_list, *add_list]))
            else:
                merged[key] = value
        return merged

    def scope(self, profile: str | None = None) -> ScopePolicy:
        """Build the policy for `profile`, or the default one."""
        name = profile or self.default_profile
        return ScopePolicy.from_config(self._profile_data(name), base=self.path.parent)

    # ---- index ---------------------------------------------------------------

    @property
    def index_root(self) -> Path:
        section = self.data.get("index") or {}
        raw = section.get("path") if isinstance(section, Mapping) else None
        base = Path(str(raw)).expanduser() if raw else Path.home() / ".local/share/marginalia"
        if not base.is_absolute():
            base = (self.path.parent / base).resolve()
        return base

    def index_dir(self, profile: str | None = None) -> Path:
        """Per-profile index directory, so profiles cannot return each other's results."""
        return self.index_root / (profile or self.default_profile)

    # ---- trust ----------------------------------------------------------------

    def trust_policy(self, profile: str | None = None) -> TrustPolicy:
        """Trust rules for a profile, falling back to a top-level `trust` block.

        A profile may override the whole block, which is the point: the authored corpus
        and the wider library have different provenance by definition.
        """
        name = profile or self.default_profile
        section = self._scope_section
        profiles = section.get("profiles")
        if isinstance(profiles, Mapping) and name in profiles:
            entry = profiles[name]
            if isinstance(entry, Mapping) and isinstance(entry.get("trust"), Mapping):
                return TrustPolicy.from_config(entry["trust"])
        top = self.data.get("trust")
        return TrustPolicy.from_config(top if isinstance(top, Mapping) else None)

    # ---- generation ----------------------------------------------------------

    @property
    def generation_provider(self) -> str:
        section = self.data.get("generation") or {}
        if not isinstance(section, Mapping):
            return "none"
        return str(section.get("provider", "none")).lower()
