"""Plugin registry for managing algorithm plugin types."""

from __future__ import annotations

import logging
from typing import Type

from app.plugins.base import AlgoPluginBase

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Global registry of algorithm plugin classes.

    Usage:
        PluginRegistry.register(MyPlugin)
        plugin_cls = PluginRegistry.get("my_algo_name")
        instance = plugin_cls()
    """

    _plugins: dict[str, Type[AlgoPluginBase]] = {}

    @classmethod
    def register(cls, plugin_class: Type[AlgoPluginBase]) -> Type[AlgoPluginBase]:
        """Register a plugin class by its .name attribute."""
        name = plugin_class.name
        if name in cls._plugins:
            logger.warning("Plugin '%s' is being re-registered, overwriting.", name)
        cls._plugins[name] = plugin_class
        logger.info("Registered algorithm plugin: %s", name)
        return plugin_class

    @classmethod
    def get(cls, name: str) -> Type[AlgoPluginBase] | None:
        """Get a registered plugin class by name."""
        return cls._plugins.get(name)

    @classmethod
    def list_plugins(cls) -> list[str]:
        """Return all registered plugin names."""
        return list(cls._plugins.keys())

    @classmethod
    def create_instance(cls, name: str) -> AlgoPluginBase | None:
        """Create and return a new instance of the named plugin."""
        plugin_cls = cls.get(name)
        if plugin_cls is None:
            return None
        return plugin_cls()
