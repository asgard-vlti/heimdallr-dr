from importlib import import_module
from inspect import isclass
from pkgutil import iter_modules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataBundler import DataBundler as DataBundler
    from .FTPerformanceBundler import (
        FTPerformanceBundler as FTPerformanceBundler,
        PerformanceFile as PerformanceFile,
    )
    from .FtSettingsBundler import (
        FTSettingsBundler as FTSettingsBundler,
        FringeSettingsFile as FringeSettingsFile,
    )


__all__ = []

for _, module_name, _ in iter_modules(__path__):
    if module_name.startswith("_"):
        continue

    module = import_module(f".{module_name}", __name__)
    for name, value in vars(module).items():
        if isclass(value) and value.__module__ == module.__name__:
            globals()[name] = value
            __all__.append(name)
