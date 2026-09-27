"""
Dependency graph over ModuleRecords.

Modules never see each other: they depend on ids and receive
DataSpaceReader handles. Pure functions over plain maps.
"""

from __future__ import annotations

from typing import Mapping

from .model import (
    DataSpace,
    DataSpaceReader,
    ModuleRecord,
)


def build_dependency_maps(
    modules: Mapping[str, ModuleRecord],
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
]:
    dependencies = {
        module_id: set(
            record.cls.requires
        )
        for module_id, record
        in modules.items()
    }

    dependents = {
        module_id: set()
        for module_id in modules
    }

    for (
        module_id,
        required_ids,
    ) in dependencies.items():
        for dependency_id in required_ids:
            if (
                dependency_id
                in dependents
            ):
                dependents[
                    dependency_id
                ].add(
                    module_id
                )

    return dependencies, dependents


def bind_instance(
    record: ModuleRecord,
    dataspaces: Mapping[str, DataSpace],
) -> None:
    record.instance.data = record.data

    dependencies: dict[
        str,
        DataSpaceReader,
    ] = {}

    for dependency_id in (
        record.cls.requires
    ):
        space = dataspaces.get(
            dependency_id
        )

        if space is None:
            # Missing dependencies are allowed at bind time.
            # Facade may still start this Module.
            continue

        dependencies[
            dependency_id
        ] = DataSpaceReader(
            space
        )

    record.instance.dependencies = dependencies


def topological_order(
    modules: Mapping[str, ModuleRecord],
    dependencies: Mapping[str, set[str]],
    dependents: Mapping[str, set[str]],
) -> list[str]:
    indegree = {
        module_id: 0
        for module_id in modules
    }

    # An unresolved dependency id contributes no edge: a missing
    # Module cannot be ordered, and Facade reports it separately.
    for (
        module_id,
        required_ids,
    ) in dependencies.items():
        for dependency_id in required_ids:
            if (
                dependency_id
                in indegree
            ):
                indegree[
                    module_id
                ] += 1

    queue = sorted(
        module_id
        for module_id, degree
        in indegree.items()
        if degree == 0
    )

    result: list[str] = []

    while queue:
        current = queue.pop(0)

        result.append(
            current
        )

        for dependent in sorted(
            dependents.get(
                current,
                (),
            )
        ):
            indegree[
                dependent
            ] -= 1

            if (
                indegree[dependent]
                == 0
            ):
                queue.append(
                    dependent
                )

        queue.sort()

    if len(result) != len(
        modules
    ):
        remaining = sorted(
            module_id
            for module_id in modules
            if module_id not in result
        )

        raise RuntimeError(
            "Module dependency cycle "
            "detected: "
            + " -> ".join(
                remaining
            )
        )

    return result
