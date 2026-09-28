"""
Dependency graph over ModuleRecords.

Modules never see each other: they depend on ids and receive
DataSpaceReader handles. Pure functions over plain maps.
"""

from __future__ import annotations

from loguru import logger

from typing import Any, Mapping

from .model import (
    DataSpaceReader,
)



def build_dependency_maps(
    modules: Mapping[str, Any],
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
    record,
    dataspaces: Mapping[str, "DataSpace"],
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


def attach_dependency(
    record,
    dependency_id: str,
    *,
    dataspaces: Mapping[str, "DataSpace"],
) -> None:
    """
    Attach one dependency edge to an already-bound instance.

    Targeted single-edge operation: only the given record may be
    touched, and only when it declares the dependency and does
    not already hold a reader for it. Existing readers are left
    untouched.
    """
    if dependency_id not in record.cls.requires:
        return

    space = dataspaces.get(
        dependency_id
    )

    if space is None:
        return

    dependencies = getattr(
        record.instance,
        "dependencies",
        None,
    )

    if dependencies is None:
        return

    if dependency_id in dependencies:
        return

    dependencies[
        dependency_id
    ] = DataSpaceReader(
        space
    )


def detach_dependency(
    record,
    dependency_id: str,
) -> None:
    """
    Detach one dependency edge from an already-bound instance.

    Targeted single-edge operation: only the given record is
    touched. An absent reader is ignored.
    """
    dependencies = getattr(
        record.instance,
        "dependencies",
        None,
    )

    if dependencies is None:
        return

    dependencies.pop(
        dependency_id,
        None,
    )


def topological_order(
    modules: Mapping[str, Any],
    dependencies: Mapping[str, set[str]],
    dependents: Mapping[str, set[str]],
) -> list[str]:
    indegree = {
        module_id: 0
        for module_id in modules
    }

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
            else:
                logger.warning(f'''Module {module_id} requires missing Module {dependency_id}''')

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


def validate_addition(
    modules: Mapping[str, Any],
    record: Any,
) -> None:
    """
    Reject a record whose arrival would close a dependency cycle.

    Pure pre-check: the graph is recomputed on a copy, so this never
    mutates the live mapping. Callers can therefore validate first and
    install second, and a rejected Module simply never reaches the
    live tables -- there is no half-installed state to undo.

    A record that replaces an existing id is validated as a
    replacement, since the trial copy overwrites that entry.
    """
    trial = dict(
        modules
    )

    trial[record.id] = record

    (
        dependencies,
        dependents,
    ) = build_dependency_maps(
        trial
    )

    topological_order(
        trial,
        dependencies,
        dependents,
    )
