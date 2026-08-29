"""Limited, verifiable call/dependency context for the evidence_anchored_v3
pair-classification prompt variant.

Isolated experiment: does not modify v1/v2, DCAR, any raw Data/ file, or
configs/final_framework_config.yaml. Pure static text analysis, no LLM
calls -- every edge is a deterministic, auditable match against another
REAL file in the same dataset (never a fabricated reference), so
"dependency context" always traces back to a concrete, inspectable
import/class-usage (Java) or function-call (C) pattern in the source.

Two extractors, dispatched by file extension:

- Java (.java): (1) `import <project-package>.ClassName;` statements
  resolved against the dataset's own class names: (2) as a fallback, a
  bare, word-boundary reference to another file's class name anywhere in
  the source (catches same-package usage with no import statement).
- C (.c/.h): this project's curated LibEST source has had all LOCAL
  (project-relative) #include directives stripped -- confirmed zero
  `#include "somefile.h"` occurrences anywhere in code/; the only
  #include lines present are external library headers
  (`#include <openssl/...>`), which carry no file-to-file dependency
  information -- so file-level includes are not a usable signal here.
  Instead: a project-wide function-name index is built from
  each file's own function DEFINITIONS (a line ending without a
  semicolon, i.e. a definition, not a declaration/prototype), and a file
  gets a dependency edge to another file if it calls a function defined
  there (a real, verifiable cross-file call site -- confirmed to exist,
  e.g. `ossl_dump_ssl_errors` defined in est_ossl_util.c is called from 5
  other LibEST files).
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_JAVA_IMPORT = re.compile(r"^\s*import\s+([\w.]+)\s*;", re.MULTILINE)
_C_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "return", "sizeof", "else", "do",
    "typedef", "struct", "union", "enum", "case", "default", "goto",
})
_C_FUNCTION_DEFINITION = re.compile(
    r"^(?:static\s+|extern\s+|inline\s+)*"
    r"[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$",
    re.MULTILINE,
)


def _java_class_name(code_id: str) -> str:
    return PurePosixPath(code_id).stem


def extract_java_dependencies(
    code_id: str, code_text: str, class_name_to_code_id: dict[str, str]
) -> set[str]:
    """class_name_to_code_id maps every OTHER Java file's class name
    (basename without .java) to its code_id, so a match always resolves
    to a real file in this dataset."""
    own_class = _java_class_name(code_id)
    deps: set[str] = set()

    for imported in _JAVA_IMPORT.findall(code_text):
        simple_name = imported.rsplit(".", 1)[-1]
        target = class_name_to_code_id.get(simple_name)
        if target is not None and target != code_id:
            deps.add(target)

    for class_name, target in class_name_to_code_id.items():
        if target == code_id or class_name == own_class or target in deps:
            continue
        if re.search(rf"\b{re.escape(class_name)}\b", code_text):
            deps.add(target)

    return deps


def build_java_class_index(code_files: dict[str, str]) -> dict[str, str]:
    """class name (file basename without .java) -> code_id, for every
    .java file in the dataset."""
    return {
        _java_class_name(code_id): code_id
        for code_id in code_files
        if code_id.endswith(".java")
    }


def build_c_function_index(code_files: dict[str, str]) -> dict[str, str]:
    """function name -> the code_id of the file that DEFINES it (first
    definition wins on an ambiguous duplicate, which is rare and does not
    change the dependency conclusion either way)."""
    index: dict[str, str] = {}
    for code_id, text in code_files.items():
        if not (code_id.endswith(".c") or code_id.endswith(".h")):
            continue
        for match in _C_FUNCTION_DEFINITION.finditer(text):
            name = match.group(1)
            if name in _C_KEYWORDS:
                continue
            index.setdefault(name, code_id)
    return index


def extract_c_dependencies(
    code_id: str, code_text: str, function_index: dict[str, str]
) -> set[str]:
    deps: set[str] = set()
    for function_name, owner in function_index.items():
        if owner == code_id:
            continue
        if re.search(rf"\b{re.escape(function_name)}\s*\(", code_text):
            deps.add(owner)
    return deps


def build_dependency_graph(code_files: dict[str, str]) -> dict[str, set[str]]:
    """code_id -> set of other code_ids this file has a verifiable
    import/class-usage (Java) or function-call (C) dependency on. Files
    of an unrecognized extension get an empty dependency set (not an
    error) -- this project only has Java and C datasets, but a graceful
    empty result keeps this reusable without a hard failure."""
    java_ids = [c for c in code_files if c.endswith(".java")]
    c_ids = [c for c in code_files if c.endswith(".c") or c.endswith(".h")]

    graph: dict[str, set[str]] = {code_id: set() for code_id in code_files}

    if java_ids:
        class_index = build_java_class_index(code_files)
        for code_id in java_ids:
            graph[code_id] = extract_java_dependencies(
                code_id, code_files[code_id], class_index
            )

    if c_ids:
        function_index = build_c_function_index(code_files)
        for code_id in c_ids:
            graph[code_id] = extract_c_dependencies(
                code_id, code_files[code_id], function_index
            )

    return graph


def dependency_coverage_rate(dependency_graph: dict[str, set[str]]) -> float:
    """Fraction of ALL code files in the dataset's dependency graph that
    have at least one detected dependency edge (to or from another file)
    -- a whole-corpus, graph-connectivity view. This is NOT the same
    thing as how many of a specific run's manifest candidates actually
    get a dependency_context block shown (see
    manifest_dependency_coverage_rate) -- report both, never conflate
    them: this one can be well above the manifest-scoped rate simply
    because most of the corpus happens to be interconnected, independent
    of which files any particular manifest samples."""
    if not dependency_graph:
        return 0.0
    has_incoming: dict[str, bool] = {code_id: False for code_id in dependency_graph}
    for code_id, deps in dependency_graph.items():
        for target in deps:
            if target in has_incoming:
                has_incoming[target] = True
    covered = sum(
        1 for code_id, deps in dependency_graph.items()
        if deps or has_incoming[code_id]
    )
    return covered / len(dependency_graph)


def manifest_dependency_coverage_rate(
    dependency_graph: dict[str, set[str]], candidate_code_ids: set[str]
) -> float:
    """Fraction of THIS RUN's manifest candidate code_ids that actually
    have at least one dependency edge -- i.e. that will genuinely display
    a non-empty dependency_context block in the v3 prompt for this split.
    This is the primary, run-scoped coverage number; dependency_coverage_rate
    (whole corpus) is secondary/reference only and must be reported under
    a clearly distinct name, never as if it were this number."""
    if not candidate_code_ids:
        return 0.0
    covered = sum(1 for code_id in candidate_code_ids if dependency_graph.get(code_id))
    return covered / len(candidate_code_ids)


def dependency_neighbors_to_summarize(
    dependency_graph: dict[str, set[str]],
    candidate_code_ids: set[str],
    max_dependencies_shown: int,
) -> set[str]:
    """Exact set of neighbor code_ids that will actually be rendered in a
    v3 prompt's dependency_context block for this manifest scope, bounded
    by max_dependencies_shown per candidate (matching
    DependencyContext*PromptBuilder's own truncation) -- used to fetch or
    estimate summaries for only these files, never the whole dataset
    corpus."""
    shown: set[str] = set()
    for code_id in candidate_code_ids:
        neighbors = sorted(dependency_graph.get(code_id, set()))[:max_dependencies_shown]
        shown.update(neighbors)
    return shown
