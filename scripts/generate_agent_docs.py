#!/usr/bin/env python3
"""Generate AGENT.md documentation from Python module introspection.

This script extracts function signatures and docstrings to create
agent-friendly navigation documentation. Uses AST for static analysis
combined with runtime introspection for accurate signatures.

Usage:
    python generate_agent_docs.py [module_path] [--output FILE]

Example:
    python generate_agent_docs.py src/ml4t/diagnostic/splitters
    python generate_agent_docs.py src/ml4t/diagnostic/evaluation/stats --output stats_docs.md
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionDoc:
    """Documentation for a function/method."""

    name: str
    signature: str
    docstring_first_line: str
    is_public: bool
    lineno: int


@dataclass
class ClassDoc:
    """Documentation for a class."""

    name: str
    docstring_first_line: str
    methods: list[FunctionDoc]
    is_public: bool
    lineno: int


@dataclass
class ModuleDoc:
    """Documentation for a module."""

    name: str
    path: Path
    docstring_first_line: str
    functions: list[FunctionDoc]
    classes: list[ClassDoc]
    line_count: int


def get_first_line(docstring: str | None) -> str:
    """Extract first non-empty line from docstring."""
    if not docstring:
        return ""
    for line in docstring.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def extract_signature_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract function signature from AST node."""
    args = []

    # Positional args
    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass

        # Check for default value
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        if i >= defaults_offset:
            default = node.args.defaults[i - defaults_offset]
            try:
                arg_str += f" = {ast.unparse(default)}"
            except Exception:
                arg_str += " = ..."

        args.append(arg_str)

    # *args
    if node.args.vararg:
        arg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            try:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            except Exception:
                pass
        args.append(arg_str)
    elif node.args.kwonlyargs:
        args.append("*")

    # Keyword-only args
    for i, arg in enumerate(node.args.kwonlyargs):
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        if node.args.kw_defaults[i] is not None:
            try:
                arg_str += f" = {ast.unparse(node.args.kw_defaults[i])}"
            except Exception:
                arg_str += " = ..."
        args.append(arg_str)

    # **kwargs
    if node.args.kwarg:
        arg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            try:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            except Exception:
                pass
        args.append(arg_str)

    # Return annotation
    return_str = ""
    if node.returns:
        try:
            return_str = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass

    return f"({', '.join(args)}){return_str}"


def analyze_module_ast(path: Path) -> ModuleDoc:
    """Analyze a Python module using AST parsing."""
    source = path.read_text()
    tree = ast.parse(source)
    line_count = len(source.splitlines())

    # Get module docstring
    module_docstring = ast.get_docstring(tree) or ""

    functions: list[FunctionDoc] = []
    classes: list[ClassDoc] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            is_public = not node.name.startswith("_")
            functions.append(
                FunctionDoc(
                    name=node.name,
                    signature=extract_signature_ast(node),
                    docstring_first_line=get_first_line(ast.get_docstring(node)),
                    is_public=is_public,
                    lineno=node.lineno,
                )
            )
        elif isinstance(node, ast.ClassDef):
            is_public = not node.name.startswith("_")
            methods: list[FunctionDoc] = []

            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    method_public = not item.name.startswith("_") or item.name in (
                        "__init__",
                        "__call__",
                        "__enter__",
                        "__exit__",
                    )
                    methods.append(
                        FunctionDoc(
                            name=item.name,
                            signature=extract_signature_ast(item),
                            docstring_first_line=get_first_line(ast.get_docstring(item)),
                            is_public=method_public,
                            lineno=item.lineno,
                        )
                    )

            classes.append(
                ClassDoc(
                    name=node.name,
                    docstring_first_line=get_first_line(ast.get_docstring(node)),
                    methods=methods,
                    is_public=is_public,
                    lineno=node.lineno,
                )
            )

    return ModuleDoc(
        name=path.stem,
        path=path,
        docstring_first_line=get_first_line(module_docstring),
        functions=functions,
        classes=classes,
        line_count=line_count,
    )


def format_module_doc(doc: ModuleDoc, include_private: bool = False) -> str:
    """Format module documentation as markdown."""
    lines = []

    # Module header
    lines.append(f"### {doc.name}.py ({doc.line_count} lines)")
    if doc.docstring_first_line:
        lines.append(f"*{doc.docstring_first_line}*")
    lines.append("")

    # Classes
    public_classes = [c for c in doc.classes if c.is_public or include_private]
    if public_classes:
        for cls in public_classes:
            prefix = "" if cls.is_public else "_"
            lines.append(f"**class {prefix}{cls.name}**")
            if cls.docstring_first_line:
                lines.append(f"  {cls.docstring_first_line}")

            # Key methods (skip dunder except __init__)
            key_methods = [
                m
                for m in cls.methods
                if m.is_public and (not m.name.startswith("__") or m.name == "__init__")
            ]
            if key_methods:
                for method in key_methods[:5]:  # Limit to 5 methods
                    sig = method.signature
                    # Truncate long signatures
                    if len(sig) > 60:
                        sig = sig[:57] + "..."
                    lines.append(f"  - `{method.name}{sig}`")
                    if method.docstring_first_line:
                        lines.append(f"    {method.docstring_first_line}")
                if len(key_methods) > 5:
                    lines.append(f"  - ... and {len(key_methods) - 5} more methods")
            lines.append("")

    # Functions
    public_funcs = [f for f in doc.functions if f.is_public or include_private]
    if public_funcs:
        lines.append("**Functions:**")
        for func in public_funcs:
            sig = func.signature
            if len(sig) > 60:
                sig = sig[:57] + "..."
            prefix = "" if func.is_public else "_"
            lines.append(f"- `{prefix}{func.name}{sig}`")
            if func.docstring_first_line:
                lines.append(f"  {func.docstring_first_line}")
        lines.append("")

    return "\n".join(lines)


def analyze_directory(dir_path: Path, recursive: bool = True, include_private: bool = False) -> str:
    """Analyze all Python files in a directory."""
    output_lines = []

    # Get all Python files
    pattern = "**/*.py" if recursive else "*.py"
    py_files = sorted(dir_path.glob(pattern))

    # Filter out __pycache__ and test files
    py_files = [
        f for f in py_files if "__pycache__" not in str(f) and not f.name.startswith("test_")
    ]

    # Group by subdirectory
    by_subdir: dict[str, list[Path]] = {}
    for f in py_files:
        rel = f.relative_to(dir_path)
        if len(rel.parts) == 1:
            key = "."
        else:
            key = str(rel.parent)
        by_subdir.setdefault(key, []).append(f)

    # Process each subdirectory
    for subdir in sorted(by_subdir.keys()):
        files = by_subdir[subdir]
        if subdir != ".":
            output_lines.append(f"\n## {subdir}/\n")

        for py_file in files:
            if py_file.name == "__init__.py":
                continue  # Skip __init__.py files for now
            try:
                doc = analyze_module_ast(py_file)
                # Only include if has public content
                has_public = any(c.is_public for c in doc.classes) or any(
                    f.is_public for f in doc.functions
                )
                if has_public or include_private:
                    output_lines.append(format_module_doc(doc, include_private))
            except Exception as e:
                output_lines.append(f"### {py_file.name}\n*Error parsing: {e}*\n")

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate AGENT.md documentation")
    parser.add_argument("path", type=Path, help="Module or directory to analyze")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")
    parser.add_argument(
        "--include-private", action="store_true", help="Include private functions/classes"
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="Don't recurse into subdirectories"
    )

    args = parser.parse_args()

    if args.path.is_file():
        doc = analyze_module_ast(args.path)
        output = format_module_doc(doc, args.include_private)
    elif args.path.is_dir():
        output = analyze_directory(
            args.path,
            recursive=not args.no_recursive,
            include_private=args.include_private,
        )
    else:
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
