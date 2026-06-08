#!/usr/bin/env python3
import sys
import argparse
import pytest


def main():
    parser = argparse.ArgumentParser(
        description="ACCEPT test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "python run_tests.py\n"
            "python run_tests.py tests/test_reg_ranges.py\n"
            "python run_tests.py tests/test_reg_ranges.py::TestHyperparameterArrayContracts::test_tunetype_mapping\n"
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="tests/",
        help="Test file, or file::Class::method for a single test (default: tests/)",
    )
    parser.add_argument("-x", "--failfast", action="store_true", help="Stop after first failure")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress verbose output")
    args, passthrough = parser.parse_known_args()

    flags = [args.target]
    if not args.no_verbose:
        flags.append("-v")
    if args.failfast:
        flags.append("-x")
    flags.extend(passthrough)

    sys.exit(pytest.main(flags))


if __name__ == "__main__":
    main()
