import sys
import re
from typing import Tuple
from io import StringIO
from main import main


class TerminalColors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'


def get_pyautogen_version() -> Tuple[bool, str]:
    """Return (is_installed, version)."""
    try:
        try:
            from importlib.metadata import version, PackageNotFoundError
        except ImportError:  # For Python < 3.8
            from importlib_metadata import version, PackageNotFoundError
        return True, version("pyautogen")
    except PackageNotFoundError:
        return False, ""


def check_pyautogen_version(expected: str = "0.9.0") -> bool:
    """Check if pyautogen is installed and matches the expected version."""
    present, ver = get_pyautogen_version()
    if not present:
        print(f"{TerminalColors.RED}[X] Test 0 Failed. Pyautogen missing.{TerminalColors.RESET}")
        return False
    if ver == expected:
        print(f"{TerminalColors.GREEN}[V] Test 0 Passed. Pyautogen {ver}{TerminalColors.RESET}")
        return True
    print(
        f"{TerminalColors.RED}pyautogen version mismatch — installed: {ver} expected: {expected}{TerminalColors.RESET}"
    )
    return False


# ─────────────────────────────────────────────
# Configuration: data file path
# ─────────────────────────────────────────────
DATA_FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else "restaurant-data.txt"


def contains_num_with_tolerance(text: str, target: float, tol: float) -> Tuple[bool, float]:
    """
    Extract only the FIRST number (int or float) from the text.
    Compare it against the expected target value with a given tolerance.
    
    Returns:
        passed (bool): Whether the prediction is within tolerance.
        pred (float or None): The parsed prediction value or None if not found.
    """
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return False, None
    pred = float(match.group())
    return abs(pred - target) <= tol, pred


def public_tests() -> None:
    queries = [
        "How good is the restaurant taco bell overall?",
        "How good is the restaurant Chick-fil-A overall?",
        "What is the overall score for Starbucks?",
        "What is the overall score for In-n-Out",
        "What is the overall score for McDonald's?",
    ]
    expected = [3.00, 9.35, 8.06, 9.54, 3.65]
    tolerances = [0.20, 0.20, 0.15, 0.15, 0.15]

    abs_errors, passed = [], 0
    logs = []

    # Clear old runtime log
    open("runtime-log.txt", "w").close()

    for i, q in enumerate(queries):
        # Capture printed stdout
        buffer = StringIO()
        sys.stdout = buffer

        # Get predicted value from main()
        predicted = main(q, DATA_FILE_PATH)

        # Restore normal stdout
        sys.stdout = sys.__stdout__

        # Get captured printed output
        printed_output = buffer.getvalue()
        logs.append(printed_output)

        # Append captured output and return value to log file
        with open("runtime-log.txt", "a") as f:
            f.write(f"Query {i+1}: {q}\n")
            f.write("Captured stdout:\n")
            f.write(printed_output + "\n")
            f.write("Returned value:\n")
            f.write(str(predicted) + "\n\n")

        # Evaluate prediction
        ok, pred = contains_num_with_tolerance(str(predicted), expected[i], tolerances[i])
        error_cap = 10.0

        if pred is None:
            abs_errors.append(error_cap)
        elif ok:
            abs_errors.append(abs(pred - expected[i]))
        else:
            abs_errors.append(min(abs(pred - expected[i]), error_cap))

        if ok:
            print(
                f"{TerminalColors.GREEN}[V] Test {i+1} Passed.{TerminalColors.RESET} "
                f"Expected: {expected[i]:.3f}  Predicted: {pred:.3f}  "
                f"Query: {q}"
            )
            passed += 1
        else:
            print(
                f"{TerminalColors.RED}[X] Test {i+1} Failed.{TerminalColors.RESET} "
                f"Expected: {expected[i]:.3f}  Predicted: {pred}  "
                f"Query: {q}"
            )

    mae = sum(abs_errors) / len(abs_errors)
    print(f"---------- {passed}/{len(queries)} Tests Passed. MAE: {mae:.4f} ----------")


if __name__ == "__main__":
    check_pyautogen_version("0.9.0")
    public_tests()
