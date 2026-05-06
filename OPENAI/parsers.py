import json
from pathlib import Path
from typing import Callable, Any


# ── err_extraction ────────────────────────────────────────────────────────────

def parse_err_extraction_old_old(text: str) -> tuple[bool, list[bool]]:
    """
    Expects a comma-separated string of T/F values, e.g. "T,F,T,T,F".
    Also handles newline-separated responses by normalising to commas first.
    Raises ValueError if any token is unexpected.
    """
    import re

    # We define the separator once to keep the main pattern clean
    # it matches: 1+ spaces OR a single comma OR a single newline
    # Logic: (Optional space) + (One comma OR One newline OR One or more spaces) + (Optional space)
    sep = r"[ \t]*[,\n \t][ \t]*"    # Using re.VERBOSE (re.X) allows us to write the regex across multiple lines
    pattern = rf"""
    ^
    (\w+)          # Group 1: First Value
    {sep}
    (\w+)          # Group 2
    {sep}
    (\w+)          # Group 3
    $
    """

    value_map = {
        "T": True,
        "True": True,
        "1": True,
        "F": False,
        "False": False,
        "0": False
    }
    # Use re.X to enable verbose mode
    try:
        match = re.search(pattern, text, re.VERBOSE)
        if match:
            # 1. Accessing individually

            # 2. Getting all values as a tuple
            result = []
            for i in range(1, 4):
                v = match.group(i)
                if v in value_map:
                    result.append(value_map[v])
                else:
                    raise RuntimeError("Unknown value!")

            # print(f"Extraction Successful!")
            # print(f"First: {v1}, Last: {v6}")
            # print(f"Full List: {all_values}")
        else:
            raise ValueError("Not able to parse this")
    except TypeError as e:
        return False, []

    assert len(result) == 3
    return True, result


def parse_err_extraction_old(text: str) -> tuple[bool, bool]:
    value_map = {
        "Yes": True,
        "T": True,
        "True": True,
        "1": True,
        "No": False,
        "F": False,
        "False": False,
        "0": False
    }

    if text in value_map:
        return True, value_map[text]
    print("not found:", text)
    raise ValueError(text)


def parse_err_extraction(text: str) -> tuple[bool, bool]:
    value_map = {
        "Yes": True,
        "T": True,
        "True": True,
        "1": True,
        "No": False,
        "F": False,
        "False": False,
        "0": False
    }
    pattern = r"^Answer = (Yes|No)*"
    import re
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
        if answer in value_map:
            return True, value_map[answer]

    raise ValueError(text)

    # ── concept_identification ────────────────────────────────────────────────────


def parse_concept_identification_old(text: str) -> tuple[bool, int]:
    """
    Expects a single integer as the model response.
    Raises ValueError if parsing fails.
    """
    try:
        return True, int(text.strip())
    except ValueError:
        end = text.strip().split("\n")
        final = int(end[-1].strip())
        return True, final


def parse_concept_identification(text: str) -> tuple[bool, int]:
    pattern = r"^Answer = ([\d]+)$"
    import re
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
        return True, int(answer)
    elif "\n" in text:
        # last_line = text.split("\n")[-1]
        # match_ll = re.search(pattern, last_line)
        # if match_ll:
        #    answer = match_ll.group(1)
        #    return True, int(answer)

        first_line = text.split("\n")[0]
        match_fl = re.search(pattern, first_line)
        if match_fl:
            answer = match_fl.group(1)
            return True, int(answer)

    raise ValueError("Non valid ansewer", text)


def parse_concept_identification_multi(text: str) -> tuple[bool, list[int]]:
    import re

    # 1. Validate the format and capture the content inside {}
    format_pattern = r"^Answer = \{(?P<content>.*?)\}"
    match = re.search(format_pattern, text)

    if match:
        content = match.group("content")
        # 2. Pull all individual integers out of the captured content
        integers = [int(n) for n in re.findall(r"\d+", content)]
        print(integers)  # Output: [7, 11, 23]
        return True, integers
    elif "\n" in text:
        first_line = text.split("\n")[0]
        match_fl = re.search(format_pattern, first_line)
        if match_fl:
            content_fl = match_fl.group("content")
            # 2. Pull all individual integers out of the captured content
            integers = [int(n) for n in re.findall(r"\d+", content_fl)]
            return True, integers

    raise ValueError("Non valid ansewer", text)


def parse_with_mapping_wrapper(
    parse_fn:     Callable[[str], tuple[bool, Any]],
    mapping_path: str = "err_mapping.json",
    non_answer: str = "non_answer.json"
):
    return lambda text: parse_with_mapping(
        text=text,
        parse_fn=parse_fn,
        mapping_path=mapping_path,
        non_answer=non_answer
    )

# ── Fallback wrapper ──────────────────────────────────────────────────────────


def parse_with_mapping(
    text:         str,
    parse_fn:     Callable[[str], tuple[bool, Any]],
    mapping_path: str = "err_mapping.json",
    non_answer: str = "non_answer.json"
) -> tuple[bool, list[bool]]:
    """
    Attempts parse_fn(text) first. If that fails, loads mapping_path and
    checks for a fallback entry. Raises ValueError if both fail.

    Args:
        text:         Raw model response string.
        parse_fn:     The canonical parser for this query type.
        mapping_path: Path to the JSON fallback mapping file.

    Returns:
        The parsed value, from either parse_fn or the mapping.
    """

    try:
        return parse_fn(text)
    except (ValueError, AttributeError):
        pass
    print("org_fail", mapping_path)
    path = Path(mapping_path)
    if path.exists():
        with path.open() as f:
            mapping = json.load(f)
        if text in mapping:
            return True, mapping[text]

    path_non = Path(non_answer)
    if path_non.exists():
        with path_non.open() as f:
            non_answer = json.load(f)
        if text in non_answer:
            return False, []

    raise ValueError(
        f"Parsing failed and no mapping entry found for response:\n"
        f"  '{text}'\n"
        f"Add it to {mapping_path} to proceed."
    )


def test():
    test_str1 = "F,F,F"
    succ, ans = parse_with_mapping(text=test_str1, parse_fn=parse_err_extraction)
    print(succ, ans)


if __name__ == "__main__":
    test()
