import os
import sys

# Add src to sys.path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from videocuts.utils.system import format_timestamp, parse_timestamp

def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1) == "00:00:01,000"
    assert format_timestamp(3661.123) == "01:01:01,123"

def test_parse_timestamp():
    assert parse_timestamp("00:00:00,000") == 0.0
    assert parse_timestamp("00:00:01,000") == 1.0
    assert abs(parse_timestamp("01:01:01,123") - 3661.123) < 1e-6

if __name__ == "__main__":
    test_format_timestamp()
    test_parse_timestamp()
    print("Basic tests passed!")
