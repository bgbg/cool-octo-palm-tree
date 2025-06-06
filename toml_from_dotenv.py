import os
import sys
from dotenv import dotenv_values

# Default .env path
env_path = sys.argv[1] if len(sys.argv) > 1 else ".env"

if not os.path.exists(env_path):
    print(f"File not found: {env_path}", file=sys.stderr)
    sys.exit(1)

# Load .env as dict
env_dict = dotenv_values(env_path)


def to_toml_value(val: str) -> str:
    # TOML: always quote strings
    return '"' + val.replace('"', '"') + '"'


for k, v in env_dict.items():
    print(f"{k} = {to_toml_value(v)}")
