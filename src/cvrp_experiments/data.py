import yaml


def read_logs(path_to_logs: str) -> list[str]:
  with open(path_to_logs, 'r', encoding="utf-8") as f:
    logs = f.read()
  return logs.split("---\n")[:-1]


def parse_log_line(log_line: str) -> dict:
  return yaml.load(log_line, Loader=yaml.SafeLoader)
