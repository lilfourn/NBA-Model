import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Send an email via SMTP.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", default=None)
    args = parser.parse_args()

    load_env(Path(".env"))

    smtp_host = require_env("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = require_env("SMTP_USER")
    smtp_password = require_env("SMTP_PASSWORD")
    smtp_to = require_env("SMTP_TO")

    body = args.body
    if args.body_file:
        body = Path(args.body_file).read_text(encoding="utf-8")

    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = smtp_to
    msg["Subject"] = args.subject
    msg.set_content(body or "(no body)")

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    except Exception as exc:
        print(f"Failed to send email: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
