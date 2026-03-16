#!/usr/bin/env python3
"""
notify.py — Send email notifications for pipeline job events.

All events for the same job thread together in one email conversation.

Usage (from SLURM scripts):
    python scripts/utils/notify.py --config config/my_config.yaml \
        --method proseg --sample-id SAMPLE__000001 --event start --job-id $SLURM_JOB_ID

    python scripts/utils/notify.py --config config/my_config.yaml \
        --method proseg --sample-id SAMPLE__000001 --event success --job-id $SLURM_JOB_ID \
        --elapsed "45m12s"
"""

import os
import yaml
import argparse
import subprocess
from email.mime.text import MIMEText

MESSAGES = {
    "start":   "{method} started for {sample_id}",
    "success": "{method} finished for {sample_id} ({elapsed})",
    "failed":  "{method} failed for {sample_id} ({elapsed}) — check logs: logs/seg_{method}_{sample_id}_{job_id}.{{out,err}}",
}


def load_notification_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("notifications", {})


def send_email(to_addr: str, subject: str, body: str, thread_id: str, is_start: bool):
    node = os.uname().nodename
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["To"] = to_addr
    msg["From"] = f"segmentation-pipeline@{node}"
    if is_start:
        msg["Message-ID"] = thread_id
    else:
        msg["In-Reply-To"] = thread_id
        msg["References"] = thread_id

    try:
        proc = subprocess.run(
            ["sendmail", "-t"],
            input=msg.as_string(),
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback to mail
    try:
        proc = subprocess.run(
            ["mail", "-s", subject, to_addr],
            input=body,
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(description="Send pipeline notifications")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--event", required=True, choices=["start", "success", "failed"])
    parser.add_argument("--job-id", default="unknown")
    parser.add_argument("--elapsed", default="unknown")
    args = parser.parse_args()

    notif = load_notification_config(args.config)
    email = notif.get("email", "")
    if not email:
        return

    node = os.uname().nodename
    subject = f"[seg] {args.method} — {args.sample_id} ({args.job_id})"
    body = MESSAGES[args.event].format(
        method=args.method,
        sample_id=args.sample_id,
        job_id=args.job_id,
        elapsed=args.elapsed,
    )
    thread_id = f"<slurm-seg-{args.method}-{args.sample_id}-{args.job_id}@{node}>"

    if send_email(email, subject, body, thread_id, is_start=(args.event == "start")):
        print(f"[NOTIFY] {args.event} → {email}")


if __name__ == "__main__":
    main()
