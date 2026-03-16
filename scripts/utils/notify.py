#!/usr/bin/env python3
"""
notify.py — Send email/SMS notifications for pipeline job events.

For SMS: sends to ALL major carrier gateways simultaneously. Only the
correct carrier delivers; the rest silently fail. No carrier selection needed.

Usage (from SLURM scripts):
    python scripts/utils/notify.py \
        --config config/my_config.yaml \
        --method proseg \
        --sample-id XETG00143__0032645 \
        --status success \
        --job-id $SLURM_JOB_ID \
        --elapsed "45m12s"
"""

import os
import sys
import yaml
import argparse
import subprocess
from email.mime.text import MIMEText
from datetime import datetime

# All major US carrier email-to-SMS gateways
SMS_GATEWAYS = [
    "txt.att.net",               # AT&T
    "tmomail.net",               # T-Mobile
    "vtext.com",                 # Verizon
    "messaging.sprintpcs.com",   # Sprint / T-Mobile
    "msg.fi.google.com",         # Google Fi
    "email.uscc.net",            # US Cellular
    "sms.myboostmobile.com",     # Boost Mobile
    "vmobl.com",                 # Virgin Mobile
    "mmst5.tracfone.com",        # Tracfone
    "mymetropcs.com",            # Metro by T-Mobile
]


def load_notification_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("notifications", {})


def send_via_sendmail(to_addr: str, subject: str, body: str) -> bool:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["To"] = to_addr
    msg["From"] = f"segmentation-pipeline@{os.uname().nodename}"
    try:
        proc = subprocess.run(
            ["sendmail", "-t"],
            input=msg.as_string(),
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def send_via_mail(to_addr: str, subject: str, body: str) -> bool:
    try:
        proc = subprocess.run(
            ["mail", "-s", subject, to_addr],
            input=body,
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def send_one(to_addr: str, subject: str, body: str) -> bool:
    return send_via_sendmail(to_addr, subject, body) or \
           send_via_mail(to_addr, subject, body)


def send_sms(phone: str, subject: str, body: str):
    """Send to all carrier gateways. Only the right one delivers."""
    for gateway in SMS_GATEWAYS:
        addr = f"{phone}@{gateway}"
        send_one(addr, subject, body)


def build_message(method, sample_id, status, job_id, elapsed, node):
    tag = "COMPLETED" if status == "success" else "FAILED"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = f"[seg] {method} {tag} — {sample_id}"

    body = (
        f"Segmentation Pipeline\n"
        f"{'=' * 35}\n"
        f"Method:    {method}\n"
        f"Sample:    {sample_id}\n"
        f"Status:    {tag}\n"
        f"Job ID:    {job_id}\n"
        f"Node:      {node}\n"
        f"Elapsed:   {elapsed}\n"
        f"Time:      {ts}\n"
    )
    if status != "success":
        body += f"\nCheck logs: logs/seg_{method}_{sample_id}_{job_id}.{{out,err}}\n"

    sms_body = f"{method} {tag}\n{sample_id}\nJob {job_id} | {elapsed}"

    return subject, body, sms_body


def main():
    parser = argparse.ArgumentParser(description="Send pipeline notifications")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--status", required=True, choices=["success", "failed"])
    parser.add_argument("--job-id", default="unknown")
    parser.add_argument("--elapsed", default="unknown")
    args = parser.parse_args()

    notif = load_notification_config(args.config)
    if not notif:
        return

    email = notif.get("email", "")
    phone = notif.get("phone", "")

    if not email and not phone:
        return

    node = os.uname().nodename
    subject, body, sms_body = build_message(
        args.method, args.sample_id, args.status,
        args.job_id, args.elapsed, node,
    )

    if email:
        if send_one(email, subject, body):
            print(f"[NOTIFY] email → {email}")

    if phone:
        send_sms(phone, subject, sms_body)
        print(f"[NOTIFY] text → {phone}")


if __name__ == "__main__":
    main()
