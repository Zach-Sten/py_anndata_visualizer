#!/usr/bin/env python3
"""
notify.py — Pipeline job notifier via email, threaded per job.

All events for the same job (start → finish/error) land in one email thread.
Optionally attaches a file (e.g. QC PDF report) to the finish event.

Usage (from SLURM scripts):
    python3 scripts/utils/notify.py --email you@ucsf.edu \\
        --method proseg --sample-id XETG... --event start
    python3 scripts/utils/notify.py --email you@ucsf.edu \\
        --method proseg --sample-id XETG... --event finish --elapsed 45m12s
    python3 scripts/utils/notify.py --email you@ucsf.edu \\
        --method cellspa_qc --sample-id XETG... --event finish \\
        --elapsed 12m03s --attachment /path/to/qc_report.pdf
"""

import os
import argparse
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

MESSAGES = {
    "start":  "Job {job} started  (SLURM {job_id} on {node})",
    "finish": "Job {job} finished (SLURM {job_id} | {elapsed})",
    "error":  "Job {job} FAILED   (SLURM {job_id} | {elapsed})",
}

# All major US carrier email-to-SMS gateways
SMS_GATEWAYS = [
    "txt.att.net",
    "tmomail.net",
    "vtext.com",
    "messaging.sprintpcs.com",
    "msg.fi.google.com",
    "email.uscc.net",
    "sms.myboostmobile.com",
    "vmobl.com",
    "mmst5.tracfone.com",
    "mymetropcs.com",
]


def build_msg(to_addr: str, subject: str, body: str,
              thread_id: str, event: str,
              attachment_path: str = None):
    """Build MIME message with threading headers and optional attachment."""
    node = os.uname().nodename
    from_addr = f"slurm@{node}"

    if attachment_path and os.path.exists(attachment_path):
        msg = MIMEMultipart()
        msg.attach(MIMEText(body))
        fname = os.path.basename(attachment_path)
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header("Content-Disposition", "attachment", filename=fname)
            msg.attach(part)
    else:
        msg = MIMEText(body)

    msg["Subject"] = subject
    msg["To"]      = to_addr
    msg["From"]    = from_addr

    # Threading: start opens the thread; finish/error reply to it
    if event == "start":
        msg["Message-ID"] = thread_id
    else:
        msg["In-Reply-To"] = thread_id
        msg["References"]  = thread_id

    return msg


def sendmail(msg) -> bool:
    try:
        proc = subprocess.run(
            ["sendmail", "-t"],
            input=msg.as_string(),
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def send_sms(phone: str, subject: str, body: str, node: str):
    """Blast all carrier gateways — only the right one delivers."""
    for gateway in SMS_GATEWAYS:
        addr = f"{phone}@{gateway}"
        sms = MIMEText(body)
        sms["Subject"] = subject
        sms["To"]      = addr
        sms["From"]    = f"slurm@{node}"
        sendmail(sms)


def main():
    parser = argparse.ArgumentParser(description="Send pipeline job notifications")
    parser.add_argument("--email",     required=True)
    parser.add_argument("--method",    required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--event",     required=True, choices=["start", "finish", "error"])
    parser.add_argument("--elapsed",   default="")
    parser.add_argument("--phone",     default="", help="Phone number for SMS (optional)")
    parser.add_argument("--attachment", default=None,
                        help="File to attach (PDF report, finish event only)")
    args = parser.parse_args()

    email = args.email
    phone = args.phone
    if not email and not phone:
        return

    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    node   = os.uname().nodename
    job    = f"{args.method} — {args.sample_id}"

    text = MESSAGES[args.event].format(
        job=job, job_id=job_id, node=node, elapsed=args.elapsed or "—"
    )
    subject   = f"[seg] {args.method} {args.event} — {args.sample_id}"
    thread_id = f"<slurm-{args.method}-{args.sample_id}-{job_id}@{node}>"

    attachment = None
    if args.attachment:
        if os.path.exists(args.attachment):
            attachment = args.attachment
        else:
            print(f"[NOTIFY] attachment not found, sending without: {args.attachment}")

    if email:
        msg = build_msg(email, subject, text, thread_id, args.event, attachment)
        if sendmail(msg):
            suffix = f" (+{os.path.basename(attachment)})" if attachment else ""
            print(f"[NOTIFY] {args.event} → {email}{suffix}")

    if phone and args.event != "start":
        # SMS only on finish/error (skip start to avoid spam)
        send_sms(phone, subject, text, node)
        print(f"[NOTIFY] {args.event} → {phone}")


if __name__ == "__main__":
    main()
