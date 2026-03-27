#!/usr/bin/env python3
"""
notify_chain.py — One email at chain start, one at chain finish.

start   Called from the master script right after submitting all jobs.
finish  Called by the chain_notify SLURM job (runs afterany all jobs).
        Uses sacct to get per-job elapsed time and exit status — no
        status files needed inside individual SLURM jobs.

Usage:
    # Submission time (login node):
    python notify_chain.py --email you@ucsf.edu --manifest logs/chain_20260327_142000/manifest.json --event start

    # Chain_notify SLURM job:
    python notify_chain.py --email you@ucsf.edu --manifest logs/chain_20260327_142000/manifest.json --event finish \
        --attachments /path/qc_report_SAMPLE1.pdf /path/qc_report_SAMPLE2.pdf
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


# All major US carrier SMS gateways (blast all — only the right one delivers)
SMS_GATEWAYS = [
    "txt.att.net", "tmomail.net", "vtext.com", "messaging.sprintpcs.com",
    "msg.fi.google.com", "email.uscc.net", "sms.myboostmobile.com",
    "vmobl.com", "mmst5.tracfone.com", "mymetropcs.com",
]


def _sendmail(msg) -> bool:
    try:
        proc = subprocess.run(
            ["sendmail", "-t"],
            input=msg.as_string(),
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_msg(to_addr: str, subject: str, body: str,
               thread_id: str, is_reply: bool = False,
               attachments: list = None):
    node = os.uname().nodename
    valid_attach = [p for p in (attachments or []) if p and os.path.exists(p)]

    if valid_attach:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body))
        for path in valid_attach:
            with open(path, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="pdf")
                part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
                msg.attach(part)
    else:
        msg = MIMEText(body)

    msg["Subject"] = subject
    msg["To"]      = to_addr
    msg["From"]    = f"slurm@{node}"
    if is_reply:
        msg["In-Reply-To"] = thread_id
        msg["References"]  = thread_id
    else:
        msg["Message-ID"] = thread_id
    return msg


def _query_sacct(job_ids: list) -> dict:
    """
    Return {job_id_str: {state, elapsed}} for each job.
    Uses sacct —available on any SLURM node without entering the container.
    """
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            ["sacct",
             "-j", ",".join(str(j) for j in job_ids),
             "--format=JobID,State,Elapsed",
             "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=30,
        )
        out = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            jid = parts[0].strip()
            # Skip sub-steps like 12345.batch or 12345.extern
            if "." in jid or "_" in jid:
                continue
            out[jid] = {"state": parts[1].strip(), "elapsed": parts[2].strip()}
        return out
    except Exception:
        return {}


def _send_sms(phone: str, subject: str, body: str):
    node = os.uname().nodename
    for gateway in SMS_GATEWAYS:
        sms = MIMEText(body)
        sms["Subject"] = subject
        sms["To"]      = f"{phone}@{gateway}"
        sms["From"]    = f"slurm@{node}"
        _sendmail(sms)


def send_start(manifest: dict, email: str, phone: str = ""):
    chain_id = manifest["chain_id"]
    jobs     = manifest["jobs"]  # [{method, sample_id, job_id}]
    n        = len(jobs)
    node     = os.uname().nodename
    thread_id = f"<seg-chain-{chain_id}@{node}>"

    subject = f"[seg] Chain started — {n} job{'s' if n != 1 else ''} — {chain_id}"

    lines = [
        f"Segmentation chain submitted  ({chain_id})",
        "",
        f"{'Method':<20} {'Sample':<42} {'Job ID':>10}",
        "─" * 74,
    ]
    for j in jobs:
        lines.append(f"{j['method']:<20} {j['sample_id']:<42} {str(j['job_id']):>10}")
    lines += [
        "",
        f"{n} job(s) submitted.",
        "You will receive one summary email when the chain completes.",
    ]
    body = "\n".join(lines)

    msg = _build_msg(email, subject, body, thread_id)
    if _sendmail(msg):
        print(f"[NOTIFY] chain start → {email}")
    else:
        print(f"[NOTIFY] chain start send failed (sendmail not available?)")

    # No SMS on start (avoid noise)


def send_finish(manifest: dict, email: str, phone: str = "", attachments: list = None):
    chain_id  = manifest["chain_id"]
    jobs      = manifest["jobs"]
    node      = os.uname().nodename
    thread_id = f"<seg-chain-{chain_id}@{node}>"

    job_ids   = [str(j["job_id"]) for j in jobs]
    sacct     = _query_sacct(job_ids)

    lines = [f"Segmentation chain complete  ({chain_id})", ""]
    lines.append(f"{'Method':<20} {'Sample':<38} {'Job ID':>10} {'Elapsed':>10}  Status")
    lines.append("─" * 84)

    all_ok = True
    for j in jobs:
        jid    = str(j["job_id"])
        info   = sacct.get(jid, {})
        state   = info.get("state", "UNKNOWN")
        elapsed = info.get("elapsed", "—")
        ok      = (state == "COMPLETED")
        if not ok:
            all_ok = False
        mark = "✓" if ok else f"✗  {state}"
        lines.append(
            f"{j['method']:<20} {j['sample_id']:<38} {jid:>10} {elapsed:>10}  {mark}"
        )

    lines += ["", "All jobs completed successfully." if all_ok else "One or more jobs failed — see above."]

    # Note any QC PDFs that weren't found
    missing = [p for p in (attachments or []) if p and not os.path.exists(p)]
    if missing:
        lines += ["", "QC report(s) not yet written or path not found:"]
        lines += [f"  {p}" for p in missing]

    body        = "\n".join(lines)
    status_word = "complete" if all_ok else "ERROR"
    subject     = f"[seg] Chain {status_word} — {chain_id}"

    valid_attach = [p for p in (attachments or []) if p and os.path.exists(p)]
    msg = _build_msg(email, subject, body, thread_id, is_reply=True, attachments=valid_attach)
    if _sendmail(msg):
        attach_note = f" (+{len(valid_attach)} PDF)" if valid_attach else ""
        print(f"[NOTIFY] chain finish → {email}{attach_note}")
    else:
        print(f"[NOTIFY] chain finish send failed")

    if phone:
        n_ok  = sum(1 for j in jobs if sacct.get(str(j["job_id"]), {}).get("state") == "COMPLETED")
        sms   = f"Seg chain {'done' if all_ok else 'FAILED'} ({chain_id}): {n_ok}/{len(jobs)} OK"
        _send_sms(phone, subject, sms)
        print(f"[NOTIFY] chain finish SMS → {phone}")


def main():
    parser = argparse.ArgumentParser(description="Chain-level pipeline notifications")
    parser.add_argument("--email",       required=True)
    parser.add_argument("--phone",       default="")
    parser.add_argument("--manifest",    required=True, help="Path to chain manifest JSON")
    parser.add_argument("--event",       required=True, choices=["start", "finish"])
    parser.add_argument("--attachments", nargs="*", default=[], help="QC PDF paths (finish only)")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())

    if args.event == "start":
        send_start(manifest, args.email, args.phone)
    else:
        send_finish(manifest, args.email, args.phone, args.attachments)


if __name__ == "__main__":
    main()
