#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher v2 - bridge-aware.
Modes:
  1) Screener  -> quant_prototype.py  (watchlist from bridge/watchlists.json;
                  manual watchlist.txt override available)
  2) Portfolio -> bridge_module.py    (X-ray + alpha ledger on live holdings)
Double-click run.bat to open this window (Windows).
"""
import os
import sys
import json
import subprocess
import threading
import webbrowser
import tkinter as tk
from tkinter import scrolledtext, messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_FILE = os.path.join(HERE, "watchlist.txt")
SCREENER = os.path.join(HERE, "quant_prototype.py")
PORTFOLIO = os.path.join(HERE, "bridge_module.py")
REPORT = os.path.join(HERE, "quant_report.html")
PORT_REPORT = os.path.join(HERE, "portfolio_report.txt")
BRIDGE_DIR = os.path.join(HERE, "bridge")
DEFAULT = "MU\nNVDA\nAMD\nAVGO\nTSM\nINTC"

BG, PANEL, FG, DIM, ACCENT, WARN, ERR = ("#0e141f", "#1a2230", "#dce4f0",
                                          "#8b98ae", "#4ade9a", "#e0a92e", "#ff6b70")


def bridge_status():
    """Return (ok, summary_line, symbols_union, pending_contracts)."""
    try:
        with open(os.path.join(BRIDGE_DIR, "watchlists.json"), encoding="utf-8") as f:
            wl = json.load(f)
        syms = sorted({s for l in wl["watchlists"] for s in l["symbols"]})
        exported = wl.get("exported_at", "?")[:16].replace("T", " ")
        pending = []
        cpath = os.path.join(BRIDGE_DIR, "contracts.json")
        if os.path.exists(cpath):
            with open(cpath, encoding="utf-8") as f:
                cj = json.load(f)
            pending = [c["id"] for c in cj.get("contracts", [])
                       if "PENDING" in c.get("status", "").upper()
                       or "VIOLATION" in c.get("status", "").upper()]
        line = (f"bridge OK \u00b7 {len(wl['watchlists'])} lists / {len(syms)} symbols "
                f"\u00b7 exported {exported}")
        return True, line, syms, pending
    except Exception as e:
        return False, f"bridge not found ({e.__class__.__name__}) \u2014 manual watchlist mode", [], []


def load_manual_text():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, encoding="utf-8") as f:
            lines = [ln.split("#")[0].strip() for ln in f]
        syms = [s for s in lines if s]
        if syms:
            return "\n".join(syms)
    return DEFAULT


class App:
    def __init__(self, root):
        self.root = root
        root.title("Yifan the Starbound Nightfarer - Quant Console v2")
        root.geometry("520x760")
        root.configure(bg=BG)

        tk.Label(root, text="Yifan the Starbound Nightfarer", bg=BG, fg=FG,
                 font=("Consolas", 14, "bold")).pack(pady=(18, 2))
        tk.Label(root, text="Quant Console \u00b7 bridge-aware", bg=BG, fg=DIM,
                 font=("Consolas", 9)).pack(pady=(0, 10))

        # --- bridge status panel
        self.b_ok, b_line, self.b_syms, self.b_pending = bridge_status()
        self.bridge_lbl = tk.Label(root, text=b_line, bg=PANEL,
                                   fg=ACCENT if self.b_ok else WARN,
                                   font=("Consolas", 9), padx=10, pady=6,
                                   wraplength=460, justify="left")
        self.bridge_lbl.pack(fill="x", padx=20)
        if self.b_pending:
            tk.Label(root, text="\u26a0 contracts pending: " + ", ".join(self.b_pending),
                     bg=PANEL, fg=ERR, font=("Consolas", 9), padx=10, pady=4,
                     wraplength=460, justify="left").pack(fill="x", padx=20, pady=(2, 0))

        # --- watchlist editor + override toggle
        tk.Label(root, text="Watchlist", bg=BG, fg=ACCENT,
                 font=("Consolas", 14, "bold")).pack(pady=(12, 2))
        self.override = tk.BooleanVar(value=not self.b_ok)
        self.chk = tk.Checkbutton(
            root, text="manual override (use the list below \u2192 watchlist.txt)",
            variable=self.override, command=self._toggle_editor,
            bg=BG, fg=DIM, selectcolor=PANEL, activebackground=BG,
            activeforeground=FG, font=("Consolas", 9))
        self.chk.pack()

        self.text = scrolledtext.ScrolledText(
            root, width=30, height=10, font=("Consolas", 12),
            bg=PANEL, fg=FG, insertbackground=ACCENT,
            relief="flat", borderwidth=8)
        self.text.pack(pady=8)
        self._toggle_editor()

        # --- run buttons
        row = tk.Frame(root, bg=BG); row.pack(pady=6)
        self.btn_screen = tk.Button(
            row, text="\u25b6 Screener", command=lambda: self.run("screener"),
            bg=ACCENT, fg="#06231a", font=("Consolas", 12, "bold"),
            relief="flat", padx=16, pady=8, cursor="hand2", activebackground="#3bc285")
        self.btn_screen.pack(side="left", padx=6)
        self.btn_port = tk.Button(
            row, text="\u25b6 Portfolio X-Ray", command=lambda: self.run("portfolio"),
            bg="#3b82d4", fg="#061523", font=("Consolas", 12, "bold"),
            relief="flat", padx=16, pady=8, cursor="hand2", activebackground="#2f6cb0",
            state=("normal" if (self.b_ok and os.path.exists(PORTFOLIO)) else "disabled"))
        self.btn_port.pack(side="left", padx=6)

        self.status = tk.Label(root, text="", bg=BG, fg=DIM,
                               font=("Consolas", 9), wraplength=470, justify="left")
        self.status.pack(pady=8, fill="x", padx=20)

        # --- output pane (portfolio mode prints here)
        self.out = scrolledtext.ScrolledText(
            root, width=58, height=12, font=("Consolas", 9),
            bg=PANEL, fg=FG, relief="flat", borderwidth=8, state="disabled")
        self.out.pack(pady=(0, 14), padx=14, fill="both", expand=True)

    # ---------- helpers ----------
    def _toggle_editor(self):
        manual = self.override.get()
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        if manual:
            self.text.insert("1.0", load_manual_text())
            self.text.config(fg=FG)
        else:
            self.text.insert("1.0", "\n".join(self.b_syms))
            self.text.config(fg=DIM, state="disabled")  # bridge is read-only here:
            # edit lists in Robinhood, then re-export via Claude.

    def set_status(self, msg, color=DIM):
        self.status.config(text=msg, fg=color)
        self.root.update_idletasks()

    def _print(self, s):
        self.out.config(state="normal")
        self.out.insert("end", s)
        self.out.see("end")
        self.out.config(state="disabled")

    def save_manual(self):
        syms = [s.strip().upper() for s in self.text.get("1.0", "end").splitlines() if s.strip()]
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            f.write("# one ticker per line (manual override)\n")
            f.write("\n".join(syms) + "\n")
        return syms

    # ---------- run ----------
    def run(self, mode):
        if mode == "screener" and self.override.get():
            syms = self.save_manual()
            if not syms:
                messagebox.showwarning("Notice", "Watchlist cannot be empty")
                return
            note = f"manual list saved ({len(syms)})"
        else:
            note = "source: bridge (Robinhood via Claude)" if self.b_ok else "manual"
        for b in (self.btn_screen, self.btn_port):
            b.config(state="disabled")
        self.set_status(f"{note}\nrunning {mode}\u2026 (1\u20132 min, network-bound)", WARN)
        self.out.config(state="normal"); self.out.delete("1.0", "end"); self.out.config(state="disabled")
        threading.Thread(target=self._run_thread, args=(mode,), daemon=True).start()

    def _run_thread(self, mode):
        script = SCREENER if mode == "screener" else PORTFOLIO
        cmd = [sys.executable, script] + (["--no-open"] if mode == "portfolio" else [])
        try:
            proc = subprocess.run(cmd, cwd=HERE,
                                  capture_output=True, text=True, timeout=900)
            ok = proc.returncode == 0
            payload = proc.stdout or ""
            err = (proc.stderr or proc.stdout or "Unknown error")[-1200:]
            self.root.after(0, lambda: self._done(mode, ok, payload, err))
        except subprocess.TimeoutExpired:
            self.root.after(0, lambda: self._done(mode, False, "", "Timed out (>15 min)."))
        except Exception as e:
            self.root.after(0, lambda: self._done(mode, False, "", str(e)))

    def _done(self, mode, ok, payload, err):
        self.btn_screen.config(state="normal")
        self.btn_port.config(state=("normal" if (self.b_ok and os.path.exists(PORTFOLIO)) else "disabled"))
        if not ok:
            self.set_status("\u2717 Error:\n" + err, ERR)
            return
        if mode == "screener" and os.path.exists(REPORT):
            self.set_status("\u2713 Screener done \u2014 opening HTML report\u2026", ACCENT)
            webbrowser.open("file://" + REPORT.replace(os.sep, "/"))
        else:
            with open(PORT_REPORT, "w", encoding="utf-8") as f:
                f.write(payload)
            self._print(payload if payload.strip() else "(no output)")
            html = os.path.join(HERE, "portfolio_report.html")
            if os.path.exists(html):
                self.set_status("\u2713 Portfolio X-Ray done \u2014 opening HTML report\u2026", ACCENT)
                webbrowser.open("file://" + html.replace(os.sep, "/"))
            else:
                self.set_status(f"\u2713 done \u2014 text shown below (HTML not found)", ACCENT)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
