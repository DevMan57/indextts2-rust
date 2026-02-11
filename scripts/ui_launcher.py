#!/usr/bin/env python3
"""Simple local UI for IndexTTS2 CLI.

No third-party dependencies required (Tkinter only).
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "target" / "release" / "indextts2.exe"


class IndexTTS2UI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("IndexTTS2 Rust Launcher")
        self.root.geometry("1024x760")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.proc: subprocess.Popen[str] | None = None

        self._init_vars()
        self._build_ui()
        self._refresh_mode_fields()
        self.root.after(100, self._drain_logs)

    def _init_vars(self) -> None:
        self.exe_var = tk.StringVar(value=str(DEFAULT_EXE))
        self.speaker_var = tk.StringVar(value="checkpoints/speaker_16k.wav")
        self.output_var = tk.StringVar(value="debug/ui_output.wav")
        self.text_var = tk.StringVar(value="This is a UI test for IndexTTS2 Rust.")

        self.mode_var = tk.StringVar(value="VoiceClone")
        self.emotion_audio_var = tk.StringVar(value="speaker.wav")
        self.emotion_alpha_var = tk.StringVar(value="0.9")
        self.emotion_vector_var = tk.StringVar(value="0.60,0.00,0.00,0.00,0.00,0.00,0.10,0.20")
        self.emotion_text_var = tk.StringVar(value="I feel very happy and excited today.")

        self.temperature_var = tk.StringVar(value="0.8")
        self.top_k_var = tk.StringVar(value="0")
        self.top_p_var = tk.StringVar(value="1.0")
        self.rep_penalty_var = tk.StringVar(value="1.05")
        self.flow_steps_var = tk.StringVar(value="25")
        self.flow_cfg_var = tk.StringVar(value="0.7")
        self.derumble_cutoff_var = tk.StringVar(value="180")

        self.cpu_var = tk.BooleanVar(value=False)
        self.derumble_var = tk.BooleanVar(value=True)
        self.autoplay_var = tk.BooleanVar(value=True)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        ttk.Label(frame, text="Executable").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.exe_var, width=90).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(frame, text="Browse", command=self._pick_exe).grid(row=row, column=2, sticky="e")
        row += 1

        ttk.Label(frame, text="Speaker WAV").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.speaker_var, width=90).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(frame, text="Browse", command=lambda: self._pick_file(self.speaker_var)).grid(row=row, column=2, sticky="e")
        row += 1

        ttk.Label(frame, text="Output WAV").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.output_var, width=90).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(frame, text="Browse", command=self._pick_output).grid(row=row, column=2, sticky="e")
        row += 1

        ttk.Label(frame, text="Mode").grid(row=row, column=0, sticky="w")
        mode_box = ttk.Combobox(
            frame,
            textvariable=self.mode_var,
            values=["VoiceClone", "EmotionAudio", "EmotionAudioBlend", "EmotionVector", "EmotionText"],
            state="readonly",
            width=28,
        )
        mode_box.grid(row=row, column=1, sticky="w", padx=6)
        mode_box.bind("<<ComboboxSelected>>", lambda _: self._refresh_mode_fields())
        row += 1

        ttk.Label(frame, text="Text").grid(row=row, column=0, sticky="nw")
        self.text_box = tk.Text(frame, height=5, wrap="word")
        self.text_box.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=6)
        self.text_box.insert("1.0", self.text_var.get())
        row += 1

        self.mode_section = ttk.LabelFrame(frame, text="Mode Options", padding=8)
        self.mode_section.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        self.mode_audio_row = ttk.Frame(self.mode_section)
        ttk.Label(self.mode_audio_row, text="Emotion audio").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.mode_audio_row, textvariable=self.emotion_audio_var, width=80).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(self.mode_audio_row, text="Browse", command=lambda: self._pick_file(self.emotion_audio_var)).grid(row=0, column=2, sticky="e")

        self.mode_alpha_row = ttk.Frame(self.mode_section)
        ttk.Label(self.mode_alpha_row, text="Emotion alpha").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.mode_alpha_row, textvariable=self.emotion_alpha_var, width=12).grid(row=0, column=1, sticky="w", padx=6)

        self.mode_vector_row = ttk.Frame(self.mode_section)
        ttk.Label(self.mode_vector_row, text="Emotion vector").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.mode_vector_row, textvariable=self.emotion_vector_var, width=85).grid(row=0, column=1, sticky="we", padx=6)

        self.mode_text_row = ttk.Frame(self.mode_section)
        ttk.Label(self.mode_text_row, text="Emotion text").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.mode_text_row, textvariable=self.emotion_text_var, width=85).grid(row=0, column=1, sticky="we", padx=6)

        self.gen_section = ttk.LabelFrame(frame, text="Generation", padding=8)
        self.gen_section.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        self._labeled_entry(self.gen_section, 0, 0, "Temperature", self.temperature_var)
        self._labeled_entry(self.gen_section, 0, 2, "Top-k", self.top_k_var)
        self._labeled_entry(self.gen_section, 0, 4, "Top-p", self.top_p_var)

        self._labeled_entry(self.gen_section, 1, 0, "Repetition penalty", self.rep_penalty_var)
        self._labeled_entry(self.gen_section, 1, 2, "Flow steps", self.flow_steps_var)
        self._labeled_entry(self.gen_section, 1, 4, "Flow cfg", self.flow_cfg_var)

        self._labeled_entry(self.gen_section, 2, 0, "De-rumble cutoff", self.derumble_cutoff_var)

        ttk.Checkbutton(self.gen_section, text="CPU", variable=self.cpu_var).grid(row=2, column=2, sticky="w")
        ttk.Checkbutton(self.gen_section, text="Enable de-rumble", variable=self.derumble_var).grid(row=2, column=3, sticky="w")
        ttk.Checkbutton(self.gen_section, text="Autoplay output", variable=self.autoplay_var).grid(row=2, column=4, sticky="w")

        self.button_row = ttk.Frame(frame)
        self.button_row.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        self.run_btn = ttk.Button(self.button_row, text="Run Inference", command=self._run)
        self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(self.button_row, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        ttk.Button(self.button_row, text="Build Release (CUDA)", command=self._build).pack(side=tk.LEFT, padx=8)

        ttk.Label(frame, text="Logs").grid(row=row, column=0, sticky="w")
        row += 1

        self.log_box = tk.Text(frame, height=16, wrap="word")
        self.log_box.grid(row=row, column=0, columnspan=3, sticky="nsew")

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)
        frame.rowconfigure(row, weight=2)

    def _labeled_entry(self, parent: ttk.Frame, row: int, col: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w")
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=col + 1, sticky="w", padx=6)

    def _pick_exe(self) -> None:
        path = filedialog.askopenfilename(title="Select indextts2 executable", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.exe_var.set(path)

    def _pick_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(title="Select WAV", filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if path:
            var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save output WAV", defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path:
            var = Path(path)
            if not var.parent.exists():
                var.parent.mkdir(parents=True, exist_ok=True)
            self.output_var.set(path)

    def _refresh_mode_fields(self) -> None:
        for child in self.mode_section.winfo_children():
            child.grid_forget()

        mode = self.mode_var.get()
        r = 0

        if mode in {"EmotionAudio", "EmotionAudioBlend"}:
            self.mode_audio_row.grid(row=r, column=0, sticky="we", pady=2)
            r += 1
            self.mode_alpha_row.grid(row=r, column=0, sticky="we", pady=2)
            r += 1
        elif mode == "EmotionVector":
            self.mode_vector_row.grid(row=r, column=0, sticky="we", pady=2)
            r += 1
            self.mode_alpha_row.grid(row=r, column=0, sticky="we", pady=2)
            r += 1
        elif mode == "EmotionText":
            self.mode_text_row.grid(row=r, column=0, sticky="we", pady=2)
            r += 1
            self.mode_alpha_row.grid(row=r, column=0, sticky="we", pady=2)
            r += 1

        if r == 0:
            ttk.Label(self.mode_section, text="No additional options for VoiceClone mode.").grid(row=0, column=0, sticky="w")

    def _append_log(self, line: str) -> None:
        self.log_box.insert(tk.END, line)
        self.log_box.see(tk.END)

    def _drain_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_logs)

    def _build_args(self) -> list[str]:
        exe = Path(self.exe_var.get().strip())
        if not exe.exists():
            raise ValueError(f"Executable not found: {exe}")

        speaker = self.speaker_var.get().strip()
        if not speaker:
            raise ValueError("Speaker path is required")

        output = self.output_var.get().strip()
        if not output:
            raise ValueError("Output path is required")

        text = self.text_box.get("1.0", tk.END).strip()
        if not text:
            raise ValueError("Text is required")

        args: list[str] = []
        if self.cpu_var.get():
            args.append("--cpu")
        args.extend([
            "infer",
            "--text", text,
            "--speaker", speaker,
            "--output", output,
            "--temperature", self.temperature_var.get().strip(),
            "--top-k", self.top_k_var.get().strip(),
            "--top-p", self.top_p_var.get().strip(),
            "--repetition-penalty", self.rep_penalty_var.get().strip(),
            "--flow-steps", self.flow_steps_var.get().strip(),
            "--flow-cfg-rate", self.flow_cfg_var.get().strip(),
        ])

        if self.derumble_var.get():
            args.extend(["--de-rumble", "--de-rumble-cutoff-hz", self.derumble_cutoff_var.get().strip()])

        mode = self.mode_var.get()
        if mode in {"EmotionAudio", "EmotionAudioBlend"}:
            args.extend([
                "--emotion-audio", self.emotion_audio_var.get().strip(),
                "--emotion-alpha", self.emotion_alpha_var.get().strip(),
            ])
        elif mode == "EmotionVector":
            args.extend([
                "--emotion-vector", self.emotion_vector_var.get().strip(),
                "--emotion-alpha", self.emotion_alpha_var.get().strip(),
            ])
        elif mode == "EmotionText":
            args.extend([
                "--use-emo-text",
                "--emo-text", self.emotion_text_var.get().strip(),
                "--emotion-alpha", self.emotion_alpha_var.get().strip(),
            ])

        return [str(exe)] + args

    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _run(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Inference is already running")
            return

        try:
            cmd = self._build_args()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._append_log(f"\n[{time.strftime('%H:%M:%S')}] Running: {' '.join(cmd)}\n")
        self._set_running(True)

        def target() -> None:
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.log_queue.put(line)
                rc = self.proc.wait()
                self.log_queue.put(f"\nProcess finished with exit code {rc}\n")

                if rc == 0:
                    out = Path(self.output_var.get().strip())
                    if out.exists() and self.autoplay_var.get():
                        try:
                            import os
                            os.startfile(str(out))
                        except Exception as exc:
                            self.log_queue.put(f"Autoplay failed: {exc}\n")
                else:
                    self.log_queue.put("Inference failed. Check logs above.\n")
            except Exception as exc:
                self.log_queue.put(f"Launcher error: {exc}\n")
            finally:
                self.proc = None
                self.root.after(0, lambda: self._set_running(False))

        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.log_queue.put("Stop requested.\n")

    def _build(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Cannot build while inference is running")
            return

        cmd = ["cargo", "build", "--release", "--features", "cuda"]
        self._append_log(f"\n[{time.strftime('%H:%M:%S')}] Building: {' '.join(cmd)}\n")
        self._set_running(True)

        def target() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_queue.put(line)
                rc = proc.wait()
                self.log_queue.put(f"\nBuild finished with exit code {rc}\n")
            except Exception as exc:
                self.log_queue.put(f"Build error: {exc}\n")
            finally:
                self.root.after(0, lambda: self._set_running(False))

        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()


def main() -> None:
    app = tk.Tk()
    IndexTTS2UI(app)
    app.mainloop()


if __name__ == "__main__":
    main()
