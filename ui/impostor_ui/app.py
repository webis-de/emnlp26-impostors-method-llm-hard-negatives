# Copyright 2026 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from datetime import datetime
from tkinter import END, BooleanVar, DoubleVar, IntVar, StringVar, Text, Tk, filedialog, messagebox, ttk
from typing import Any, Dict, Optional, Tuple

from ablations.impostor_asgalf import ASGALFImpostorDetector
from ablations.impostor_homotopy import HBCImpostorDetector
from ablations.impostor_potha2017 import Potha2017ImpostorDetector
from ablations.std_impostor import StdImpostor
from genai_detection.config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector

TECHNIQUE_LABEL_TO_VALUE = {
    val: key for key, val in CONFIG.LABEL_TRANSLATIONS.items()
}


def _read_file_content(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _ensure_text_in_mongodb(detector: ImpostorDetector, text: str, dataset_name: str):
    docs = list(
        detector.mongoDB.find_document_by_non_id_field(
            collection=detector.mongoDB.original_collection,
            document_field_name="text",
            document_value=text,
        )
    )
    if docs:
        return docs[0]["_id"]

    result = detector.mongoDB.original_collection.insert_one(
        {
            "text": text,
            "dataset_name": dataset_name,
            "author": "ui_input",
            "assignment": "ui_ad_hoc",
            "created_at": datetime.utcnow().isoformat(),
            "source": "impostor_ui_tk",
        }
    )
    return result.inserted_id


def _build_detector(
    method_mode: str,
    ablation: Optional[str],
    common_params: Dict[str, Any],
    ablation_params: Dict[str, Any],
):
    if method_mode == "original":
        return ImpostorDetector(**common_params)

    if ablation == "ASGALF":
        return ASGALFImpostorDetector(**common_params)
    if ablation == "HBC (Homotopy)":
        return HBCImpostorDetector(**common_params)
    if ablation == "Potha2017":
        p_kwargs = dict(common_params)
        p_kwargs.pop("n_impostors", None)
        return Potha2017ImpostorDetector(
            **p_kwargs,
            impostors_per_problem=ablation_params["impostors_per_problem"],
            impostors_per_round=ablation_params["impostors_per_round"],
        )
    if ablation == "StdImpostor":
        return StdImpostor(
            **common_params,
            feature_type=ablation_params["feature_type"],
            char_n=ablation_params["char_n"],
            lowercase=ablation_params["lowercase"],
        )
    raise ValueError(f"Unknown ablation: {ablation}")


class ImpostorUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Impostor Method UI")
        self.root.geometry("1280x900")

        self.method_mode = StringVar(value="original")
        self.ablation = StringVar(value="ASGALF")
        self.technique_label = StringVar(value="Two-step LLM")
        self.dataset_name = StringVar(value=CONFIG.STUDENT_ESSAYS)

        self.rounds = IntVar(value=100)
        self.top_n = IntVar(value=100000)
        self.n_impostors = IntVar(value=25)
        self.portion_delete = DoubleVar(value=0.5)
        self.threshold = DoubleVar(value=0.1)
        self.min_n_tokens = IntVar(value=500)
        self.upsample = BooleanVar(value=True)

        self.impostors_per_problem = IntVar(value=50)
        self.impostors_per_round = IntVar(value=5)
        self.std_feature_type = StringVar(value="char")
        self.std_char_n = IntVar(value=3)
        self.std_lowercase = BooleanVar(value=False)

        self.disputed_file_path = StringVar(value="")
        self.candidate_file_path = StringVar(value="")

        self.worker_thread = None
        self.worker_result = None
        self.worker_error = None
        self.progress_value = 0

        self._build_layout()
        self._refresh_dynamic_controls()

    def _build_layout(self):
        title = ttk.Label(self.root, text="Impostor Method UI", font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 4), sticky="w")
        subtitle = ttk.Label(
            self.root,
            text="Select method, generation strategy, and parameters. Input disputed and candidate documents by file or text.",
        )
        subtitle.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

        cfg_frame = ttk.LabelFrame(self.root, text="Configuration")
        cfg_frame.grid(row=2, column=0, padx=12, pady=8, sticky="nsew")

        docs_frame = ttk.LabelFrame(self.root, text="Documents")
        docs_frame.grid(row=2, column=1, padx=12, pady=8, sticky="nsew")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        row = 0
        ttk.Label(cfg_frame, text="Method family").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        method_combo = ttk.Combobox(cfg_frame, textvariable=self.method_mode, state="readonly", values=["original", "ablations"])
        method_combo.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        method_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_dynamic_controls())
        row += 1

        self.ablation_row = row
        self.ablation_label = ttk.Label(cfg_frame, text="Ablation method")
        self.ablation_combo = ttk.Combobox(
            cfg_frame,
            textvariable=self.ablation,
            state="readonly",
            values=["ASGALF", "HBC (Homotopy)", "Potha2017", "StdImpostor"],
        )
        self.ablation_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_dynamic_controls())
        row += 1

        ttk.Label(cfg_frame, text="Impostor generation").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(
            cfg_frame,
            textvariable=self.technique_label,
            state="readonly",
            values=list(TECHNIQUE_LABEL_TO_VALUE.keys()),
        ).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        row += 1

        ttk.Label(cfg_frame, text="Dataset").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(
            cfg_frame,
            textvariable=self.dataset_name,
            state="readonly",
            values=[CONFIG.STUDENT_ESSAYS, CONFIG.BLOG],
        ).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        row += 1

        self._add_int_input(cfg_frame, row, "Rounds", self.rounds)
        row += 1
        self._add_int_input(cfg_frame, row, "Top n features", self.top_n)
        row += 1
        self._add_int_input(cfg_frame, row, "Number of impostors", self.n_impostors)
        row += 1
        self._add_float_input(cfg_frame, row, "Delete portion", self.portion_delete)
        row += 1
        self._add_float_input(cfg_frame, row, "Threshold", self.threshold)
        row += 1
        self._add_int_input(cfg_frame, row, "Min tokens", self.min_n_tokens)
        row += 1

        ttk.Checkbutton(cfg_frame, text="Upsample short texts", variable=self.upsample).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=4
        )
        row += 1

        self.ablation_specific_title = ttk.Label(cfg_frame, text="Ablation-specific parameters", font=("TkDefaultFont", 10, "bold"))
        self.ablation_specific_row_start = row

        self.potha_problem_label = ttk.Label(cfg_frame, text="Impostors per problem")
        self.potha_problem_entry = ttk.Entry(cfg_frame, textvariable=self.impostors_per_problem)
        self.potha_round_label = ttk.Label(cfg_frame, text="Impostors per round")
        self.potha_round_entry = ttk.Entry(cfg_frame, textvariable=self.impostors_per_round)

        self.std_feature_label = ttk.Label(cfg_frame, text="Feature type")
        self.std_feature_combo = ttk.Combobox(cfg_frame, textvariable=self.std_feature_type, state="readonly", values=["char", "word"])
        self.std_char_label = ttk.Label(cfg_frame, text="Char n")
        self.std_char_entry = ttk.Entry(cfg_frame, textvariable=self.std_char_n)
        self.std_lower_check = ttk.Checkbutton(cfg_frame, text="Lowercase TF-STD", variable=self.std_lowercase)

        cfg_frame.grid_columnconfigure(1, weight=1)

        self._build_document_panel(docs_frame)
        self._build_result_panel()

    def _build_document_panel(self, docs_frame):
        docs_frame.grid_columnconfigure(0, weight=1)
        docs_frame.grid_rowconfigure(3, weight=1)
        docs_frame.grid_rowconfigure(7, weight=1)

        ttk.Label(docs_frame, text="Disputed document", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ttk.Button(docs_frame, text="Choose disputed file", command=self._choose_disputed_file).grid(
            row=1, column=0, sticky="w", padx=8, pady=4
        )
        self.disputed_file_label = ttk.Label(docs_frame, text="No file selected")
        self.disputed_file_label.grid(row=2, column=0, sticky="w", padx=8, pady=2)
        self.disputed_text = Text(docs_frame, height=10, wrap="word")
        self.disputed_text.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)

        ttk.Label(docs_frame, text="Candidate document", font=("TkDefaultFont", 10, "bold")).grid(
            row=4, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ttk.Button(docs_frame, text="Choose candidate file", command=self._choose_candidate_file).grid(
            row=5, column=0, sticky="w", padx=8, pady=4
        )
        self.candidate_file_label = ttk.Label(docs_frame, text="No file selected")
        self.candidate_file_label.grid(row=6, column=0, sticky="w", padx=8, pady=2)
        self.candidate_text = Text(docs_frame, height=10, wrap="word")
        self.candidate_text.grid(row=7, column=0, sticky="nsew", padx=8, pady=4)

        note = ttk.Label(
            docs_frame,
            text="Input priority: if a file is selected, its content is used; otherwise pasted text is used.",
        )
        note.grid(row=8, column=0, sticky="w", padx=8, pady=6)

    def _build_result_panel(self):
        controls_frame = ttk.Frame(self.root)
        controls_frame.grid(row=3, column=0, columnspan=2, padx=12, pady=8, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(controls_frame, text="Start", command=self._on_start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.next_btn = ttk.Button(controls_frame, text="Next test", command=self._on_next_test)
        self.next_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        progress_frame = ttk.Frame(self.root)
        progress_frame.grid(row=4, column=0, columnspan=2, padx=12, pady=4, sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", pady=4)
        self.status_label = ttk.Label(progress_frame, text="Idle")
        self.status_label.grid(row=1, column=0, sticky="w")

        result_frame = ttk.LabelFrame(self.root, text="Result")
        result_frame.grid(row=5, column=0, columnspan=2, padx=12, pady=(4, 12), sticky="ew")
        result_frame.grid_columnconfigure(0, weight=1)
        self.result_text = ttk.Label(result_frame, text="No result yet.", justify="left")
        self.result_text.grid(row=0, column=0, sticky="w", padx=8, pady=8)

    @staticmethod
    def _add_int_input(frame, row, label, var):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)

    @staticmethod
    def _add_float_input(frame, row, label, var):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)

    def _refresh_dynamic_controls(self):
        show_ablation = self.method_mode.get() == "ablations"
        if show_ablation:
            self.ablation_label.grid(row=self.ablation_row, column=0, sticky="w", padx=8, pady=4)
            self.ablation_combo.grid(row=self.ablation_row, column=1, sticky="ew", padx=8, pady=4)
        else:
            self.ablation_label.grid_remove()
            self.ablation_combo.grid_remove()

        for widget in [
            self.ablation_specific_title,
            self.potha_problem_label,
            self.potha_problem_entry,
            self.potha_round_label,
            self.potha_round_entry,
            self.std_feature_label,
            self.std_feature_combo,
            self.std_char_label,
            self.std_char_entry,
            self.std_lower_check,
        ]:
            widget.grid_remove()

        if not show_ablation:
            return

        base_row = self.ablation_specific_row_start
        self.ablation_specific_title.grid(row=base_row, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 2))
        base_row += 1

        if self.ablation.get() == "Potha2017":
            self.potha_problem_label.grid(row=base_row, column=0, sticky="w", padx=8, pady=4)
            self.potha_problem_entry.grid(row=base_row, column=1, sticky="ew", padx=8, pady=4)
            base_row += 1
            self.potha_round_label.grid(row=base_row, column=0, sticky="w", padx=8, pady=4)
            self.potha_round_entry.grid(row=base_row, column=1, sticky="ew", padx=8, pady=4)
        elif self.ablation.get() == "StdImpostor":
            self.std_feature_label.grid(row=base_row, column=0, sticky="w", padx=8, pady=4)
            self.std_feature_combo.grid(row=base_row, column=1, sticky="ew", padx=8, pady=4)
            base_row += 1
            self.std_char_label.grid(row=base_row, column=0, sticky="w", padx=8, pady=4)
            self.std_char_entry.grid(row=base_row, column=1, sticky="ew", padx=8, pady=4)
            base_row += 1
            self.std_lower_check.grid(row=base_row, column=0, columnspan=2, sticky="w", padx=8, pady=4)

    def _choose_disputed_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt *.md *.text"), ("All files", "*.*")])
        if path:
            self.disputed_file_path.set(path)
            self.disputed_file_label.config(text=path)

    def _choose_candidate_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt *.md *.text"), ("All files", "*.*")])
        if path:
            self.candidate_file_path.set(path)
            self.candidate_file_label.config(text=path)

    def _collect_inputs(self) -> Tuple[str, str]:
        disputed = ""
        candidate = ""

        if self.disputed_file_path.get():
            disputed = _read_file_content(self.disputed_file_path.get()).strip()
        else:
            disputed = self.disputed_text.get("1.0", END).strip()

        if self.candidate_file_path.get():
            candidate = _read_file_content(self.candidate_file_path.get()).strip()
        else:
            candidate = self.candidate_text.get("1.0", END).strip()

        return disputed, candidate

    def _build_param_dicts(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        common_params = {
            "rounds": int(self.rounds.get()),
            "top_n": int(self.top_n.get()),
            "portion_delete": float(self.portion_delete.get()),
            "n_impostors": int(self.n_impostors.get()),
            "threshold": float(self.threshold.get()),
            "impostor_technique": TECHNIQUE_LABEL_TO_VALUE[self.technique_label.get()],
            "dataset_name": self.dataset_name.get(),
            "min_n_tokens": int(self.min_n_tokens.get()),
            "upsample": bool(self.upsample.get()),
        }
        ablation_params: Dict[str, Any] = {}
        if self.ablation.get() == "Potha2017":
            ablation_params["impostors_per_problem"] = int(self.impostors_per_problem.get())
            ablation_params["impostors_per_round"] = int(self.impostors_per_round.get())
        elif self.ablation.get() == "StdImpostor":
            ablation_params["feature_type"] = self.std_feature_type.get()
            ablation_params["char_n"] = int(self.std_char_n.get())
            ablation_params["lowercase"] = bool(self.std_lowercase.get())
        return common_params, ablation_params

    def _run_detection(self, disputed_text: str, candidate_text: str, common_params: Dict[str, Any], ablation_params: Dict[str, Any]):
        method_mode = self.method_mode.get()
        ablation_name = self.ablation.get() if method_mode == "ablations" else None

        detector = _build_detector(method_mode, ablation_name, common_params, ablation_params)
        disputed_id = _ensure_text_in_mongodb(detector, disputed_text, common_params["dataset_name"])
        candidate_id = _ensure_text_in_mongodb(detector, candidate_text, common_params["dataset_name"])

        score_arr = detector.get_score([str(disputed_id), str(candidate_id)])
        if score_arr is None or len(score_arr) == 0:
            raise RuntimeError("Detector did not return a score.")

        score = float(score_arr[0])
        same_author = bool(score > common_params["threshold"])
        return {
            "score": score,
            "same_author": same_author,
            "threshold": common_params["threshold"],
            "disputed_id": str(disputed_id),
            "candidate_id": str(candidate_id),
            "method_mode": method_mode,
            "ablation": ablation_name,
            "technique": common_params["impostor_technique"],
        }

    def _on_start(self):
        disputed_text, candidate_text = self._collect_inputs()
        if not disputed_text or not candidate_text:
            messagebox.showerror(
                "Missing input",
                "Please provide both disputed and candidate documents (file or text).",
            )
            return

        try:
            common_params, ablation_params = self._build_param_dicts()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.start_btn.config(state="disabled")
        self.next_btn.config(state="disabled")
        self.status_label.config(text="Running impostor method...")
        self.progress_value = 0
        self.progress["value"] = 0
        self.worker_result = None
        self.worker_error = None

        def target():
            try:
                self.worker_result = self._run_detection(disputed_text, candidate_text, common_params, ablation_params)
            except Exception as exc:
                self.worker_error = str(exc)

        self.worker_thread = threading.Thread(target=target, daemon=True)
        self.worker_thread.start()
        self._poll_worker()

    def _poll_worker(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.progress_value = min(self.progress_value + 2, 95)
            self.progress["value"] = self.progress_value
            self.status_label.config(text=f"Processing... {self.progress_value}%")
            self.root.after(200, self._poll_worker)
            return

        self.progress["value"] = 100
        if self.worker_error:
            self.status_label.config(text="Run failed.")
            messagebox.showerror("Run failed", self.worker_error)
            self.result_text.config(text="No result available due to error.")
        else:
            self.status_label.config(text="Done.")
            result = self.worker_result
            summary = (
                f"Impostor score: {result['score']:.4f}\n"
                f"Same author: {'Yes' if result['same_author'] else 'No'}\n"
                f"Threshold: {result['threshold']:.2f}\n"
                f"Technique: {result['technique']}\n"
                f"Method family: {result['method_mode']}\n"
                f"Ablation: {result['ablation'] or '-'}\n"
                f"Disputed ID: {result['disputed_id']}\n"
                f"Candidate ID: {result['candidate_id']}"
            )
            self.result_text.config(text=summary)

        self.start_btn.config(state="normal")
        self.next_btn.config(state="normal")

    def _on_next_test(self):
        self.disputed_file_path.set("")
        self.candidate_file_path.set("")
        self.disputed_file_label.config(text="No file selected")
        self.candidate_file_label.config(text="No file selected")
        self.disputed_text.delete("1.0", END)
        self.candidate_text.delete("1.0", END)
        self.progress["value"] = 0
        self.progress_value = 0
        self.status_label.config(text="Idle")
        self.result_text.config(text="No result yet.")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ImpostorUI()
    app.run()
