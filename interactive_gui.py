import json
import sys

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError as error:
    print(f"tkinter is not available in this Python environment: {error}")
    sys.exit(1)

from hybrid_path_nlp import HybridPathNLP


DEFAULT_TEXT = "从蓝色点出发，经过青色点和紫色点，最后到绿色点"


class PathNLPDemo:
    def __init__(self, root):
        self.root = root
        self.parser = HybridPathNLP()
        self.root.title("Path NLP Parser Demo")
        status = self.parser.model_status()
        if status["model_loaded"]:
            status_text = f"loaded slot tagger from {status['model_path']}"
        else:
            status_text = status.get("warning") or "slot_tagger.pkl not found, using fallback-heavy mode"
        self.status_label = tk.Label(root, text=status_text, anchor="w")
        self.status_label.pack(fill=tk.X, padx=10, pady=(8, 0))

        self.input_box = tk.Text(root, height=6, width=80)
        self.input_box.pack(fill=tk.BOTH, padx=10, pady=(10, 6))
        self.input_box.insert("1.0", DEFAULT_TEXT)

        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, padx=10, pady=4)

        parse_button = tk.Button(button_frame, text="Parse", command=self.parse)
        parse_button.pack(side=tk.LEFT, padx=(0, 8))

        clear_button = tk.Button(button_frame, text="Clear", command=self.clear)
        clear_button.pack(side=tk.LEFT)

        self.output_box = tk.Text(root, height=18, width=80)
        self.output_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

    def parse(self):
        text = self.input_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Path NLP Parser Demo", "请输入中文路径请求。")
            return

        debug = self.parser.parse_with_debug(text)
        output = {
            "result": debug["result"],
            "debug": {
                "used_fallback": debug.get("used_fallback"),
                "slot_source": debug.get("slot_source"),
                "fallback_reason": debug.get("fallback_reason"),
                "model_slots": debug.get("model_slots"),
                "model_avoid_points": debug.get("model_avoid_points"),
                "model_loaded": debug.get("model_loaded"),
                "model_path": debug.get("model_path"),
                "warning": debug.get("warning"),
            },
        }
        self.output_box.delete("1.0", tk.END)
        self.output_box.insert("1.0", json.dumps(output, ensure_ascii=False, indent=2))

    def clear(self):
        self.input_box.delete("1.0", tk.END)
        self.output_box.delete("1.0", tk.END)


def main() -> None:
    root = tk.Tk()
    PathNLPDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
