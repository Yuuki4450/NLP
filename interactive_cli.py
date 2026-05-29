import json

from hybrid_path_nlp import HybridPathNLP
from mdp import run_navigation

def main() -> None:
    parser = HybridPathNLP()
    print("Path NLP interactive CLI")
    print("输入中文路径请求，系统会输出 NLP 解析 JSON。")
    print("输入 exit 或 quit 退出。")
    status = parser.model_status()
    if status["model_loaded"]:
        print(f"loaded slot tagger from {status['model_path']}")
    else:
        print(status.get("warning") or "slot_tagger.pkl not found, using fallback-heavy mode")

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            print("请输入中文路径请求，或输入 exit / quit 退出。")
            continue

        debug = parser.parse_with_debug(text)
        result = debug["result"]
        debug_info = {
            "used_fallback": debug.get("used_fallback"),
            "slot_source": debug.get("slot_source"),
            "fallback_reason": debug.get("fallback_reason"),
            "model_slots": debug.get("model_slots"),
            "model_avoid_points": debug.get("model_avoid_points"),
            "model_loaded": debug.get("model_loaded"),
            "model_path": debug.get("model_path"),
            "warning": debug.get("warning"),
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["is_complete"]:
            print("\nStarting MDP Navigation...")
            run_navigation(result)
        else:
            print("\nNavigation instruction incomplete")
            print("Missing:", result["missing_slots"])
        print("\nDebug:")
        print(json.dumps(debug_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
