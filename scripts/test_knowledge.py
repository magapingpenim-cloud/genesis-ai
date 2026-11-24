"""Smoke-test local knowledge retrieval from the repository root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis.knowledge import find_knowledge

CASES = [
    ("Объясни простыми словами, что такое гравитация", "Гравитация"),
    ("Для чего нужен GPU в нейросетях?", "GPU"),
    ("Что такое фотосинтез у растений?", "Фотосинтез"),
    ("Назови столицу Азербайджана", "Баку"),
    ("Что такое переобучение модели?", "Переобучение"),
]


def main():
    failed = 0
    for query, expected_title in CASES:
        result = find_knowledge(query)
        ok = bool(result and result["title"] == expected_title)
        print(("PASS" if ok else "FAIL") + f" | {query}")
        if result:
            print(f"  matched: {result['title']} (score={result['score']})")
        else:
            print("  matched: nothing")
        failed += not ok
    print(f"\nKnowledge checks: {len(CASES) - failed}/{len(CASES)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
