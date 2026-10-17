"""Run the LogiShield analysis service locally: ``python -m tokenization_ai``.

Binds to 127.0.0.1 only (local development / demonstration).
"""

import uvicorn


def main() -> None:
    uvicorn.run("tokenization_ai.app:app", host="127.0.0.1", port=8004)


if __name__ == "__main__":
    main()
