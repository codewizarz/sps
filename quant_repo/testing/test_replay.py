import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.data.replay import MarketReplay


def test_market_replay():
    print("[TEST] Options Market Replay Tool...")

    # 1. Initialize (uses mock data)
    replay = MarketReplay()

    assert len(replay.timestamps) > 0
    print(f"Loaded {len(replay.timestamps)} frames.")

    # 2. Check Initial Frame
    frame0 = replay.get_current_frame()
    assert len(frame0) > 0
    print("Initial Frame Loaded.")

    # 3. Render
    print("Rendering Frame 0:")
    replay.render()

    # 4. Navigation
    replay.next_frame()
    assert replay.current_idx == 1

    print("Rendering Frame 1:")
    replay.render()

    replay.prev_frame()
    assert replay.current_idx == 0

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_market_replay()
