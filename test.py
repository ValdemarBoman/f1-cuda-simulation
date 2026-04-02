#!/usr/bin/env python3
"""
Quick test to verify the Python F1 simulation is working
"""

import csv
import math
from pathlib import Path
from sac import SAC

def test_setup():
    """Verify all components are working"""
    
    print("Testing Python F1 Simulation Setup")
    print("=" * 50)
    
    # Test 1: Track loading
    print("\n1. Loading track...", end=" ")
    track = []
    with open('Silverstone.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and len(row) >= 4:
                track.append(row)
    print(f"✓ {len(track)} points loaded")
    
    # Test 2: SAC initialization
    print("2. Initializing SAC agent...", end=" ")
    sac = SAC(8, 2, 1234)
    print("✓ Done")
    
    # Test 3: Action sampling
    print("3. Testing action sampling...", end=" ")
    obs = [0.5] * 8
    action = sac.act(obs, deterministic=False)
    print(f"✓ Action: {action}")
    
    # Test 4: Replay buffer and training
    print("4. Testing training loop...", end=" ")
    Path("logs").mkdir(exist_ok=True)
    for i in range(50):
        sac.store(obs, action, 1.0, obs, False)
        sac.update_many(1)
    print(f"✓ {sac.replay_size()} transitions stored")
    
    # Test 5: GPU availability
    import torch
    gpu_available = torch.cuda.is_available()
    print(f"5. GPU Support: {'✓ CUDA available' if gpu_available else '⚠ CPU only'}")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("\nReady to run: python3 main.py")
    print("=" * 50)

if __name__ == "__main__":
    test_setup()
