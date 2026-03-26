"""
Quick validation that efficiency tracking works for VRT.
Tests on a tiny subset (1 sequence) to verify the modifications don't break anything.
"""
import subprocess
import sys
import os

def test_vrt_rb():
    """Test VRT RB with efficiency tracking"""
    print("Testing VRT RB efficiency tracking...")
    cmd = [
        'conda', 'run', '-n', 'cvpr', 'python',
        '/data/yc/KAIR-master/main_test_vrt_rb.py',
        '--opt', '/data/yc/KAIR-master/options/vrt/010_train_vrt_videosr_rb.json',
        '--checkpoint', '/data/yc/KAIR-master/experiments/010_train_vrt_videosr_rb/models/20000_G.pth',
        '--output-dir', '/tmp/vrt_rb_test',
        '--device', 'cuda:0'
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print("✓ VRT RB test passed")

        # Check if efficiency file was created
        eff_file = '/tmp/vrt_rb_test/VRT_RB_efficiency.json'
        if os.path.exists(eff_file):
            print(f"✓ Efficiency file created: {eff_file}")
            return True
        else:
            print(f"✗ Efficiency file not found: {eff_file}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ VRT RB test timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✗ VRT RB test failed:")
        print(e.stderr)
        return False

def test_vrt_kf256():
    """Test VRT KF256 with efficiency tracking"""
    print("\nTesting VRT KF256 efficiency tracking...")
    cmd = [
        'conda', 'run', '-n', 'cvpr', 'python',
        '/data/yc/KAIR-master/main_test_vrt_kf256.py',
        '--opt', '/data/yc/KAIR-master/options/vrt/011_train_vrt_videosr_kf256.json',
        '--checkpoint', '/data/yc/KAIR-master/experiments/011_train_vrt_videosr_kf256/models/20000_G.pth',
        '--output-dir', '/tmp/vrt_kf256_test',
        '--device', 'cuda:0'
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print("✓ VRT KF256 test passed")

        # Check if efficiency file was created
        eff_file = '/tmp/vrt_kf256_test/VRT_KF256_efficiency.json'
        if os.path.exists(eff_file):
            print(f"✓ Efficiency file created: {eff_file}")
            return True
        else:
            print(f"✗ Efficiency file not found: {eff_file}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ VRT KF256 test timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✗ VRT KF256 test failed:")
        print(e.stderr)
        return False

if __name__ == '__main__':
    print("="*60)
    print("Validating VRT efficiency tracking modifications")
    print("="*60)

    results = []
    results.append(("VRT RB", test_vrt_rb()))
    results.append(("VRT KF256", test_vrt_kf256()))

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
