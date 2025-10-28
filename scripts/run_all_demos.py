"""
Runner script for all parallel coordinates demos.
Run with: uv run python scripts/run_all_demos.py [demo_name]
"""

import sys
import os
import subprocess

# Available demo scripts
DEMOS = {
    "iris_vertical": "demo_iris_vertical.py",
    "iris_horizontal": "demo_iris_horizontal.py",
    "iris_combined": "demo_iris_combined.py",
    "iris_parallel": "demo_iris_parallel.py",
    "tips": "demo_tips.py",
    "normalization": "demo_normalization.py",
    "custom_styling": "demo_custom_styling.py",
    "categorical_axes": "demo_categorical_axes.py",
    "comparison": "demo_comparison.py",
    "scaling_verification": "demo_scaling_verification.py",
}


def run_demo(script_name):
    """Run a specific demo script."""
    script_path = os.path.join("scripts", script_name)
    try:
        result = subprocess.run(
            ["uv", "run", "python", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run {script_name}")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Run all demos or a specific demo."""
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        if demo_name in DEMOS:
            print(f"🚀 Running {demo_name} demo...")
            success = run_demo(DEMOS[demo_name])
            if success:
                print(f"✅ {demo_name} demo completed!")
            else:
                print(f"❌ {demo_name} demo failed!")
        else:
            print(f"❌ Unknown demo: {demo_name}")
            print(f"Available demos: {', '.join(DEMOS.keys())}")
            sys.exit(1)
    else:
        print("🚀 Running all parallel coordinates demos...\n")

        success_count = 0
        total_demos = len(DEMOS)

        for demo_name, script_name in DEMOS.items():
            print(f"--- Running {demo_name} ---")
            if run_demo(script_name):
                success_count += 1
                print(f"✅ {demo_name} completed\n")
            else:
                print(f"❌ {demo_name} failed\n")

        print(f"📊 Results: {success_count}/{total_demos} demos completed successfully")

        if success_count == total_demos:
            print("🎉 All demos completed successfully!")
        else:
            print(f"⚠️  {total_demos - success_count} demo(s) failed")


if __name__ == "__main__":
    main()
