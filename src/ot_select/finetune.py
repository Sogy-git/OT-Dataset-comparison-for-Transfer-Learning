"""Fine-tune on KNN / OT / OTDD selected MNIST samples (single default configuration)."""

from pipeline import get_device, run_experiment

if __name__ == "__main__":
    device = get_device()
    print(f"Using {device} device")

    run_experiment(
        seed=69,
        semeion_train_ratio=0.10,
        output_dir=".",
        subset_size=5000,
        batch_size=10,
        epochs=5,
        lr=0.0001,
        verbose=True,
    )

    # Also save with legacy filenames for backward compatibility
    import os
    import shutil

    legacy_map = {
        "models/seed69_ratio10/KNN.pth": "fine_tuned_full_cnn_KNN_test.pth",
        "models/seed69_ratio10/OT.pth": "fine_tuned_full_cnn_OT_test.pth",
        "models/seed69_ratio10/OTDD.pth": "fine_tuned_full_cnn_OTDDtest.pth",
    }
    for src, dst in legacy_map.items():
        if os.path.exists(src):
            shutil.copy2(src, dst)
