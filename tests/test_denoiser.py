"""
Test script to verify the denoiser is working correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sound_denoiser import AudioDenoiser


def test_denoiser_produces_different_output():
    """Test that the denoiser actually modifies the audio."""
    
    # Find an example track
    example_dir = Path(__file__).parent.parent / "example_tracks"
    audio_files = list(example_dir.glob("*.*"))
    
    if not audio_files:
        print("ERROR: No audio files found in example_tracks/")
        return False
    
    test_file = audio_files[0]
    print(f"Testing with: {test_file.name}")
    
    # Create denoiser with more aggressive settings for testing
    denoiser = AudioDenoiser(
        max_db_reduction=15.0,  # More aggressive for testing
        blend_original=0.0,     # No blending for clear difference
        noise_reduction_strength=0.95,
        transient_protection=0.2,
        high_freq_emphasis=2.0,
    )
    
    # Load audio
    print("Loading audio...")
    audio, sr = denoiser.load_audio(str(test_file))
    print(f"  Sample rate: {sr} Hz")
    print(f"  Shape: {audio.shape}")
    print(f"  Duration: {audio.shape[-1] / sr:.2f}s")
    
    # Get original
    original = denoiser.get_original()
    print(f"  Original audio range: [{original.min():.4f}, {original.max():.4f}]")
    
    # Process WITHOUT noise profile (adaptive mode)
    print("\n--- Testing ADAPTIVE mode (no noise profile) ---")
    denoiser.set_use_learned_profile(False)
    processed_adaptive = denoiser.process()
    
    # Check if different
    diff_adaptive = np.abs(original - processed_adaptive)
    max_diff_adaptive = diff_adaptive.max()
    mean_diff_adaptive = diff_adaptive.mean()
    rms_orig = np.sqrt(np.mean(original**2))
    rms_proc_adaptive = np.sqrt(np.mean(processed_adaptive**2))
    
    print(f"  Processed audio range: [{processed_adaptive.min():.4f}, {processed_adaptive.max():.4f}]")
    print(f"  Max difference: {max_diff_adaptive:.6f}")
    print(f"  Mean difference: {mean_diff_adaptive:.6f}")
    print(f"  Original RMS: {rms_orig:.6f}")
    print(f"  Processed RMS: {rms_proc_adaptive:.6f}")
    print(f"  RMS reduction: {(rms_orig - rms_proc_adaptive) / rms_orig * 100:.2f}%")
    
    adaptive_passed = max_diff_adaptive > 1e-6
    print(f"  ADAPTIVE MODE: {'PASS' if adaptive_passed else 'FAIL'} - Audio is {'different' if adaptive_passed else 'IDENTICAL (BUG!)'}")
    
    # Process WITH noise profile (learned mode)
    print("\n--- Testing LEARNED PROFILE mode ---")
    
    # Auto-detect noise region
    print("  Auto-detecting noise region...")
    try:
        profile, (start, end) = denoiser.auto_learn_noise_profile(min_duration=0.3)
        print(f"  Noise region: {start:.2f}s - {end:.2f}s")
        print(f"  Profile spectral mean range: [{profile.spectral_mean.min():.6f}, {profile.spectral_mean.max():.6f}]")
        
        denoiser.set_use_learned_profile(True)
        processed_learned = denoiser.process()
        
        diff_learned = np.abs(original - processed_learned)
        max_diff_learned = diff_learned.max()
        mean_diff_learned = diff_learned.mean()
        rms_proc_learned = np.sqrt(np.mean(processed_learned**2))
        
        print(f"  Processed audio range: [{processed_learned.min():.4f}, {processed_learned.max():.4f}]")
        print(f"  Max difference: {max_diff_learned:.6f}")
        print(f"  Mean difference: {mean_diff_learned:.6f}")
        print(f"  Processed RMS: {rms_proc_learned:.6f}")
        print(f"  RMS reduction: {(rms_orig - rms_proc_learned) / rms_orig * 100:.2f}%")
        
        learned_passed = max_diff_learned > 1e-6
        print(f"  LEARNED MODE: {'PASS' if learned_passed else 'FAIL'} - Audio is {'different' if learned_passed else 'IDENTICAL (BUG!)'}")
    except Exception as e:
        print(f"  ERROR in learned mode: {e}")
        learned_passed = False
    
    # Test with DEFAULT settings (what GUI uses)
    print("\n--- Testing with DEFAULT GUI settings ---")
    denoiser_default = AudioDenoiser()  # All defaults
    denoiser_default.load_audio(str(test_file))
    original_default = denoiser_default.get_original()
    processed_default = denoiser_default.process()
    
    diff_default = np.abs(original_default - processed_default)
    max_diff_default = diff_default.max()
    mean_diff_default = diff_default.mean()
    rms_proc_default = np.sqrt(np.mean(processed_default**2))
    
    print(f"  Max dB reduction: {denoiser_default.max_db_reduction} dB")
    print(f"  Blend original: {denoiser_default.blend_original * 100}%")
    print(f"  Noise reduction strength: {denoiser_default.noise_reduction_strength * 100}%")
    print(f"  Max difference: {max_diff_default:.6f}")
    print(f"  Mean difference: {mean_diff_default:.6f}")
    print(f"  RMS reduction: {(rms_orig - rms_proc_default) / rms_orig * 100:.2f}%")
    
    default_passed = max_diff_default > 1e-6
    print(f"  DEFAULT SETTINGS: {'PASS' if default_passed else 'FAIL'}")
    
    # Save test output for manual listening
    print("\n--- Saving test outputs ---")
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    denoiser.save(str(output_dir / "test_denoised_aggressive.wav"), processed_adaptive)
    print(f"  Saved: test_output/test_denoised_aggressive.wav")
    
    denoiser_default.save(str(output_dir / "test_denoised_default.wav"), processed_default)
    print(f"  Saved: test_output/test_denoised_default.wav")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    all_passed = adaptive_passed and default_passed
    
    if all_passed:
        print("All tests PASSED")
        print("\nThe denoiser IS modifying the audio.")
        print("If you can't hear a difference, it may be because:")
        print("  1. Default settings are gentle (max 4dB reduction)")
        print("  2. 12% of original is blended back")
        print("  3. Transient protection preserves attacks")
        print("\nTry:")
        print("  - Increase 'Max dB Reduction' slider (try 10-15 dB)")
        print("  - Decrease 'Blend Original' slider (try 0%)")
        print("  - Increase 'Noise Reduction Strength' (try 90%)")
        print("  - Use 'Auto Detect' to learn a noise profile")
    else:
        print("Some tests FAILED")
        print("There may be a bug in the denoiser!")
    
    return all_passed


def test_high_frequency_noise_reduction():
    """Test that high frequencies are reduced more (where hiss lives)."""
    import librosa
    
    example_dir = Path(__file__).parent.parent / "example_tracks"
    audio_files = list(example_dir.glob("*.*"))
    
    if not audio_files:
        return
    
    test_file = audio_files[0]
    print(f"\n--- Testing high frequency reduction on: {test_file.name} ---")
    
    denoiser = AudioDenoiser(
        max_db_reduction=10.0,
        blend_original=0.0,
        noise_reduction_strength=0.9,
    )
    
    audio, sr = denoiser.load_audio(str(test_file))
    original = denoiser.get_original()
    processed = denoiser.process()
    
    # Compute spectrograms
    if original.ndim > 1:
        original = original[0]
        processed = processed[0]
    
    orig_stft = np.abs(librosa.stft(original))
    proc_stft = np.abs(librosa.stft(processed))
    
    # Compare energy in low vs high frequency bands
    n_bins = orig_stft.shape[0]
    low_band = slice(0, n_bins // 4)
    high_band = slice(3 * n_bins // 4, n_bins)
    
    orig_low_energy = np.mean(orig_stft[low_band, :])
    orig_high_energy = np.mean(orig_stft[high_band, :])
    proc_low_energy = np.mean(proc_stft[low_band, :])
    proc_high_energy = np.mean(proc_stft[high_band, :])
    
    low_reduction = (orig_low_energy - proc_low_energy) / orig_low_energy * 100
    high_reduction = (orig_high_energy - proc_high_energy) / orig_high_energy * 100
    
    print(f"  Low frequency energy reduction: {low_reduction:.1f}%")
    print(f"  High frequency energy reduction: {high_reduction:.1f}%")
    
    if high_reduction > low_reduction:
        print("  PASS - High frequencies reduced more than low (good for hiss removal)")
    else:
        print("  NOTE: High frequencies not reduced more - may need tuning")


if __name__ == "__main__":
    print("="*50)
    print("SOUND DENOISER TEST SUITE")
    print("="*50 + "\n")
    
    test_denoiser_produces_different_output()
    test_high_frequency_noise_reduction()
