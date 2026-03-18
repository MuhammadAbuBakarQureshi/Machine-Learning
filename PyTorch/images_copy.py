"""
merge_takeout.py
----------------
Merges multiple Google Takeout exports into one clean folder.

Structure expected:
  SOURCE_ROOT\
    takeout-...-001\Takeout\Google Photos\Bin\
    takeout-...-001\Takeout\Google Photos\Photos from 2023\
    takeout-...-002\Takeout\Google Photos\Bin\
    takeout-...-002\Takeout\Google Photos\Photos from 2023\
    ...

Result:
  OUTPUT_FOLDER\
    Bin\              ← merged from all takeout folders
    Photos from 2023\ ← merged from all takeout folders
    Photos from 2024\
    ...

Rules:
  - COPY only, originals are never touched
  - Same-named folders are MERGED together
  - Duplicate filenames get _1, _2 ... suffix
  - Voice announces at major milestones only
  - Print updates on every file
"""

import os
import shutil
import subprocess
import sys

# ── CONFIG ───────────────────────────────────────────────────────────────────
SOURCE_ROOT   = r"D:\Media\Pictures\Photos\Google Photos"
OUTPUT_FOLDER = r"D:\Media\Pictures\Photos\Takeout_Merged"

# The fixed path inside each takeout zip after extraction
# takeout-XXXXX\Takeout\Google Photos\  <-- we go into this
INNER_PATH = os.path.join("Takeout", "Google Photos")

MEDIA_EXTENSIONS = {
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".tiff", ".tif", ".heic", ".heif", ".raw", ".cr2",
    ".nef", ".arw", ".dng",
    # Videos
    ".mp4", ".mov", ".avi", ".mkv", ".wmv",
    ".3gp", ".m4v", ".flv", ".webm",
}

VOICE_EVERY_N = 300   # speak every N files copied
# ─────────────────────────────────────────────────────────────────────────────


def speak(text: str):
    """Non-blocking Windows TTS — no install needed."""
    escaped = text.replace("'", "''")
    ps_cmd = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{escaped}')"
    )
    subprocess.Popen(
        ["powershell", "-WindowStyle", "Hidden", "-Command", ps_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_unique_path(dest_folder: str, filename: str) -> str:
    """Append _1, _2 … if a file with the same name already exists."""
    dest = os.path.join(dest_folder, filename)
    if not os.path.exists(dest):
        return dest
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        dest = os.path.join(dest_folder, new_name)
        if not os.path.exists(dest):
            return dest
        counter += 1


def find_google_photos_dirs(source_root: str) -> list:
    """
    Find all  takeout-XXXXX\Takeout\Google Photos  directories
    directly inside source_root.
    """
    found = []
    for entry in os.scandir(source_root):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("takeout-"):
            continue
        gp_path = os.path.join(entry.path, INNER_PATH)
        if os.path.isdir(gp_path):
            found.append(gp_path)
        else:
            print(f"[WARN] No 'Takeout\\Google Photos' inside: {entry.name}")
    return found


def collect_jobs(google_photos_dirs: list) -> list:
    """
    Walk every Google Photos dir and collect (src_file, dest_subfolder_name) tuples.
    dest_subfolder_name is the immediate subfolder under Google Photos
    e.g. 'Bin', 'Photos from 2023', 'Hiking' …
    Files sitting directly under Google Photos (no subfolder) go to OUTPUT root.
    """
    jobs = []  # list of (src_path, dest_folder_path)

    for gp_dir in google_photos_dirs:
        for dirpath, dirnames, filenames in os.walk(gp_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in MEDIA_EXTENSIONS:
                    continue

                src_path = os.path.join(dirpath, fname)

                # relative path from Google Photos root
                rel = os.path.relpath(dirpath, gp_dir)

                if rel == ".":
                    # File is directly inside Google Photos (no subfolder)
                    dest_folder = OUTPUT_FOLDER
                else:
                    # Keep the FULL relative sub-path so nested albums stay nested
                    dest_folder = os.path.join(OUTPUT_FOLDER, rel)

                jobs.append((src_path, dest_folder))

    return jobs


def main():
    # ── Validate ─────────────────────────────────────────────────────────────
    if not os.path.isdir(SOURCE_ROOT):
        print(f"[ERROR] Source folder not found:\n  {SOURCE_ROOT}")
        speak("Error. Source folder not found. Please check the path in the script.")
        sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"[INFO] Source root  : {SOURCE_ROOT}")
    print(f"[INFO] Output folder: {OUTPUT_FOLDER}")
    print()

    # ── Find takeout folders ──────────────────────────────────────────────────
    print("[STEP 1] Locating takeout folders …")
    speak("Locating takeout folders. Please wait.")

    gp_dirs = find_google_photos_dirs(SOURCE_ROOT)

    if not gp_dirs:
        print("[ERROR] No takeout-XXXXX folders with Takeout\\Google Photos found.")
        speak("No takeout folders found. Check the source path.")
        sys.exit(1)

    print(f"[INFO] Found {len(gp_dirs)} takeout export(s):")
    for d in gp_dirs:
        print(f"       {d}")
    print()

    # ── Collect all copy jobs ─────────────────────────────────────────────────
    print("[STEP 2] Scanning all files …")
    speak(f"Found {len(gp_dirs)} takeout exports. Scanning files now.")

    jobs = collect_jobs(gp_dirs)
    total = len(jobs)

    print(f"[INFO] Total media files to copy: {total:,}")
    print()

    if total == 0:
        print("[WARN] No media files found. Check extensions or folder structure.")
        speak("No media files found. Please check the folder structure.")
        return

    speak(f"Scan complete. {total} files ready to copy. Starting now.")

    # ── Preview destination folders ───────────────────────────────────────────
    dest_folders = sorted(set(j[1] for j in jobs))
    print(f"[INFO] Destination subfolders that will be created ({len(dest_folders)}):")
    for f in dest_folders:
        rel = os.path.relpath(f, OUTPUT_FOLDER)
        print(f"       {rel if rel != '.' else '(root)'}")
    print()

    # ── Copy ──────────────────────────────────────────────────────────────────
    print("[STEP 3] Copying files …")
    copied = 0
    errors = 0
    next_voice_at = VOICE_EVERY_N

    for src_path, dest_folder in jobs:
        os.makedirs(dest_folder, exist_ok=True)
        fname = os.path.basename(src_path)
        dest_path = make_unique_path(dest_folder, fname)

        try:
            shutil.copy2(src_path, dest_path)
            copied += 1
            dest_rel = os.path.relpath(dest_path, OUTPUT_FOLDER)
            print(f"[{copied}/{total}] {dest_rel}")

            if copied >= next_voice_at:
                pct = int(copied / total * 100)
                speak(f"{copied} files copied. {pct} percent complete.")
                next_voice_at += VOICE_EVERY_N

        except PermissionError as e:
            errors += 1
            print(f"[SKIP – Permission Denied] {fname}: {e}")
        except Exception as e:
            errors += 1
            print(f"[ERROR] {fname}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ALL DONE!")
    print(f"  Takeout exports  : {len(gp_dirs)}")
    print(f"  Total files found: {total:,}")
    print(f"  Copied           : {copied:,}")
    print(f"  Errors           : {errors:,}")
    print(f"  Output folder    : {OUTPUT_FOLDER}")
    print("=" * 60)
    speak(
        f"All done! {copied} files copied from {len(gp_dirs)} takeout exports. "
        f"{errors} errors. Check the output folder."
    )


if __name__ == "__main__":
    main()