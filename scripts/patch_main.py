#!/usr/bin/env python3
import sys
import os

# ---------------------------------------------------------------------------
# Resolve target path
# ---------------------------------------------------------------------------
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir    = os.path.dirname(scripts_dir)
    target = os.path.join(
        base_dir,
        "tools", "pytorch-examples", "word_language_model", "main.py"
    )

if not os.path.isfile(target):
    print(f"ERROR: main.py not found at: {target}")
    print("Run ./scripts/install_packages.sh first, or pass the path explicitly.")
    sys.exit(1)

print(f"  Target: {target}")

with open(target, 'r', encoding='utf-8') as f:
    src = f.read()

# ---------------------------------------------------------------------------
# Idempotency guard
# ---------------------------------------------------------------------------
SENTINEL = '# [MT-EX2-PATCHED-V3]'
if SENTINEL in src:
    print(f"  Already patched — nothing to do.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Patch definitions
# ---------------------------------------------------------------------------

# Patch 0: Add --export-log argument to argparse, anchored after --dry-run
OLD_0 = ("parser.add_argument('--dry-run', action='store_true',\n"
         "                    help='verify the code and the model')")
NEW_0 = ("parser.add_argument('--dry-run', action='store_true',\n"
         "                    help='verify the code and the model')\n"
         "parser.add_argument('--export-log', type=str, default='',\n"
         "                    help='path to write per-epoch CSV log (epoch,train_ppl,valid_ppl)')"
         "  # [MT-EX2-PATCHED-V3]")

# Patch 1: CSV writer setup injected right after "best_val_loss = None"
# Handles clean upstream, V1-patched, and V2-patched states.
OLD_1_CLEAN = 'best_val_loss = None'
OLD_1_V1 = ('best_val_loss = None\n'
            '# [MT-EX2-PATCHED] --------------------------------------------------------\n'
            'import csv as _csv\n'
            '_csv_file   = None\n'
            '_csv_writer = None\n'
            'if args.export_log:\n'
            '    os.makedirs(os.path.dirname(args.export_log) or \'.\', exist_ok=True)\n'
            '    _csv_file   = open(args.export_log, \'w\', newline=\'\', encoding=\'utf-8\')\n'
            '    _csv_writer = _csv.writer(_csv_file)\n'
            '    _csv_writer.writerow([\'epoch\', \'train_ppl\', \'valid_ppl\'])\n'
            '# -------------------------------------------------------------------------')
OLD_1_V2 = ('best_val_loss = None\n'
            '# [MT-EX2-PATCHED-V2] -----------------------------------------------------\n'
            'import csv as _csv\n'
            '_csv_file   = None\n'
            '_csv_writer = None\n'
            'if args.export_log:\n'
            '    os.makedirs(os.path.dirname(args.export_log) or \'.\', exist_ok=True)\n'
            '    _csv_file   = open(args.export_log, \'w\', newline=\'\', encoding=\'utf-8\')\n'
            '    _csv_writer = _csv.writer(_csv_file)\n'
            '    _csv_writer.writerow([\'epoch\', \'train_ppl\', \'valid_ppl\'])\n'
            '# -------------------------------------------------------------------------')

NEW_1 = '''\
best_val_loss = None
# [MT-EX2-PATCHED-V3] -----------------------------------------------------
import csv as _csv
_csv_file   = None
_csv_writer = None
if args.export_log:
    os.makedirs(os.path.dirname(args.export_log) or '.', exist_ok=True)
    _csv_file   = open(args.export_log, 'w', newline='', encoding='utf-8')
    _csv_writer = _csv.writer(_csv_file)
    _csv_writer.writerow(['epoch', 'train_ppl', 'valid_ppl'])
# -------------------------------------------------------------------------'''

# Patch 2: Capture train() return value
OLD_2      = '        train()\n        val_loss = evaluate(val_data)'
OLD_2_DONE = '        _train_loss = train()\n        val_loss = evaluate(val_data)'
NEW_2      = '        _train_loss = train()\n        val_loss = evaluate(val_data)'

# Patch 3: Write CSV row after the validation print block
OLD_3 = ("        print('-' * 89)\n"
         "        # Save the model if the validation loss is the best we've seen so far.")
OLD_3_DONE = ("        print('-' * 89)\n"
              "        if _csv_writer:\n"
              "            _train_ppl = math.exp(min(_train_loss, 20))\n"
              "            _valid_ppl = math.exp(val_loss)\n"
              "            _csv_writer.writerow([epoch, f'{_train_ppl:.4f}', f'{_valid_ppl:.4f}'])\n"
              "            _csv_file.flush()\n"
              "        # Save the model if the validation loss is the best we've seen so far.")
NEW_3 = """\
        print('-' * 89)
        if _csv_writer:
            _train_ppl = math.exp(min(_train_loss, 20))
            _valid_ppl = math.exp(val_loss)
            _csv_writer.writerow([epoch, f'{_train_ppl:.4f}', f'{_valid_ppl:.4f}'])
            _csv_file.flush()
        # Save the model if the validation loss is the best we've seen so far."""

# Patch 4: Convert try/except to try/except/finally so the CSV is always closed
OLD_4      = "    print('-' * 89)\n    print('Exiting from training early')"
OLD_4_DONE = ("    print('-' * 89)\n"
              "    print('Exiting from training early')\n"
              "finally:\n"
              "    if _csv_file:\n"
              "        _csv_file.close()")
NEW_4 = """\
    print('-' * 89)
    print('Exiting from training early')
finally:
    if _csv_file:
        _csv_file.close()"""

# Patch 5: Make train() return the average loss for the epoch.
#
# The original train() resets total_loss to 0 every log_interval batches and
# has no return statement, so _train_loss is None after every call.
# We add a separate epoch-level accumulator that is never reset, and return
# the mean at the end of the function.

OLD_5 = ("def train():\n"
         "    # Turn on training mode which enables dropout.\n"
         "    model.train()\n"
         "    total_loss = 0.\n"
         "    start_time = time.time()")
NEW_5 = ("def train():\n"
         "    # Turn on training mode which enables dropout.\n"
         "    model.train()\n"
         "    total_loss = 0.\n"
         "    _epoch_loss = 0.       # [MT-EX2-PATCHED-V3] epoch-level accumulator\n"
         "    _epoch_batches = 0     # [MT-EX2-PATCHED-V3]\n"
         "    start_time = time.time()")

OLD_5B = "        total_loss += loss.item()"
NEW_5B = ("        total_loss += loss.item()\n"
          "        _epoch_loss += loss.item()    # [MT-EX2-PATCHED-V3]\n"
          "        _epoch_batches += 1           # [MT-EX2-PATCHED-V3]")

OLD_5C = ("        if args.dry_run:\n"
          "            break\n"
          "\n"
          "\n"
          "def export_onnx")
NEW_5C = ("        if args.dry_run:\n"
          "            break\n"
          "    return _epoch_loss / _epoch_batches if _epoch_batches else 0.0"
          "  # [MT-EX2-PATCHED-V3]\n"
          "\n"
          "\n"
          "def export_onnx")

# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------
failed = False

# --- Patch 0: argparse ---
if OLD_0 not in src:
    if "--export-log" in src:
        print(f"  Patch 0: --export-log already in argparse, skipping.")
    else:
        print(f"  ERROR: Patch 0 anchor not found and --export-log missing.")
        failed = True
else:
    src = src.replace(OLD_0, NEW_0, 1)
    print(f"  Patch 0: applied (added --export-log to argparse).")

# --- Patch 1: CSV setup block ---
if OLD_1_V2 in src:
    src = src.replace(OLD_1_V2, NEW_1, 1)
    print(f"  Patch 1: re-applied (replaced V2 sentinel with V3).")
elif OLD_1_V1 in src:
    src = src.replace(OLD_1_V1, NEW_1, 1)
    print(f"  Patch 1: re-applied (replaced V1 sentinel with V3).")
elif OLD_1_CLEAN in src:
    src = src.replace(OLD_1_CLEAN, NEW_1, 1)
    print(f"  Patch 1: applied.")
else:
    print(f"  ERROR: Patch 1 anchor not found.")
    failed = True

# --- Patch 2: capture train() return value ---
if OLD_2_DONE in src:
    print(f"  Patch 2: already applied, skipping.")
elif OLD_2 in src:
    src = src.replace(OLD_2, NEW_2, 1)
    print(f"  Patch 2: applied.")
else:
    print(f"  ERROR: Patch 2 anchor not found.")
    failed = True

# --- Patch 3: CSV row write ---
if OLD_3_DONE in src:
    print(f"  Patch 3: already applied, skipping.")
elif OLD_3 in src:
    src = src.replace(OLD_3, NEW_3, 1)
    print(f"  Patch 3: applied.")
else:
    print(f"  ERROR: Patch 3 anchor not found.")
    failed = True

# --- Patch 4: finally block ---
if OLD_4_DONE in src:
    print(f"  Patch 4: already applied, skipping.")
elif OLD_4 in src:
    src = src.replace(OLD_4, NEW_4, 1)
    print(f"  Patch 4: applied.")
else:
    print(f"  ERROR: Patch 4 anchor not found.")
    failed = True

# --- Patch 5: make train() return average epoch loss ---
if '_epoch_loss' in src:
    print(f"  Patch 5: already applied, skipping.")
else:
    p5_ok = True
    if OLD_5 not in src:
        print(f"  ERROR: Patch 5a anchor (train() header) not found.")
        p5_ok = False
        failed = True
    if OLD_5B not in src:
        print(f"  ERROR: Patch 5b anchor (total_loss accumulation) not found.")
        p5_ok = False
        failed = True
    if OLD_5C not in src:
        print(f"  ERROR: Patch 5c anchor (end of train()) not found.")
        p5_ok = False
        failed = True
    if p5_ok:
        src = src.replace(OLD_5,  NEW_5,  1)
        src = src.replace(OLD_5B, NEW_5B, 1)
        src = src.replace(OLD_5C, NEW_5C, 1)
        print(f"  Patch 5: applied (train() now returns average epoch loss).")

if failed:
    print("\nPatch aborted — file was NOT modified.")
    sys.exit(1)

with open(target, 'w', encoding='utf-8') as f:
    f.write(src)

print(f"\n  main.py patched successfully.")