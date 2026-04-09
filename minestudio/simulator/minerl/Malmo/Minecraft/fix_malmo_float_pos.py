#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import difflib

FILES = [
    "src/main/java/com/microsoft/Malmo/MissionHandlers/MissionBehaviour.java",
    "src/main/java/com/microsoft/Malmo/Utils/PositionHelper.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/ObservationFromSubgoalPositionListImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/ObservationFromGridImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/MazeDecoratorImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/AnimationDecoratorImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/MarkingDecoratorImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/RandomizedStartDecoratorImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/SnakeDecoratorImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/NavigationDecoratorImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/ClassroomDecoratorImplementation.java",
    "src/main/java/com/microsoft/Malmo/MissionHandlers/MovingTargetDecoratorImplementation.java",
]

DRY_RUN = False

RECEIVER = r'([A-Za-z_][A-Za-z0-9_]*(?:\([^()]*\))?(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\([^()]*\))?)*)'

RULES = [
    # doubleValue -> (double)
    (re.compile(RECEIVER + r'\.getX\(\)\.doubleValue\(\)'), r'(double) \1.getX()'),
    (re.compile(RECEIVER + r'\.getY\(\)\.doubleValue\(\)'), r'(double) \1.getY()'),
    (re.compile(RECEIVER + r'\.getZ\(\)\.doubleValue\(\)'), r'(double) \1.getZ()'),

    # intValue -> (int)
    (re.compile(RECEIVER + r'\.getX\(\)\.intValue\(\)'), r'(int) \1.getX()'),
    (re.compile(RECEIVER + r'\.getY\(\)\.intValue\(\)'), r'(int) \1.getY()'),
    (re.compile(RECEIVER + r'\.getZ\(\)\.intValue\(\)'), r'(int) \1.getZ()'),

    # floatValue -> remove
    (re.compile(RECEIVER + r'\.getX\(\)\.floatValue\(\)'), r'\1.getX()'),
    (re.compile(RECEIVER + r'\.getY\(\)\.floatValue\(\)'), r'\1.getY()'),
    (re.compile(RECEIVER + r'\.getZ\(\)\.floatValue\(\)'), r'\1.getZ()'),
    (re.compile(RECEIVER + r'\.getYaw\(\)\.floatValue\(\)'), r'\1.getYaw()'),
    (re.compile(RECEIVER + r'\.getPitch\(\)\.floatValue\(\)'), r'\1.getPitch()'),

    # new BigDecimal(...) -> (float)(...)
    (re.compile(r'\.setX\(new BigDecimal\((.*?)\)\)'), r'.setX((float) (\1))'),
    (re.compile(r'\.setY\(new BigDecimal\((.*?)\)\)'), r'.setY((float) (\1))'),
    (re.compile(r'\.setZ\(new BigDecimal\((.*?)\)\)'), r'.setZ((float) (\1))'),

    # 修正已经错误替换的：setX((float) a + b) -> setX((float)(a + b))
    (re.compile(r'\.setX\(\(float\)\s*([^)]+?)\s*\+\s*([^)]+?)\)'), r'.setX((float) (\1 + \2))'),
    (re.compile(r'\.setY\(\(float\)\s*([^)]+?)\s*\+\s*([^)]+?)\)'), r'.setY((float) (\1 + \2))'),
    (re.compile(r'\.setZ\(\(float\)\s*([^)]+?)\s*\+\s*([^)]+?)\)'), r'.setZ((float) (\1 + \2))'),

    # guiScale
    (
        re.compile(
            r'(guiScale\s*=\s*\(a0start\.getGuiScale\(\)\s*==\s*null\)\s*\?\s*2\s*:\s*)a0start\.getGuiScale\(\)(\s*;)'
        ),
        r'\1(int) a0start.getGuiScale()\2'
    ),
]

def apply_rules(text):
    out = text
    for pat, repl in RULES:
        out = pat.sub(repl, out)
    return out

changed = 0

for rel in FILES:
    p = Path(rel)
    if not p.exists():
        continue

    old = p.read_text(encoding="utf-8")
    new = apply_rules(old)

    if new != old:
        changed += 1
        print(f"\n=== {rel} ===")
        diff = list(difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=rel,
            tofile=rel,
            lineterm=""
        ))
        for line in diff[:120]:
            print(line)
        if len(diff) > 120:
            print("... (diff truncated)")

        if not DRY_RUN:
            backup = p.with_suffix(p.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(p, backup)
            p.write_text(new, encoding="utf-8")

print(f"\nChanged files: {changed}")
print("DRY_RUN =", DRY_RUN)