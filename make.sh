#!/bin/bash

TARGET_PLATFORM=""
if [ -n "${1:-}" ]; then
  if [ "${1}" = "apple2" ] || [ "${1}" = "pet" ]; then
    TARGET_PLATFORM="${1}"
  else
    echo "Unknown platform: ${1}"
    echo "Supported platforms: apple2, pet"
    exit 1
  fi
fi

if [ "${TARGET_PLATFORM}" = "apple2" ] || [ "${TARGET_PLATFORM}" = "pet" ]; then
  REALIO_VAL="4"
  if [ "${TARGET_PLATFORM}" = "pet" ]; then
    REALIO_VAL="3"
  fi
  REALIO_VAL="${REALIO_VAL}" python3 - <<'PY'
from pathlib import Path
import os
import re

path = Path("BASIC-M6502/m6502.asm")
lines = path.read_text(encoding="utf-8").splitlines(True)
pattern = re.compile(r'^(?P<prefix>\s*REALIO\s*=\s*)\d(?P<suffix>.*)$')
value = os.environ.get("REALIO_VAL", "")
if not value:
    raise SystemExit("REALIO_VAL not set")
out = []
for line in lines:
    m = pattern.match(line)
    if m:
        line_end = "\n" if line.endswith("\n") else ""
        suffix = m.group('suffix').rstrip("\r\n")
        out.append(f"{m.group('prefix')}{value}{line_end}")
        if suffix.strip():
            out.append(f"\t\t\t\t{suffix.lstrip()}{line_end}")
    else:
        out.append(line)
path.write_text("".join(out), encoding="utf-8")
PY
fi

PRINTX_LOG="$(mktemp)"
PRINTX_OUT="$(python macro10_to_ca65.py -o ./m6502.s BASIC-M6502/m6502.asm | tee "${PRINTX_LOG}")"

# ca65 m6502.s -o m6502.o
ca65 m6502.s --cpu 6502 -l m6502.lst -o m6502.o

INIT_ADDR="$(awk '/[[:space:]]INIT:/{print $1; exit}' m6502.lst | sed 's/^.*://')"
PTR_BASE_ADDR="$(awk '/[[:space:]]TXTTAB:/{print $1; exit}' m6502.lst)"
IO_PROFILE=""
if [ -n "${TARGET_PLATFORM}" ]; then
  IO_PROFILE="${TARGET_PLATFORM}"
fi
if [ -z "${IO_PROFILE}" ]; then
  IO_PROFILE="pet"
  if command -v rg >/dev/null 2>&1; then
    if rg -q "^PRINTX APPLE$" "${PRINTX_LOG}"; then
      IO_PROFILE="apple2"
    elif rg -q "^PRINTX COMMODORE$" "${PRINTX_LOG}"; then
      IO_PROFILE="pet"
    fi
  else
    if grep -q "^PRINTX APPLE$" "${PRINTX_LOG}"; then
      IO_PROFILE="apple2"
    elif grep -q "^PRINTX COMMODORE$" "${PRINTX_LOG}"; then
      IO_PROFILE="pet"
    fi
  fi
fi
ROMLOC_DEC="$(python3 - <<'PY'
import re
val = ""
with open("m6502.s", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.match(r'^\s*\.define\s+ROMLOC\s+([0-9]+)\b', line)
        if m:
            val = m.group(1)
print(val)
PY
)"
if [ -z "${ROMLOC_DEC}" ]; then
  ROMLOC_DEC="49152"
fi
ROM_START_HEX="$(printf '$%04X' "${ROMLOC_DEC}")"
ROM_START_0X="$(printf '0x%04X' "${ROMLOC_DEC}")"
LINKER_CFG_BASE="linker_pet.cfg"
if [ "${IO_PROFILE}" = "apple2" ]; then
  LINKER_CFG_BASE="linker_apple.cfg"
fi
LINKER_CFG="${LINKER_CFG_BASE}"
if [ "${IO_PROFILE}" = "apple2" ]; then
  LINKER_CFG="$(mktemp)"
  awk -v repl="${ROM_START_HEX}" '{sub(/start = \\$[0-9A-Fa-f]+/,"start = "repl); print}' "${LINKER_CFG_BASE}" > "${LINKER_CFG}"
fi
ld65 m6502.o -o m6502.bin -C "${LINKER_CFG}"
if [ -n "$INIT_ADDR" ]; then
  INIT_ADDR_HEX="0x${INIT_ADDR#00}"
  echo "INIT address: ${INIT_ADDR_HEX}"
  if [ -n "$PTR_BASE_ADDR" ]; then
    PTR_BASE_HEX="0x${PTR_BASE_ADDR#00}"
  else
    PTR_BASE_HEX="0x26"
  fi
  echo "IO profile: ${IO_PROFILE}"
  echo "PTR_BASE (TXTTAB): ${PTR_BASE_HEX}"
  if [ "${IO_PROFILE}" = "apple2" ]; then
    echo "ROM start: ${ROM_START_0X}"
    echo "Run: ./m6502emu/run_m6502emu.sh --io ${IO_PROFILE} --rom /workspaces/m6502.bin:${ROM_START_0X} --start ${INIT_ADDR_HEX}"
  else
    echo "Run: ./m6502emu/run_m6502emu.sh --io ${IO_PROFILE} --ptr-base ${PTR_BASE_HEX} --rom /workspaces/m6502.bin:0xC000 --start ${INIT_ADDR_HEX}"
  fi
else
  echo "INIT address not found in m6502.lst"
fi

rm -f "${PRINTX_LOG}"
if [ "${IO_PROFILE}" = "apple2" ]; then
  rm -f "${LINKER_CFG}"
fi
