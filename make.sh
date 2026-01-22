#!/bin/bash

python macro10_to_ca65.py -o ./m6502.s BASIC-M6502/m6502.asm

# ca65 m6502.s -o m6502.o
ca65 m6502.s --cpu 6502 -l m6502.lst -o m6502.o
ld65 m6502.o -o m6502.bin -C linker.cfg

INIT_ADDR="$(awk '/[[:space:]]INIT:/{print $1; exit}' m6502.lst | sed 's/^.*://')"
if [ -n "$INIT_ADDR" ]; then
  INIT_ADDR_HEX="0x${INIT_ADDR#00}"
  echo "INIT address: ${INIT_ADDR_HEX}"
  echo "Run: ./m6502emu/run_m6502emu.sh --io pet --rom /workspaces/m6502.bin:0xC000 --start ${INIT_ADDR_HEX}"
else
  echo "INIT address not found in m6502.lst"
fi
