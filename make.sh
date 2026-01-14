#!/bin/bash

python macro10_to_ca65.py -o ./m6502.s BASIC-M6502/m6502.asm

# ca65 m6502.s -o m6502.o
ca65 m6502.s --cpu 6502 -l m6502.lst -o m6502.o
ld65 m6502.o -o m6502.bin -C linker.cfg
