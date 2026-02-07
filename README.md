## Real BASIC-M6502 on ca65 (Happy Retro BASIC) 😊

日本語の`README`は、[README_JA](./README_JA.md)を参照してください。

The Microsoft-published MOS 6502 BASIC source in
[BASIC-M6502](https://github.com/microsoft/BASIC-M6502) is written in MACRO-10,
so it cannot be assembled directly with the `ca65` assembler from the standard
6502 toolchain `cc65`.

This project builds a converter that transforms the MACRO-10 source into
ca65-compatible assembly. You can then assemble the converted source into a
binary and run it on the bundled emulator. 🎉

### Goals
- Convert `BASIC-M6502/m6502.asm` into ca65-ready `m6502.s`
- Build a binary with `ca65` / `ld65`
- Run it with `m6502emu`

### Required Tools
- `python3`
- `ca65` and `ld65` (cc65 toolchain)
- `bash`

This project includes dev container configuration, and those tools are already
installed in the container. ✅

### Submodules
This repository uses git submodules (for example, `m6502emu`). Since
`BASIC-M6502` is also a submodule, you need to clone with submodules enabled.

#### Initialize submodules

```sh
git submodule update --init --recursive
```

Clone with submodules from the start:

```sh
git clone --recurse-submodules https://github.com/nosuz/conv_BASIC-M6502.git
```

### Platform Switch (REALIO)

The target platform is controlled by `REALIO` in `BASIC-M6502/m6502.asm`. Change
that value to switch the platform.

### Build (Create Binary)

Use `make.sh` to rewrite `REALIO` for the chosen platform, generate ca65
assembly, and build the binary.

```sh
./make.sh apple2
```

Or:

```sh
./make.sh pet
```

If you pass an unsupported value, the script prints the allowed options and
exits.

### Run
The command to run the binary on the MOS 6502 emulator is printed by `make.sh`.
Typical examples:

Apple II:
```sh
./m6502emu/run_m6502emu.sh --io apple2 --rom /workspaces/m6502.bin:0x0800 --start 0x26D4 --break-target 0x0E1F --iscntc 0x0E12
```

PET:
```sh
./m6502emu/run_m6502emu.sh --io pet --ptr-base 0x0026 --rom /workspaces/m6502.bin:0xC000 --start 0xE03F --break-target 0xC66
```

#### Stop the Emulator
`m6502emu` must be stopped by an external signal. Open another terminal and run:

```sh
./m6502emu/run_m6502emu.sh --kill
```

### Apple SAVE/LOAD Note
In `BASIC-M6502/m6502.asm`, the SAVE/LOAD commands are disabled by default.
Enable `DISKO==1` inside the `REALIO=4` block to build with SAVE/LOAD.

### Easter Eggs 🥚

`BASIC-M6502` is known to contain several easter eggs. We confirmed the Apple
and Commodore PET ones, and it appears KIM has the same one as Apple.

#### Apple Easter Egg

At the memory-size prompt (`MEMORY SIZE ?`), enter `A` and then `Enter` to see
the author message:
```
MEMORY SIZE ? A

WRITTEN BY WEILAND & GATES
MEMORY SIZE ?
```

#### PET Easter Egg (Screen Memory)
Running `WAIT6502,1` (or `WAIT6502,2`) writes `MICROSOFT!` to PET video RAM
(`$8000`) the number of times given by the last digit.

Because it writes to VRAM, enable the capture option `--pet-screen-capture` to
see it in the console:
```sh
./m6502emu/run_m6502emu.sh --io pet --ptr-base 0x0026 --rom /workspaces/m6502.bin:0xC000 --start 0xE03F --pet-screen-capture
```
Then enter `WAIT6502,1` (the last digit can be any number):
```
READY.
WAIT6502,2
[PET SCREEN $8000] $0D 'M'
[PET SCREEN $8001] $09 'I'
[PET SCREEN $8002] $03 'C'
[PET SCREEN $8003] $12 'R'
[PET SCREEN $8004] $0F 'O'
[PET SCREEN $8005] $13 'S'
[PET SCREEN $8006] $0F 'O'
[PET SCREEN $8007] $06 'F'
[PET SCREEN $8008] $14 'T'
[PET SCREEN $8009] $21 '!'
[PET SCREEN $800A] $0D 'M'
[PET SCREEN $800B] $09 'I'
[PET SCREEN $800C] $03 'C'
[PET SCREEN $800D] $12 'R'
[PET SCREEN $800E] $0F 'O'
[PET SCREEN $800F] $13 'S'
[PET SCREEN $8010] $0F 'O'
[PET SCREEN $8011] $06 'F'
[PET SCREEN $8012] $14 'T'
[PET SCREEN $8013] $21 '!'
READY.
```

### Quick Test
```sh
printf '10 A=1\r20 PRINT A\rRUN\r' | ./m6502emu/run_m6502emu.sh --io pet --ptr-base 0x0026 --rom /workspaces/m6502.bin:0xC000 --start 0xE03F
```

### Dev Container 🐳
This project is designed to run inside a dev container.

Open the folder in VS Code, then use the Command Palette (Ctrl+Shift+P) and run
`Dev Containers: Rebuild and Reopen in Container` or
`Dev Containers: Reopen in Container`. The first run takes some time while the
container builds.

### Apple Ctrl-C Note
In the Apple build, `ISCNTC` lives in `BASIC-M6502/m6502.asm` and reads the
keyboard strobe at `$C000`. The relevant code is:
```asm
ISCNTC: LDA $C000        ; ^O140000
        CMP #$83         ; ^O203 (Ctrl-C)
        BEQ ISCCAP
        RTS
ISCCAP: JSR INCHR
        CMP #$83
        ; next is STOP
```
However, `INCHR` masks the input to 7-bit ASCII:
```asm
INCHR:  JSR CQINCH       ; FD0C
        AND #$7F
        RTS
```
So the second `CMP #$83` fails (the value becomes `$03`), and `STOP` does not
recognize Ctrl-C. The original code therefore ignores Ctrl-C while a program is
running.

To make BREAK work, the emulator traps `JSR ISCNTC` and jumps to `STOP` with
Carry/Z set for the Ctrl-C case.

### PET Ctrl-C Note
First, note that the BASIC statement loop calls `ISCNTC` but does not branch on
its return value:
```asm
NEWSTT:
        JSR ISCNTC
        ; LISTEN FOR CONTROL-C.
        LDWD TXTPTR
        ...
        JMP NEWSTT
```
There is no conditional branch after `JSR ISCNTC`.

In the source, `ISCNTC` handles Ctrl-C by falling through into `STOP`:
```asm
ISCNTC: ...
        CMP #3
        ; Ctrl-C → fall through
STOP:   BCS STOPC
END:    CLC
STOPC:  BNE CONTRT
        ...
```

On PET, `ISCNTC` is a KERNAL entry (`ISCNTC=^O177741` → `$FFF1`), so BASIC calls
into ROM instead of its local `ISCNTC`. The key point is that the **KERNAL
`ISCNTC` knows the BASIC `STOP` address** on real hardware and can transfer
control there when Ctrl-C is pressed. In the emulator there is no KERNAL ROM, so
the stub must jump to `STOP` explicitly. That is why PET builds require
`--break-target`.
