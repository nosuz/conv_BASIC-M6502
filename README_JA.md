## Real BASIC-M6502 on ca65 (Happy Retro BASIC) 😊

Microsoftが公開したMOS 6502用のBASIC[BASIC-M6502](https://github.com/microsoft/BASIC-M6502)は、MACRO-10で書かれています。そのため6502用の一般的な開発環境`cc65`に含まれるアセンブラ`ca65`で実行ファイルを作成できません。


このプロジェクトは、MACRO10形式で書かれたこの BASIC-M6502 を ca65 でアセンブルできる形に変換するコンバータを作成したプロジェクトです。そして、実際に変換されたアセンブリコードからバイナリを作成し、内臓のエミュレータ上で作成したバイナリを動かして確認できます。 🎉

### 目的
- `BASIC-M6502/m6502.asm` を ca65 用に変換して `m6502.s` を生成
- `ca65` / `ld65` でバイナリを作成
- `m6502emu` で実行

### 必要ツール
- `python3`
- `ca65` と `ld65`（cc65 ツールチェーン）
- `bash`

このプロジェクトは dev container の設定を含んでおり、コンテナにはこれらのツールが最初から揃っています ✅

### サブモジュール
このリポジトリはいくつかの git サブモジュール（例: `m6502emu`）を使用しています。`BASIC-M6502`もサブモジュールとして取り込んでいるため、サブモジュールも含めてレポジトリのクローンが必要です。

#### サブモジュールの取り込み方法

```sh
git submodule update --init --recursive
```

最初からサブモジュールを含めてクローンする場合：

```sh
git clone --recurse-submodules https://github.com/nosuz/conv_BASIC-M6502.git
```

### プラットフォーム切り替え（REALIO）

プラットフォームは、`BASIC-M6502/m6502.asm` の `REALIO`で指定されています。ここを書き換えることで、ターゲットのプラットフォームを書き換えます。

### ビルド（バイナリ作成）

`make.sh`を使用すると、オプションで指定したプラットフォーム用に`REALIO`を書き換えて`ca65`用のアセンブリコードの作成と、実行可能なバイナリを作成できます。

```sh
./make.sh apple2
```

または:

```sh
./make.sh pet
```

サポートされない値を渡すと、利用可能な値を表示して終了します。

### 実行方法
MOS 6502エミュレータで作成したバイナルを実行するコマンドは、`make.sh` の出力含まれます。代表例：

Apple II:
```sh
./m6502emu/run_m6502emu.sh --io apple2 --rom /workspaces/m6502.bin:0x0800 --start 0x26D4 --break-target 0x0E1F --iscntc 0x0E12
```

PET:
```sh
./m6502emu/run_m6502emu.sh --io pet --ptr-base 0x0026 --rom /workspaces/m6502.bin:0xC000 --start 0xE03F --break-target 0xC66
```

#### 停止方法
`m6502emu`は、外部からのシグナルで停止させる必要があります。別のターミナルを開いて、次のコマンドを実行してください。

```sh
./m6502emu/run_m6502emu.sh --kill
```

### Apple の SAVE/LOAD について
`BASIC-M6502/m6502.asm`では、SAVE/LOADコマンドが無効になっています。
`REALIO=4` ブロック内で `DISKO==1` を有効にしてバイナリを作成すると、SAVE/LOADを行えます。


### イースターエッグ 🥚

`BASIC-M6502`には、イースターエッグが仕込まれていることが知られています。この内 AppleとCommodore PETでイースターエッグを表示させることができました。また、KIMにもAppleと同じイースターエッグが仕込まれているようです。

#### Appleのイースターエッグ

起動時のメモリサイズ入力`MEMORY SIZE ?`で、そのまま`Enter`ではなく`A` を入力してから`Enter`を入力すると作者メッセージが表示されます。
```
MEMORY SIZE ? A

WRITTEN BY WEILAND & GATES
MEMORY SIZE ?
```

#### PETのイースターエッグ（画面メモリ）
`WAIT6502,1`（または `WAIT6502,2`）を実行すると、PETのビデオRAM（`$8000`）に最後の数字の回数だけ `MICROSOFT!` が書き込まれます。

コンソールではなくビデオメモリ(VRAM)に書き込まれるので、表示するにはキャプチャオプション`--pet-screen-capture`を有効にしてください。
```sh
./m6502emu/run_m6502emu.sh --io pet --ptr-base 0x0026 --rom /workspaces/m6502.bin:0xC000 --start 0xE03F --pet-screen-capture
```
実行後に`WAIT6502,1`を入力します。最後の`1`は、別の数字でもOKです。
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


### 簡単な動作確認
```sh
printf '10 A=1\r20 PRINT A\rRUN\r' | ./m6502emu/run_m6502emu.sh --io pet --ptr-base 0x0026 --rom /workspaces/m6502.bin:0xC000 --start 0xE03F
```


### Dev Container 🐳
このプロジェクトは dev container 環境での使用を想定しています。

dev containerは、`VSCode`でこのプロジェクトフォルダーを開き、コマンドパレット(Ctrl+Shift+P)から `Dev Containers: Rebuild and Reopen in Container` または `Dev Containers: Reopen in Container` を選択するとコンテナが開きます。初めての場合はコンテナをビルドするため、フォルダが開くまでしばらく時間がかかります。


### Apple の Ctrl-C について
Apple 版では `ISCNTC` が `BASIC-M6502/m6502.asm` にあり、`$C000` のキーボード
ストローブを直接読みます。該当部分は次の通りです:
```asm
ISCNTC: LDA $C000        ; ^O140000
        CMP #$83         ; ^O203 (Ctrl-C)
        BEQ ISCCAP
        RTS
ISCCAP: JSR INCHR
        CMP #$83
        ; 次が STOP
```
しかし `INCHR` が 7ビットにマスクするため:
```asm
INCHR:  JSR CQINCH       ; FD0C
        AND #$7F
        RTS
```
二度目の `CMP #$83` が一致せず、`STOP` が Ctrl-C として扱われません。
その結果、オリジナルのままでは実行中の Ctrl-C は無視されます。

そこでエミュレータ側で `JSR ISCNTC` をトラップし、`STOP` へ遷移する際に
Carry/Z を Ctrl-C ケースに合わせて BREAK を成立させています。

### PET の Ctrl-C について
まず、BASIC のステートメントループは `ISCNTC` を呼ぶだけで、戻り値に
応じた分岐は行いません:
```asm
NEWSTT:
        JSR ISCNTC
        ; LISTEN FOR CONTROL-C.
        LDWD TXTPTR
        ...
        JMP NEWSTT
```
`JSR ISCNTC` の直後にフラグや A を見た分岐が無い点が重要です。

ソース上の `ISCNTC` は、Ctrl-C 判定後に `STOP` へフォールスルーします:
```asm
ISCNTC: ...
        CMP #3
        ; Ctrl-C → fall through
STOP:   BCS STOPC
END:    CLC
STOPC:  BNE CONTRT
        ...
```

しかし PET では `ISCNTC` は KERNAL のエントリ（`ISCNTC=^O177741` → `$FFF1`）で、
BASIC 側の `ISCNTC` 本体は呼ばれません。重要なのは **KERNAL の `ISCNTC` が
BASIC の `STOP` のアドレスを知っている** 点です。実機では KERNAL がそこに
制御を移しますが、エミュレータには KERNAL ROM が無いので、スタブ側で
`STOP` へ飛ばす必要があります。そのため PET ビルドでは `--break-target`
が必要になります。
