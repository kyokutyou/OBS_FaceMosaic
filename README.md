# OBS_FaceMosaic

Windows x64用のOBS Studio用の顔モザイクフィルターです。標準メディアソース（`ffmpeg_source`）の映像を顔検出し、検出したフレーム自身へモザイクを適用します。

対象は **OBS 32.2.2／30.2.3**（版ごとに別ビルド）。[プラグイン試作版0.1.1](https://github.com/kyokutyou/OBS_FaceMosaic/releases/tag/v0.1.1)の使用するOBS版に合うZIPと、[共通モデルパック0.1.1](https://github.com/kyokutyou/OBS_FaceMosaic/releases/tag/models-v0.1.1)を取得してください。ソース内にモデル・DLLは含みません。

## 制限

- 検出漏れや遮蔽不足があり、すべての顔の秘匿を保証しません。
- モデル未準備・処理障害時は黒画面またはフレーム破棄へ移ります。
- 音声は加工しません。録画でずれを測定し、OBSの同期オフセットを調整してください。

## ビルド

Visual StudioのC++ x64ツールチェーン、CMake 3.28以降、対象OBSと同版のlibobs SDKが必要です。CMakeがONNX Runtime DirectML **1.24.4**とDirectML **1.15.4**を取得し、SHA-256を確認します。モデルはビルドに不要、`OBSFM_ENABLE_TEST_HOOKS`は既定の`OFF`で使用します。

VS x64 Native Tools環境のPowerShellで、リポジトリルートから実行します。

<details>
<summary>libobs SDKがない場合の準備（32.2.2／30.2.3）</summary>

[準備スクリプト](cmake/PrepareObsSdk.ps1)は公式ソースと同版の`obs.dll`からSDKを作ります。32.2.2では同版OBSを標準のProgram Files配下へインストールしてから実行します。別の場所の場合は`-ObsInstallRoot`を指定します。

```powershell
git clone --branch 32.2.2 --depth 1 https://github.com/obsproject/obs-studio.git .deps/obs-studio-32.2.2
./cmake/PrepareObsSdk.ps1 -ObsSourceRoot .deps/obs-studio-32.2.2
```

30.2.3では[公式Sources.tar.gzとWindows.zip](https://github.com/obsproject/obs-studio/releases/tag/30.2.3)を取得・展開し、配置に合わせて以下のパスを変更します。

```powershell
./cmake/PrepareObsSdk.ps1 `
  -ObsTargetVersion 30.2.3 `
  -ObsSourceRoot '.deps/obs30/source/obs-studio-30.2.3-sources' `
  -ObsSourceArchivePath '.deps/obs30/OBS-Studio-30.2.3-Sources.tar.gz' `
  -ObsInstallRoot '.deps/obs30/windows' `
  -ObsInstallArchivePath '.deps/obs30/OBS-Studio-30.2.3-Windows.zip'
```

</details>

以下は32.2.2用です。既存SDKを使う場合は`$obsSdk`を`libobsConfig.cmake`のあるディレクトリーへ変更します。30.2.3用では、版・SDKパスを30.2.3、ビルド先を`build-obs30`へ変更して別ビルドにします。

```powershell
$obsSdk = (Resolve-Path '.deps/obs-sdk-32.2.2/lib/cmake/libobs').Path
cmake -S . -B build -A x64 -DOBS_TARGET_VERSION=32.2.2 -DOBS_LIBOBS_DIR="$obsSdk"
cmake --build build --config Release
cmake --install build --config Release --prefix build/package
```

## モデル

通常は[共通モデルパック0.1.1](https://github.com/kyokutyou/OBS_FaceMosaic/releases/tag/models-v0.1.1)を展開し、その`models`フォルダーを指定します。以下は変換を再現する場合の手順です。原本は[FaceMosaic v1.1.1](https://github.com/Liala1/FaceMosaic/releases/tag/v1.1.1)の`yolov11n-face.onnx`／`yolov11m-face.onnx`です。

Python 3.12で[変換スクリプト](tools/convert-models.py)と[固定依存一覧](tools/requirements.txt)を使います。原本を上書きせず、入力パスを変更し、出力先には未作成のフォルダーを指定してください。

```powershell
python -m venv .deps/model-conversion
& ./.deps/model-conversion/Scripts/python.exe -m pip install -r tools/requirements.txt
& ./.deps/model-conversion/Scripts/python.exe tools/convert-models.py 'C:/path/to/yolov11n-face.onnx' models-lite --model n
& ./.deps/model-conversion/Scripts/python.exe tools/convert-models.py 'C:/path/to/yolov11m-face.onnx' models-detail --model m
```

生成された次の4ファイルをモデルフォルダーへ集めます。FP32は原本と同一のコピーです。比較用の`*-conv-concat-mixed.onnx`は非対応です。プラグインはSHA-256が一致しないモデルを拒否します。

| ファイル名 | SHA-256 |
| --- | --- |
| `face-mosaic-lite-fp16.onnx`（初期選択） | `96BD42768D704814E7386F4C38E056F05BEF2B2C0224F5E02AB8B4A3B6125F76` |
| `face-mosaic-detail-fp16.onnx` | `8FCE465D4AD0F584B1E7E739EFE5E76AC417F5FA5AE7F9240A8A2A9F9E448D53` |
| `face-mosaic-lite-fp32.onnx` | `A4DFFE5F031E476186E3EAB59BB0BC1201364DA02FD505A2949C79FD278E30E2` |
| `face-mosaic-detail-fp32.onnx` | `325F267E8F24C22D67179A77AFE8B93D179098E787112F64E4784A4A0AD5D7EB` |

## OBSへの導入

1. OBSを終了し、対象版の配布ZIP内`obs-plugins/64bit/`の`obs-face-mosaic.dll`、`onnxruntime.dll`、`DirectML.dll`をOBSの`obs-plugins/64bit/`へ配置します。自分でビルドした場合は`build/package/`（30.2.3用は`build-obs30/package/`）を使います。更新前は同名ファイルをバックアップしてください。
2. モデルZIPを展開し、内側の`data`をOBSのインストール先（通常は`C:\Program Files\obs-studio`）へコピーします。別ドライブではそのOBSフォルダーを使います。
3. 専用テストシーンのメディアソースへ「自動顔モザイク (OBS_FaceMosaic)」を追加し、軽量モデルとDirectML GPUを選びます。モデルフォルダーは`data/obs-plugins/obs-face-mosaic/models/`を自動で使います。任意の指定も可能です。
4. テスト映像を録画し、モザイク、停止・再開、黒画面への移行、音声同期を確認します。

アンインストールはOBS終了後にプラグインDLLを削除します。共有される可能性のあるORT・DirectML DLLは一律に削除しないでください。[OBS公式の導入説明](https://obsproject.com/kb/plugins-guide)も参照してください。

## ライセンス

Copyright (C) 2026 kyokutyou

自作ソース・ビルド設定・補助スクリプト・本READMEは **AGPL-3.0-or-later** です。[LICENSE](LICENSE)に従って再配布・改変でき、法律で認められる範囲で無保証です。外部ファイルには各権利者の条件が適用されます。

- OBS libobs：[GPL-2.0-or-later](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs.h)
- ONNX Runtime：[MITと第三者構成要素の条件](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4)
- DirectMLバイナリ：[Microsoft Software License Termsと第三者構成要素の条件](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4)
- モデル：配布元の条件。AGPL表示から`or-later`の許諾まで確認したとは扱いません。

バイナリ・モデルの配布には、対応ソース、ライセンス本文・通知、各配布条件の整備が必要です。
