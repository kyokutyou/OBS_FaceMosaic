# OBS_FaceMosaic

Windows x64用のOBS Studio顔モザイクフィルター試作です。標準のメディアソース（`ffmpeg_source`）が受信・デコードした映像を顔検出し、検出したフレーム自身へモザイクを適用します。SRTの受信はOBS側が担当します。

現在はソースコードのみを公開する構成です。モデル、プラグインDLL、依存DLLは含みません。

## 対象と制限

- ビルド対象はOBS **32.2.2／30.2.3**、Windows x64、C++20です。OBSの版ごとに別ビルドが必要です。他の版・OS・ARM64は確認対象外です。
- 顔の検出漏れや遮蔽不足があり得ます。すべての顔が隠れることや匿名性を保証しません。
- モデル読み込み中や処理障害時は、動作可能な範囲で黒画面またはフレーム破棄へ移ります。処理の遅延・破棄・加工済み映像の一時的な再表示もあり得ます。
- フィルターの無効化・削除、プラグイン未読み込み、OBS全体の停止、別の未加工ソースへの切替は、このフィルターでは防げません。
- 音声は加工しません。OBSの同期オフセットを使い、録画で音声とのずれを実測してください。
- 隔離したOBSでローカル映像とlocalhost SRTの限定試験を実施しています。実スマートフォン・実ネットワーク、音声同期、2時間連続運転、広い条件での検出品質の評価は未完了です。本番利用の合格判定はしていません。

## ビルド

必要なものはVisual StudioのC++ x64ツールチェーン、CMake 3.28以降、対象OBSと同じ版のlibobs SDKです。依存するNuGetパッケージは以下の固定版をCMakeが取得し、SHA-256を確認します。

- `Microsoft.ML.OnnxRuntime.DirectML` **1.24.4**
- `Microsoft.AI.DirectML` **1.15.4**

以下は32.2.2用の構成例です。VS x64 Native Tools環境のPowerShellで実行します。`$obsSdk`には、対象版の`libobsConfig.cmake`があるディレクトリーを指定してください。

```powershell
$obsSdk = 'C:/path/to/libobs/lib/cmake/libobs'
cmake -S . -B build -A x64 -DOBS_TARGET_VERSION=32.2.2 -DOBS_LIBOBS_DIR="$obsSdk"
cmake --build build --config Release
cmake --install build --config Release --prefix build/package
```

通常のOBS WindowsインストーラーにCMake SDKがない場合は、[cmake/PrepareObsSdk.ps1](cmake/PrepareObsSdk.ps1)で公式ソースのヘッダーと同版の`obs.dll`からSDKを準備できます。32.2.2では公式ソースの正確なタグcheckoutが必要です。

```powershell
git clone --branch 32.2.2 --depth 1 https://github.com/obsproject/obs-studio.git .deps/obs-studio-32.2.2
./cmake/PrepareObsSdk.ps1 -ObsSourceRoot .deps/obs-studio-32.2.2
$obsSdk = (Resolve-Path '.deps/obs-sdk-32.2.2/lib/cmake/libobs').Path
```

30.2.3の場合は同版の[公式Sources.tar.gzとWindows.zip](https://github.com/obsproject/obs-studio/releases/tag/30.2.3)を用意し、展開先と元アーカイブを指定します。以下のパスは配置例です。

```powershell
./cmake/PrepareObsSdk.ps1 `
  -ObsTargetVersion 30.2.3 `
  -ObsSourceRoot '.deps/obs30/source/obs-studio-30.2.3-sources' `
  -ObsSourceArchivePath '.deps/obs30/OBS-Studio-30.2.3-Sources.tar.gz' `
  -ObsInstallRoot '.deps/obs30/windows' `
  -ObsInstallArchivePath '.deps/obs30/OBS-Studio-30.2.3-Windows.zip'
$obsSdk = (Resolve-Path '.deps/obs-sdk-30.2.3/lib/cmake/libobs').Path
cmake -S . -B build-obs30 -A x64 -DOBS_TARGET_VERSION=30.2.3 -DOBS_LIBOBS_DIR="$obsSdk"
cmake --build build-obs30 --config Release
cmake --install build-obs30 --config Release --prefix build-obs30/package
```

`OBSFM_ENABLE_TEST_HOOKS`は既定の`OFF`で使用してください。モデルはビルドには不要です。

## モデルの用意

対象原本は[FaceMosaic v1.1.1](https://github.com/Liala1/FaceMosaic/releases/tag/v1.1.1)に含まれる`yolov11n-face.onnx`／`yolov11m-face.onnx`です。モデルのメタデータと配布元ZIPにはAGPL-3.0の表示があります。モデルの学習由来や配布に必要なソース提供範囲などの条件整理は未完了で、このリポジトリではモデルを再配布しません。

FP32は原本を変更せず、コピー先で次の名前を使います。FP16は[変換スクリプト](tools/convert-models.py)と固定依存一覧を使用します。以下はPython 3.12での変換例です。入力・出力パスは自身の配置に合わせて指定し、出力先には未作成のフォルダーを指定してください。

```powershell
python -m venv .deps/model-conversion
& ./.deps/model-conversion/Scripts/python.exe -m pip install -r tools/requirements.txt
& ./.deps/model-conversion/Scripts/python.exe tools/convert-models.py `
  'C:/path/to/yolov11n-face.onnx' models-lite --model n
& ./.deps/model-conversion/Scripts/python.exe tools/convert-models.py `
  'C:/path/to/yolov11m-face.onnx' models-detail --model m
```

変換結果の`*-fp16.onnx`と`*-fp32.onnx`をモデルフォルダーへ集めます。`*-conv-concat-mixed.onnx`は比較用に生成されるファイルで、プラグインの対応モデルではありません。使用するファイルのSHA-256が次の値と一致することを確認してください。プラグインは一致しないモデルを拒否します。

| 配置ファイル名 | SHA-256 |
| --- | --- |
| `face-mosaic-lite-fp16.onnx`（初期選択） | `96BD42768D704814E7386F4C38E056F05BEF2B2C0224F5E02AB8B4A3B6125F76` |
| `face-mosaic-detail-fp16.onnx` | `8FCE465D4AD0F584B1E7E739EFE5E76AC417F5FA5AE7F9240A8A2A9F9E448D53` |
| `face-mosaic-lite-fp32.onnx`（原本nと同一） | `A4DFFE5F031E476186E3EAB59BB0BC1201364DA02FD505A2949C79FD278E30E2` |
| `face-mosaic-detail-fp32.onnx`（原本mと同一） | `325F267E8F24C22D67179A77AFE8B93D179098E787112F64E4784A4A0AD5D7EB` |

## OBSへの導入

1. OBSを終了し、対象版の`build/package/obs-plugins/64bit/`に生成された`obs-face-mosaic.dll`、`onnxruntime.dll`、`DirectML.dll`を、OBSの`obs-plugins/64bit/`へ配置します。30.2.3用では`build-obs30/package/`を使います。更新前は同名ファイルをバックアップしてください。
2. 専用のテスト用プロファイル・シーンコレクションを用意し、標準メディアソースへ「自動顔モザイク (OBS_FaceMosaic)」を追加します。
3. フィルターの「モデルフォルダー」を指定し、まず軽量モデルと使用するDirectML GPUを選択します。モデル読み込み中は黒画面です。
4. テスト映像の録画で、モザイク、停止・再開、黒画面への移行、音声同期を確認します。

アンインストールはOBS終了後にプラグインDLLを取り外します。ONNX RuntimeとDirectMLのDLLは他プラグインも使っている場合があるため、一律に削除しないでください。配置方式は確認したOBS版に限定します。[OBS公式の導入説明](https://obsproject.com/kb/plugins-guide)も参照してください。

## ライセンス

Copyright (C) 2026 kyokutyou

自作ソース、ビルド設定、補助スクリプト、本READMEは **AGPL-3.0-or-later**（GNU Affero General Public License 第3版またはそれ以降）で提供します。[LICENSE](LICENSE)に従って再配布・改変でき、法律で認められる範囲で無保証です。

外部のOBS、推論ランタイム、モデルには各権利者の条件が適用されます。自作部分のライセンスを外部ファイルへ付与するものではありません。

- OBS libobs：GPL-2.0-or-later。[対象版の表示](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs.h)
- ONNX Runtime：MITと第三者構成要素の条件。[固定版パッケージ](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4)
- DirectMLバイナリ：Microsoft Software License Termsと第三者構成要素の条件。[固定版パッケージ](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4)
- 顔検出モデル：前述の配布元の条件。モデルのAGPL表示から`or-later`の許諾まで確認したとは扱いません。

バイナリやモデルを第三者へ配布する場合は、対応するソース、各ライセンス本文・通知、配布条件を別途整えてください。
