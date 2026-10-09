[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ObsSourceRoot,
    [ValidateSet('32.2.2', '30.2.3')]
    [string]$ObsTargetVersion = '32.2.2',
    [string]$ObsSourceArchivePath = '',
    [string]$ObsInstallArchivePath = '',
    [string]$ObsInstallRoot = "$env:ProgramFiles\obs-studio",
    [string]$OutputRoot = ''
)

$ErrorActionPreference = "Stop"

function Get-Sha256Hex([string]$Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $hashBytes = $algorithm.ComputeHash($stream)
        return (($hashBytes | ForEach-Object { $_.ToString('X2') }) -join '')
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

function Get-ExportNames([string[]]$DumpLines) {
    $names = [System.Collections.Generic.List[string]]::new()
    foreach ($line in $DumpLines) {
        if ($line -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(.+?)\s*$') {
            $field = $Matches[1].Trim()
            if ($field.Contains(" = ")) {
                $field = $field -replace '\s+=\s+.*$', ''
            }
            if ($field -and -not $names.Contains($field)) {
                $names.Add($field)
            }
        }
    }
    return $names
}

$sourceRootResolved = (Resolve-Path -LiteralPath $ObsSourceRoot).Path
$sourceGitPath = Join-Path $sourceRootResolved '.git'
if (Test-Path -LiteralPath $sourceGitPath) {
    $sourceVersion = (& git -C $sourceRootResolved describe --tags --exact-match 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceVersion -ne $ObsTargetVersion) {
        throw "OBS source must be the exact official $ObsTargetVersion tag (found '$sourceVersion')."
    }
} else {
    if ($ObsTargetVersion -ne '30.2.3') {
        throw "OBS $ObsTargetVersion source must be an exact tagged git checkout."
    }
    if (-not $ObsSourceArchivePath) {
        throw 'Git metadata is absent. Supply the official OBS 30.2.3 Sources.tar.gz with -ObsSourceArchivePath so its identity can be verified.'
    }
    $sourceArchiveResolved = (Resolve-Path -LiteralPath $ObsSourceArchivePath).Path
    $expectedSourceArchiveSha256 = '1DDDC0D042E64329F9DD7B540295D1A03AF9B8FDA5F0A165D81DE8EF51E0373D'
    $sourceArchiveSha256 = Get-Sha256Hex $sourceArchiveResolved
    if ($sourceArchiveSha256 -ne $expectedSourceArchiveSha256) {
        throw "OBS 30.2.3 source archive SHA-256 mismatch: $sourceArchiveSha256"
    }
    if ((Split-Path -Leaf $sourceRootResolved) -ne 'obs-studio-30.2.3-sources') {
        throw 'OBS 30.2.3 source root must be the official archive extraction directory named obs-studio-30.2.3-sources.'
    }
}

if ($ObsTargetVersion -eq '30.2.3') {
    if (-not $ObsInstallArchivePath) {
        throw 'Supply the official OBS 30.2.3 Windows.zip with -ObsInstallArchivePath so the runtime DLL identity can be verified.'
    }
    $installArchiveResolved = (Resolve-Path -LiteralPath $ObsInstallArchivePath).Path
    $expectedInstallArchiveSha256 = 'C5BF721258F6D8B2342ED39F04E65CDD3A07A9D1300654471D4BE91D87FEE65B'
    $installArchiveSha256 = Get-Sha256Hex $installArchiveResolved
    if ($installArchiveSha256 -ne $expectedInstallArchiveSha256) {
        throw "OBS 30.2.3 Windows archive SHA-256 mismatch: $installArchiveSha256"
    }
}

$headerPath = Join-Path $sourceRootResolved "libobs\obs-config.h"
if (-not (Test-Path -LiteralPath $headerPath)) {
    throw "Missing official libobs header: $headerPath"
}
$headerText = Get-Content -LiteralPath $headerPath -Raw
$versionParts = $ObsTargetVersion.Split('.')
if ($headerText -notmatch "(?m)^#define LIBOBS_API_MAJOR_VER $($versionParts[0])\s*$" -or
    $headerText -notmatch "(?m)^#define LIBOBS_API_MINOR_VER $($versionParts[1])\s*$" -or
    $headerText -notmatch "(?m)^#define LIBOBS_API_PATCH_VER $($versionParts[2])\s*$") {
    throw "OBS source headers do not declare libobs API $ObsTargetVersion."
}

$obsDll = Join-Path $ObsInstallRoot "bin\64bit\obs.dll"
if (-not (Test-Path -LiteralPath $obsDll)) {
    throw "Missing installed OBS runtime DLL: $obsDll"
}
$runtimeVersion = (Get-Item -LiteralPath $obsDll).VersionInfo.ProductVersion
if ($runtimeVersion -ne $ObsTargetVersion) {
    throw "Installed obs.dll must be version $ObsTargetVersion (found '$runtimeVersion')."
}

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $PSScriptRoot "..\.deps\obs-sdk-$ObsTargetVersion"
}

$dumpbinCommand = Get-Command dumpbin.exe -ErrorAction Stop
$libCommand = Get-Command lib.exe -ErrorAction Stop
$outputResolved = [System.IO.Path]::GetFullPath($OutputRoot)
$libraryDirectory = Join-Path $outputResolved "lib"
$configDirectory = Join-Path $outputResolved "lib\cmake\libobs"
$generatedIncludeDirectory = Join-Path $outputResolved "include"
New-Item -ItemType Directory -Force -Path $libraryDirectory, $configDirectory, $generatedIncludeDirectory | Out-Null

$definitionFile = Join-Path $libraryDirectory "obs.dll.def"
$importLibrary = Join-Path $libraryDirectory "obs.lib"
$exportLines = & $dumpbinCommand.Source /nologo /exports $obsDll
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin could not read exports from $obsDll"
}
$exportNames = @(Get-ExportNames $exportLines)
if ($exportNames.Count -lt 100) {
    throw "Unexpectedly few exports found in obs.dll ($($exportNames.Count))."
}
@("LIBRARY obs.dll", "EXPORTS") + $exportNames | Set-Content -LiteralPath $definitionFile -Encoding ascii
& $libCommand.Source /nologo "/def:$definitionFile" /machine:x64 "/out:$importLibrary"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $importLibrary)) {
    throw "lib.exe failed to create the x64 obs.dll import library."
}

$generatedConfigHeader = @'
#pragma once
#define OBS_DATA_PATH "../../data/obs-studio"
#define OBS_PLUGIN_PATH "../../obs-plugins/64bit"
#define OBS_PLUGIN_DESTINATION "obs-plugins/64bit"
#define OBS_RELEASE_CANDIDATE 0
#define OBS_BETA 0
'@
Set-Content -LiteralPath (Join-Path $generatedIncludeDirectory "obsconfig.h") -Value $generatedConfigHeader -Encoding ascii

$escapedSourceInclude = $sourceRootResolved.Replace('\', '/') + "/libobs"
$escapedGeneratedInclude = $generatedIncludeDirectory.Replace('\', '/')
$escapedImportLibrary = $importLibrary.Replace('\', '/')
$escapedRuntimeDll = $obsDll.Replace('\', '/')
$packageConfig = @"
if(NOT TARGET OBS::libobs)
  add_library(OBS::libobs SHARED IMPORTED)
  set_target_properties(OBS::libobs PROPERTIES
    IMPORTED_CONFIGURATIONS RELEASE
    IMPORTED_IMPLIB_RELEASE "$escapedImportLibrary"
    IMPORTED_LOCATION_RELEASE "$escapedRuntimeDll"
    MAP_IMPORTED_CONFIG_DEBUG Release
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
    MAP_IMPORTED_CONFIG_MINSIZEREL Release
    INTERFACE_INCLUDE_DIRECTORIES "$escapedSourceInclude;$escapedGeneratedInclude"
    INTERFACE_COMPILE_DEFINITIONS HAVE_OBSCONFIG_H
  )
endif()
"@
Set-Content -LiteralPath (Join-Path $configDirectory "libobsConfig.cmake") -Value $packageConfig -Encoding utf8

$versionConfig = @"
set(PACKAGE_VERSION "$ObsTargetVersion")
if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
elseif(PACKAGE_FIND_VERSION VERSION_LESS PACKAGE_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
else()
  set(PACKAGE_VERSION_UNSUITABLE TRUE)
endif()
"@
Set-Content -LiteralPath (Join-Path $configDirectory "libobsConfigVersion.cmake") -Value $versionConfig -Encoding utf8

Write-Output "Prepared OBS $ObsTargetVersion source headers and runtime import library at: $outputResolved"
if ($ObsTargetVersion -eq '30.2.3') {
    Write-Output "Configure with: cmake -S . -B build-obs30 -G Ninja -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_BUILD_TYPE=Release -DOBS_TARGET_VERSION=30.2.3 -DOBS_LIBOBS_DIR=`"$configDirectory`""
} else {
    Write-Output "Configure with: cmake -S . -B build -A x64 -DOBS_TARGET_VERSION=32.2.2 -DOBS_LIBOBS_DIR=`"$configDirectory`""
}
