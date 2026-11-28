$ErrorActionPreference = 'Stop'
$compiler = Get-Command rustc.exe -ErrorAction SilentlyContinue
$rustc = if ($compiler) { $compiler.Source } else { Join-Path $env:USERPROFILE '.cargo/bin/rustc.exe' }
if (-not (Test-Path -LiteralPath $rustc)) { throw 'Install the Windows x64 MSVC Rust toolchain first.' }
$sysroot = & $rustc --print sysroot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$target = 'x86_64-pc-windows-msvc'
$linker = Join-Path $sysroot "lib/rustlib/$target/bin/rust-lld.exe"
$buildDir = Join-Path $PSScriptRoot ('.build/' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
$object = Join-Path $buildDir 'affine_score.obj'
$library = Join-Path $buildDir 'affine_score.dll'
try {
    & $rustc --target $target --crate-type=lib --edition=2021 --emit=obj -C opt-level=3 -C panic=abort -o $object (Join-Path $PSScriptRoot 'affine_score.rs')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $linker -flavor link /dll /noentry /nodefaultlib /export:affine_has_fma /export:affine_roi_mismatches /export:affine_rois_mismatches /export:build_affine_blocks /export:score_affine_blocks "/out:$library" $object
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $destination = Join-Path $PSScriptRoot 'affine_score.dll'
    $previous = Join-Path $buildDir 'previous.dll'
    if (Test-Path -LiteralPath $destination) {
        [IO.File]::Replace($library, $destination, $previous)
    } else {
        [IO.File]::Move($library, $destination)
    }
    Write-Output $destination
} finally {
    foreach ($file in @($object, $library, (Join-Path $buildDir 'previous.dll'), (Join-Path $buildDir 'affine_score.lib'))) {
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file }
    }
    Remove-Item -LiteralPath $buildDir
}
