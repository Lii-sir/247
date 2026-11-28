$ErrorActionPreference = 'Stop'
$target = 'x86_64-pc-windows-msvc'
$sysroot = & rustc --print sysroot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$linker = Join-Path $sysroot "lib/rustlib/$target/bin/rust-lld.exe"
$build = New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot ('.build-' + [Guid]::NewGuid().ToString('N')))
$object = Join-Path $build.FullName 'silver_scan.obj'
$library = Join-Path $build.FullName 'silver_scan.dll'
try {
    & rustc --target $target --crate-type=lib --edition=2021 --emit=obj -C opt-level=3 -C panic=abort -o $object (Join-Path $PSScriptRoot 'silver_scan.rs')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $linker -flavor link /dll /noentry /nodefaultlib /export:scan_silver_rectangle "/out:$library" $object
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $destination = Join-Path $PSScriptRoot 'silver_scan.dll'
    if (Test-Path -LiteralPath $destination) {
        [IO.File]::Replace($library, $destination, (Join-Path $build.FullName 'previous.dll'))
    } else {
        [IO.File]::Move($library, $destination)
    }
    Write-Output $destination
} finally {
    Get-ChildItem -LiteralPath $build.FullName -File | Remove-Item
    Remove-Item -LiteralPath $build.FullName
}
