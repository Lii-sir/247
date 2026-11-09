$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) {
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
}
if (-not (Test-Path -LiteralPath $compiler)) { throw '.NET Framework C# compiler not found.' }
$destination = Join-Path $PSScriptRoot 'bin\Release'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$library = Join-Path $destination 'SmartRay.PitchCorrection.dll'
& $compiler /nologo /target:library /platform:anycpu /optimize+ /checked- /warnaserror+ "/out:$library" /reference:System.Core.dll (Join-Path $PSScriptRoot 'PitchCorrector.cs')
if ($LASTEXITCODE -ne 0) { throw "DLL compilation failed: $LASTEXITCODE" }
Write-Output $library
