[CmdletBinding()]
param(
    [string]$SshHost = 'teaching@172.18.40.115',
    [int]$Port = 2692,
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\pcB_ed25519'),
    [switch]$SshVerbose
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $KeyPath -PathType Leaf)) {
    throw "SSH key not found at $KeyPath"
}
$KeyPath = (Resolve-Path -LiteralPath $KeyPath).Path
$sshOptions = @('-i', $KeyPath, '-o', 'IdentitiesOnly=yes')
if ($SshVerbose) { $sshOptions += '-v' }

$repoRoot = Split-Path -Parent $PSScriptRoot
$bundle = Join-Path $env:TEMP ("tcug-gpu-{0}.tar.gz" -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$runs = @(
    'results/checkerboard/iid_20261024_015516_rerun_001',
    'results/checkerboard/noniid_20261024_150813_189'
)
$files = @(
    'scripts',
    'src',
    'data/cifar-10-batches-py',
    'results/rollback/iid/comparison.csv',
    'results/rollback/noniid/comparison.csv'
)

foreach ($run in $runs) {
    $files += @(
        "$run/config.json",
        "$run/metrics_per_round.csv",
        "$run/client_partitions.pt",
        "$run/final_model.pt",
        "$run/trajectory/round_029/global_model.pt",
        "$run/unlearning/20261024_205805_550/federaser_Ecal2/run_config.json",
        "$run/unlearning/20261024_205805_550/federaser_Ecal2/final_model.pt"
    )
    foreach ($round in 25..29) {
        $files += ('{0}/trajectory/round_{1:D3}/delta_client_00.pt' -f $run, $round)
    }
}

foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $file))) {
        throw "Required experiment input is missing: $file"
    }
}

ssh @sshOptions -p $Port $SshHost 'mkdir -p "$HOME/tcug"'
if ($LASTEXITCODE -ne 0) {
    throw "SSH setup failed (exit $LASTEXITCODE). Check the SSH message above or rerun with -SshVerbose."
}

tar.exe -czf $bundle --exclude=__pycache__ -C $repoRoot @files
if ($LASTEXITCODE -ne 0) { throw 'Archive creation failed.' }

scp @sshOptions -P $Port $bundle "${SshHost}:tcug/tcug-gpu.tar.gz"
if ($LASTEXITCODE -ne 0) { throw "Upload failed. The local archive is at $bundle" }

Write-Host "Uploaded to ${SshHost}:~/tcug/tcug-gpu.tar.gz"
Write-Host "Local archive: $bundle"
