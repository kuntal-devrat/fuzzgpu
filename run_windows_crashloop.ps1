# Windows concurrent crash loop — the fuzzgpu suite with fully concurrent GPU
# dispatch (the serialization workaround was removed; FUZZGPU_SKIP_DISPATCH_LOCK
# is now an opt-in safety valve that *enables* serialization, so it is NOT set
# here).
# Distinguishes wgpu hard crashes from Rust test-isolation panics.
param(
    [int]$Runs    = 150,
    [int]$Threads = 8
)

$BIN = Get-ChildItem "target\debug\deps\fuzzgpu_core-*.exe" |
       Sort-Object LastWriteTime -Descending |
       Select-Object -First 1 -ExpandProperty FullName

if (-not $BIN) {
    Write-Error "No test binary found. Run: cargo test -p fuzzgpu-core --lib --no-run"
    exit 1
}

Write-Host "Binary : $BIN"
Write-Host "Runs   : $Runs   Threads: $Threads"
Write-Host "wgpu   : stock (concurrent dispatch)"
Write-Host ""

$ok = 0; $rustPanic = 0; $heapCorrupt = 0; $accessViol = 0; $otherCrash = 0

for ($i = 1; $i -le $Runs; $i++) {
    & $BIN gpu --test-threads=$Threads 2>$null | Out-Null
    $uint = [uint32]([int32]$LASTEXITCODE)

    if ($uint -eq 0) {
        $ok++
        Write-Host ("  run {0,3}  ok" -f $i)
    } elseif ($uint -eq 0x65) {
        $rustPanic++
        Write-Host ("  run {0,3}  Rust panic / test isolation (exit 101)" -f $i)
    } elseif ($uint -eq 0xC0000374) {
        $heapCorrupt++
        Write-Host ("  run {0,3}  HEAP CORRUPTION 0xC0000374  <<< wgpu bug" -f $i) -ForegroundColor Red
    } elseif ($uint -eq 0xC0000005) {
        $accessViol++
        Write-Host ("  run {0,3}  ACCESS VIOLATION 0xC0000005  <<< wgpu bug" -f $i) -ForegroundColor Red
    } else {
        $otherCrash++
        Write-Host ("  run {0,3}  exit 0x{1:X8}" -f $i, $uint) -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== $Runs runs / $Threads threads / patched wgpu ==="
Write-Host ("  clean                       : {0}" -f $ok)
Write-Host ("  Rust panic (test isolation) : {0}  (fuzzgpu bug, not wgpu)" -f $rustPanic)
Write-Host ("  0xC0000374 heap corruption  : {0}  <<< original wgpu crash" -f $heapCorrupt)
Write-Host ("  0xC0000005 access violation : {0}  <<< original wgpu crash" -f $accessViol)
Write-Host ("  other                       : {0}" -f $otherCrash)
