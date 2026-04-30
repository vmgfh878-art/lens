param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$FrontendUrl = "http://127.0.0.1:3000",
  [string]$CorsOrigin = "http://127.0.0.1:3000",
  [string]$Ticker = "AAPL",
  [string]$Timeframe = "1D"
)

$ErrorActionPreference = "Stop"

function Join-DemoUrl {
  param([string]$Path)

  if ($Path.StartsWith("http://") -or $Path.StartsWith("https://")) {
    return $Path
  }

  $root = $BaseUrl.TrimEnd("/")
  if ($Path.StartsWith("/")) {
    return "$root$Path"
  }
  return "$root/$Path"
}

function Read-ErrorPayload {
  param($ErrorRecord)

  $raw = $ErrorRecord.ErrorDetails.Message
  if (-not $raw -and $ErrorRecord.Exception.Response) {
    try {
      $stream = $ErrorRecord.Exception.Response.GetResponseStream()
      $reader = [System.IO.StreamReader]::new($stream)
      $raw = $reader.ReadToEnd()
    } catch {
      $raw = $null
    }
  }

  if (-not $raw) {
    return [pscustomobject]@{
      Code = $null
      Message = $ErrorRecord.Exception.Message
    }
  }

  try {
    $json = $raw | ConvertFrom-Json
    $apiError = $json.error
    return [pscustomobject]@{
      Code = $apiError.code
      Message = $apiError.message
    }
  } catch {
    return [pscustomobject]@{
      Code = $null
      Message = $raw
    }
  }
}

function Invoke-DemoGet {
  param(
    [string]$Name,
    [string]$Path,
    [hashtable]$Headers = @{}
  )

  $uri = Join-DemoUrl -Path $Path
  try {
    $response = Invoke-WebRequest -Uri $uri -Method Get -TimeoutSec 10 -UseBasicParsing -Headers $Headers
    $body = $null
    if ($response.Content) {
      try {
        $body = $response.Content | ConvertFrom-Json
      } catch {
        $body = $response.Content
      }
    }
    return [pscustomobject]@{
      Name = $Name
      Ok = $true
      StatusCode = [int]$response.StatusCode
      Body = $body
      Headers = $response.Headers
      ErrorCode = $null
      ErrorMessage = $null
      Uri = $uri
    }
  } catch {
    $statusCode = $null
    if ($_.Exception.Response) {
      try {
        $statusCode = [int]$_.Exception.Response.StatusCode
      } catch {
        $statusCode = $null
      }
    }
    $payload = Read-ErrorPayload -ErrorRecord $_
    return [pscustomobject]@{
      Name = $Name
      Ok = $false
      StatusCode = $statusCode
      Body = $null
      Headers = @{}
      ErrorCode = $payload.Code
      ErrorMessage = $payload.Message
      Uri = $uri
    }
  }
}

function Get-Count {
  param($Value)

  if ($null -eq $Value) {
    return 0
  }
  if ($Value -is [array]) {
    return $Value.Count
  }
  return 1
}

function Write-Check {
  param(
    [string]$Label,
    [string]$State,
    [string]$Detail
  )

  "{0,-12} {1} - {2}" -f $State, $Label, $Detail
}

Write-Host "Lens demo readiness check"
Write-Host "Backend: $BaseUrl"
Write-Host "Frontend: $FrontendUrl"
Write-Host "CORS origin: $CorsOrigin"
Write-Host ""

$live = Invoke-DemoGet -Name "health" -Path "/api/v1/health/live"
if ($live.Ok) {
  Write-Check -State "OK" -Label "health" -Detail "live $($live.StatusCode)"
} else {
  Write-Check -State "FAIL" -Label "health" -Detail "$($live.StatusCode) $($live.ErrorCode) $($live.ErrorMessage)"
}

$cors = Invoke-DemoGet -Name "cors" -Path "/api/v1/health/live" -Headers @{ Origin = $CorsOrigin }
$allowOrigin = $cors.Headers["Access-Control-Allow-Origin"]
if ($cors.Ok -and $allowOrigin -eq $CorsOrigin) {
  Write-Check -State "OK" -Label "cors" -Detail "Access-Control-Allow-Origin=$allowOrigin"
} elseif ($cors.Ok) {
  Write-Check -State "FAIL" -Label "cors" -Detail "Access-Control-Allow-Origin=$allowOrigin"
} else {
  Write-Check -State "FAIL" -Label "cors" -Detail "$($cors.StatusCode) $($cors.ErrorCode) $($cors.ErrorMessage)"
}

$frontend = Invoke-DemoGet -Name "frontend" -Path $FrontendUrl
if ($frontend.Ok) {
  Write-Check -State "OK" -Label "frontend" -Detail "$($frontend.StatusCode)"
} else {
  Write-Check -State "FAIL" -Label "frontend" -Detail "$($frontend.StatusCode) $($frontend.ErrorCode) $($frontend.ErrorMessage)"
}

$pricePath = "/api/v1/stocks/{0}/prices?timeframe={1}&limit=5" -f $Ticker, $Timeframe
$prices = Invoke-DemoGet -Name "AAPL 1D prices" -Path $pricePath
$priceCount = Get-Count -Value $prices.Body.data.data
if ($prices.Ok -and $priceCount -gt 0) {
  Write-Check -State "OK" -Label "1D prices" -Detail "$Ticker $priceCount rows"
} elseif ($prices.Ok) {
  Write-Check -State "WARN" -Label "1D prices" -Detail "$Ticker price rows missing"
} else {
  Write-Check -State "FAIL" -Label "1D prices" -Detail "$($prices.StatusCode) $($prices.ErrorCode) $($prices.ErrorMessage)"
}

$indicatorPath = "/api/v1/stocks/{0}/indicators?timeframe={1}&limit=5" -f $Ticker, $Timeframe
$indicators = Invoke-DemoGet -Name "AAPL 1D indicators" -Path $indicatorPath
$indicatorCount = Get-Count -Value $indicators.Body.data.data
if ($indicators.Ok -and $indicatorCount -gt 0) {
  Write-Check -State "OK" -Label "indicators" -Detail "$Ticker $indicatorCount rows"
} elseif ($indicators.Ok) {
  Write-Check -State "WARN" -Label "indicators" -Detail "$Ticker indicator rows missing"
} else {
  Write-Check -State "FAIL" -Label "indicators" -Detail "$($indicators.StatusCode) $($indicators.ErrorCode) $($indicators.ErrorMessage)"
}

$monthPricePath = "/api/v1/stocks/{0}/prices?timeframe=1M&limit=5" -f $Ticker
$monthPrices = Invoke-DemoGet -Name "AAPL 1M prices" -Path $monthPricePath
$monthPriceCount = Get-Count -Value $monthPrices.Body.data.data
if ($monthPrices.Ok -and $monthPriceCount -gt 0) {
  Write-Check -State "OK" -Label "1M prices" -Detail "price-only view available, $monthPriceCount rows"
} elseif ($monthPrices.Ok) {
  Write-Check -State "WARN" -Label "1M prices" -Detail "$Ticker monthly price rows missing"
} else {
  Write-Check -State "FAIL" -Label "1M prices" -Detail "$($monthPrices.StatusCode) $($monthPrices.ErrorCode) $($monthPrices.ErrorMessage)"
}

$searchPath = "/api/v1/stocks?search={0}&limit=5" -f $Ticker
$search = Invoke-DemoGet -Name "stock search" -Path $searchPath
$searchCount = Get-Count -Value $search.Body.data
if ($search.Ok -and $searchCount -gt 0) {
  Write-Check -State "OK" -Label "stock search" -Detail "$searchCount rows, stock_info or price_data fallback available"
} elseif ($search.Ok) {
  Write-Check -State "WARN" -Label "stock search" -Detail "search rows missing"
} else {
  Write-Check -State "FAIL" -Label "stock search" -Detail "$($search.StatusCode) $($search.ErrorCode) $($search.ErrorMessage)"
}

$runsPath = "/api/v1/ai/runs?model_name=&status=completed&timeframe={0}&limit=50" -f $Timeframe
$runs = Invoke-DemoGet -Name "latest completed run" -Path $runsPath
$candidateModels = @("line_band_composite", "patchtst")
$completedRuns = @($runs.Body.data | Where-Object { $candidateModels -contains $_.model_name })
$latestRun = $completedRuns | Select-Object -First 1
if ($runs.Ok -and $completedRuns.Count -gt 0) {
  Write-Check -State "OK" -Label "completed runs" -Detail "$($completedRuns.Count) demo candidate rows, latest $($latestRun.run_id)"
} elseif ($runs.Ok) {
  Write-Check -State "WARN" -Label "completed runs" -Detail "completed demo candidate run missing"
} else {
  Write-Check -State "FAIL" -Label "completed runs" -Detail "$($runs.StatusCode) $($runs.ErrorCode) $($runs.ErrorMessage)"
}

$demoRun = $null
$demoPrediction = $null

foreach ($run in $completedRuns) {
  $candidateRunId = $run.run_id
  $predictionPath = "/api/v1/stocks/{0}/predictions/latest?run_id={1}" -f $Ticker, $candidateRunId
  $candidatePrediction = Invoke-DemoGet -Name "prediction row" -Path $predictionPath
  if ($candidatePrediction.Ok -and $candidatePrediction.Body.data) {
    $forecastCount = Get-Count -Value $candidatePrediction.Body.data.forecast_dates
    $lineCount = Get-Count -Value $candidatePrediction.Body.data.line_series
    $upperCount = Get-Count -Value $candidatePrediction.Body.data.upper_band_series
    $lowerCount = Get-Count -Value $candidatePrediction.Body.data.lower_band_series
    if ($forecastCount -gt 0 -and $forecastCount -eq $lineCount -and $lineCount -eq $upperCount -and $upperCount -eq $lowerCount) {
      $demoRun = $run
      $demoPrediction = $candidatePrediction
      break
    }
  }
}

if ($demoRun) {
  $runId = $demoRun.run_id
  $forecastCount = Get-Count -Value $demoPrediction.Body.data.forecast_dates
  $lineCount = Get-Count -Value $demoPrediction.Body.data.line_series
  $upperCount = Get-Count -Value $demoPrediction.Body.data.upper_band_series
  $lowerCount = Get-Count -Value $demoPrediction.Body.data.lower_band_series
  Write-Check -State "OK" -Label "demo run" -Detail "$runId"
  Write-Check -State "OK" -Label "prediction" -Detail "forecast=$forecastCount line=$lineCount upper=$upperCount lower=$lowerCount"

  if ($latestRun -and $latestRun.run_id -ne $runId) {
    Write-Check -State "WARN" -Label "latest run" -Detail "latest $($latestRun.run_id) has no usable $Ticker prediction; demo uses $runId"
  } else {
    Write-Check -State "OK" -Label "latest run" -Detail "latest demo candidate has usable $Ticker prediction"
  }

  $evaluationPath = "/api/v1/ai/runs/{0}/evaluations?ticker={1}&timeframe={2}&limit=1" -f $runId, $Ticker, $Timeframe
  $evaluations = Invoke-DemoGet -Name "evaluation rows" -Path $evaluationPath
  $evaluationCount = Get-Count -Value $evaluations.Body.data
  if ($evaluations.Ok -and $evaluationCount -gt 0) {
    Write-Check -State "OK" -Label "evaluation" -Detail "$evaluationCount rows"
  } elseif ($evaluations.Ok) {
    Write-Check -State "WARN" -Label "evaluation" -Detail "$Ticker evaluation row missing"
  } else {
    Write-Check -State "FAIL" -Label "evaluation" -Detail "$($evaluations.StatusCode) $($evaluations.ErrorCode) $($evaluations.ErrorMessage)"
  }

  $backtestPath = "/api/v1/ai/runs/{0}/backtests?timeframe={1}&limit=1" -f $runId, $Timeframe
  $backtests = Invoke-DemoGet -Name "backtest rows" -Path $backtestPath
  $backtestCount = Get-Count -Value $backtests.Body.data
  if ($backtests.Ok -and $backtestCount -gt 0) {
    Write-Check -State "OK" -Label "backtest" -Detail "$backtestCount rows"
  } elseif ($backtests.Ok) {
    Write-Check -State "WARN" -Label "backtest" -Detail "stored backtest rows missing"
  } else {
    Write-Check -State "FAIL" -Label "backtest" -Detail "$($backtests.StatusCode) $($backtests.ErrorCode) $($backtests.ErrorMessage)"
  }
} else {
  Write-Check -State "WARN" -Label "demo run" -Detail "usable prediction row missing"
  Write-Check -State "SKIP" -Label "backtest" -Detail "demo run missing"
  Write-Check -State "SKIP" -Label "evaluation" -Detail "demo run missing"
}

Write-Host ""
Write-Host "WARN or FAIL means demo artifacts are missing or an API is not ready. This script does not create fake data."
