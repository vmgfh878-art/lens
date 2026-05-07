param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$FrontendUrl = "http://127.0.0.1:3000",
  [string]$CorsOrigin = "http://127.0.0.1:3000",
  [string]$Ticker = "AAPL",
  [string]$Timeframe = "1D",
  [string]$LineRunId = "patchtst-1D-efad3c29d803",
  [string]$BandRunId = "cnn_lstm-1D-d0c780dee5e8"
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

function Join-FrontendUrl {
  param([string]$Path)

  if ($Path.StartsWith("http://") -or $Path.StartsWith("https://")) {
    return $Path
  }

  $root = $FrontendUrl.TrimEnd("/")
  if ($Path.StartsWith("/")) {
    return "$root$Path"
  }
  return "$root/$Path"
}

function Test-FrontendStaticAsset {
  param($FrontendResponse)

  if (-not $FrontendResponse.Ok) {
    return [pscustomobject]@{
      Ok = $false
      Detail = "frontend root is not available"
    }
  }

  $html = [string]$FrontendResponse.Body
  $cssMatch = [regex]::Match($html, '<link[^>]+rel=["'']stylesheet["''][^>]+href=["'']([^"'']+\.css[^"'']*)["'']')
  if (-not $cssMatch.Success) {
    return [pscustomobject]@{
      Ok = $false
      Detail = "stylesheet link not found in frontend HTML"
    }
  }

  $cssUrl = Join-FrontendUrl -Path $cssMatch.Groups[1].Value
  try {
    $cssResponse = Invoke-WebRequest -Uri $cssUrl -Method Get -TimeoutSec 10 -UseBasicParsing
    if ([int]$cssResponse.StatusCode -eq 200) {
      return [pscustomobject]@{
        Ok = $true
        Detail = "stylesheet 200 $cssUrl"
      }
    }
    return [pscustomobject]@{
      Ok = $false
      Detail = "stylesheet $($cssResponse.StatusCode) $cssUrl"
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
    return [pscustomobject]@{
      Ok = $false
      Detail = "stylesheet $statusCode $cssUrl $($_.Exception.Message)"
    }
  }
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

$frontendStatic = Test-FrontendStaticAsset -FrontendResponse $frontend
if ($frontendStatic.Ok) {
  Write-Check -State "OK" -Label "frontend-static" -Detail $frontendStatic.Detail
} else {
  Write-Check -State "FAIL" -Label "frontend-static" -Detail $frontendStatic.Detail
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

$legacyRunsPath = "/api/v1/ai/runs?model_name=&status=completed&timeframe={0}&limit=50&include_legacy=true" -f $Timeframe
$legacyRunsResponse = Invoke-DemoGet -Name "legacy composite runs" -Path $legacyRunsPath
$legacyCompositeRuns = @($legacyRunsResponse.Body.data | Where-Object { $_.model_name -eq "line_band_composite" -or $_.is_legacy -eq $true })
if ($legacyRunsResponse.Ok -and $legacyCompositeRuns.Count -gt 0) {
  $legacyRun = $legacyCompositeRuns | Select-Object -First 1
  Write-Check -State "LEGACY_OK" -Label "legacy composite" -Detail "$($legacyRun.run_id) preserved, not product default"
} elseif ($legacyRunsResponse.Ok) {
  Write-Check -State "OK" -Label "legacy composite" -Detail "legacy composite run not required"
} else {
  Write-Check -State "WARN" -Label "legacy composite" -Detail "$($legacyRunsResponse.StatusCode) $($legacyRunsResponse.ErrorCode) $($legacyRunsResponse.ErrorMessage)"
}

Write-Host ""
Write-Host "Product line/band layer candidates"

$lineRun = Invoke-DemoGet -Name "LM line run" -Path ("/api/v1/ai/runs/{0}" -f $LineRunId)
if ($lineRun.Ok -and $lineRun.Body.data.status -eq "completed") {
  $lineRole = $lineRun.Body.data.role
  if (-not $lineRole) {
    $lineRole = $lineRun.Body.data.config_summary.role
  }
  Write-Check -State "OK" -Label "LM run" -Detail "$LineRunId status=completed role=$lineRole"
} elseif ($lineRun.Ok) {
  Write-Check -State "WARN" -Label "LM run" -Detail "$LineRunId status=$($lineRun.Body.data.status)"
} else {
  Write-Check -State "FAIL" -Label "LM run" -Detail "$($lineRun.StatusCode) $($lineRun.ErrorCode) $($lineRun.ErrorMessage)"
}

$linePredictionPath = "/api/v1/stocks/{0}/predictions/latest?run_id={1}" -f $Ticker, $LineRunId
$linePrediction = Invoke-DemoGet -Name "LM prediction" -Path $linePredictionPath
if ($linePrediction.Ok -and $linePrediction.Body.data) {
  $forecastCount = Get-Count -Value $linePrediction.Body.data.forecast_dates
  $lineCount = Get-Count -Value $linePrediction.Body.data.line_series
  if ($lineCount -eq 0) {
    $lineCount = Get-Count -Value $linePrediction.Body.data.conservative_series
  }
  if ($forecastCount -gt 0 -and $forecastCount -eq $lineCount) {
    Write-Check -State "OK" -Label "LM prediction" -Detail "forecast=$forecastCount line=$lineCount"
  } else {
    Write-Check -State "WAITING" -Label "LM prediction" -Detail "row exists but line shape not usable forecast=$forecastCount line=$lineCount"
  }
} elseif ($linePrediction.StatusCode -eq 404) {
  Write-Check -State "WAITING" -Label "LM prediction" -Detail "prediction row not stored yet"
} else {
  Write-Check -State "WAITING" -Label "LM prediction" -Detail "$($linePrediction.StatusCode) $($linePrediction.ErrorCode) $($linePrediction.ErrorMessage)"
}

$lineEvaluationPath = "/api/v1/ai/runs/{0}/evaluations?ticker={1}&timeframe={2}&limit=1" -f $LineRunId, $Ticker, $Timeframe
$lineEvaluations = Invoke-DemoGet -Name "LM evaluations" -Path $lineEvaluationPath
$lineEvaluationCount = Get-Count -Value $lineEvaluations.Body.data
if ($lineEvaluations.Ok -and $lineEvaluationCount -gt 0) {
  Write-Check -State "OK" -Label "LM evaluation" -Detail "$lineEvaluationCount rows"
} elseif ($lineEvaluations.Ok) {
  Write-Check -State "WAITING" -Label "LM evaluation" -Detail "evaluation row not stored yet"
} else {
  Write-Check -State "WAITING" -Label "LM evaluation" -Detail "$($lineEvaluations.StatusCode) $($lineEvaluations.ErrorCode) $($lineEvaluations.ErrorMessage)"
}

$lineHistoryPath = "/api/v1/stocks/{0}/predictions/history?run_id={1}&limit=90" -f $Ticker, $LineRunId
$lineHistory = Invoke-DemoGet -Name "LM rolling history" -Path $lineHistoryPath
$lineHistoryCount = Get-Count -Value $lineHistory.Body.data
if ($lineHistory.Ok -and $lineHistoryCount -ge 20) {
  Write-Check -State "OK" -Label "LM history" -Detail "$lineHistoryCount rows"
} elseif ($lineHistory.Ok -and $lineHistoryCount -gt 0) {
  Write-Check -State "WARN" -Label "LM history" -Detail "$lineHistoryCount rows, rolling history is shallow"
} else {
  Write-Check -State "WAITING" -Label "LM history" -Detail "$($lineHistory.StatusCode) $($lineHistory.ErrorCode) $($lineHistory.ErrorMessage)"
}

$bandRun = Invoke-DemoGet -Name "BM band run" -Path ("/api/v1/ai/runs/{0}" -f $BandRunId)
if ($bandRun.Ok -and $bandRun.Body.data.status -eq "completed") {
  $bandRole = $bandRun.Body.data.role
  if (-not $bandRole) {
    $bandRole = $bandRun.Body.data.config_summary.role
  }
  Write-Check -State "OK" -Label "BM run" -Detail "$BandRunId status=completed role=$bandRole"
} elseif ($bandRun.Ok) {
  Write-Check -State "WARN" -Label "BM run" -Detail "$BandRunId status=$($bandRun.Body.data.status)"
} else {
  Write-Check -State "FAIL" -Label "BM run" -Detail "$($bandRun.StatusCode) $($bandRun.ErrorCode) $($bandRun.ErrorMessage)"
}

$bandPredictionPath = "/api/v1/stocks/{0}/predictions/latest?run_id={1}" -f $Ticker, $BandRunId
$bandPrediction = Invoke-DemoGet -Name "BM prediction" -Path $bandPredictionPath
if ($bandPrediction.Ok -and $bandPrediction.Body.data) {
  $forecastCount = Get-Count -Value $bandPrediction.Body.data.forecast_dates
  $upperCount = Get-Count -Value $bandPrediction.Body.data.upper_band_series
  $lowerCount = Get-Count -Value $bandPrediction.Body.data.lower_band_series
  if ($forecastCount -gt 0 -and $forecastCount -eq $upperCount -and $upperCount -eq $lowerCount) {
    Write-Check -State "OK" -Label "BM prediction" -Detail "forecast=$forecastCount upper=$upperCount lower=$lowerCount"
  } else {
    Write-Check -State "FAIL" -Label "BM prediction" -Detail "band shape mismatch forecast=$forecastCount upper=$upperCount lower=$lowerCount"
  }
} else {
  Write-Check -State "FAIL" -Label "BM prediction" -Detail "$($bandPrediction.StatusCode) $($bandPrediction.ErrorCode) $($bandPrediction.ErrorMessage)"
}

$bandEvaluationPath = "/api/v1/ai/runs/{0}/evaluations?ticker={1}&timeframe={2}&limit=1" -f $BandRunId, $Ticker, $Timeframe
$bandEvaluations = Invoke-DemoGet -Name "BM evaluations" -Path $bandEvaluationPath
$bandEvaluationCount = Get-Count -Value $bandEvaluations.Body.data
if ($bandEvaluations.Ok -and $bandEvaluationCount -gt 0) {
  Write-Check -State "OK" -Label "BM evaluation" -Detail "$bandEvaluationCount rows"
} elseif ($bandEvaluations.Ok) {
  Write-Check -State "WARN" -Label "BM evaluation" -Detail "$Ticker evaluation row missing"
} else {
  Write-Check -State "FAIL" -Label "BM evaluation" -Detail "$($bandEvaluations.StatusCode) $($bandEvaluations.ErrorCode) $($bandEvaluations.ErrorMessage)"
}

$bandHistoryPath = "/api/v1/stocks/{0}/predictions/history?run_id={1}&limit=90" -f $Ticker, $BandRunId
$bandHistory = Invoke-DemoGet -Name "BM rolling history" -Path $bandHistoryPath
$bandHistoryCount = Get-Count -Value $bandHistory.Body.data
if ($bandHistory.Ok -and $bandHistoryCount -ge 20) {
  Write-Check -State "OK" -Label "BM history" -Detail "$bandHistoryCount rows"
} elseif ($bandHistory.Ok -and $bandHistoryCount -gt 0) {
  Write-Check -State "WARN" -Label "BM history" -Detail "$bandHistoryCount rows, rolling history is shallow"
} else {
  Write-Check -State "FAIL" -Label "BM history" -Detail "$($bandHistory.StatusCode) $($bandHistory.ErrorCode) $($bandHistory.ErrorMessage)"
}

$bandWidthRows = @($bandHistory.Body.data | Where-Object {
  (Get-Count -Value $_.upper_band_series) -gt 0 -and
  (Get-Count -Value $_.lower_band_series) -gt 0
})
if ($bandHistory.Ok -and $bandWidthRows.Count -gt 0) {
  Write-Check -State "OK" -Label "band width" -Detail "calculable from $($bandWidthRows.Count) BM rows"
} elseif ($bandHistory.Ok) {
  Write-Check -State "WARN" -Label "band width" -Detail "upper/lower band rows missing"
} else {
  Write-Check -State "FAIL" -Label "band width" -Detail "BM history unavailable"
}

Write-Host ""
Write-Host "WAITING means the product run exists but the row is not stored yet. This script does not create fake data."
Write-Host "참고: npm run build 후에는 기존 next dev 서버를 재시작해야 frontend-static 404를 피할 수 있습니다."
