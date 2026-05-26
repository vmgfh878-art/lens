param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$FrontendUrl = "http://127.0.0.1:3000",
  [string]$CorsOrigin = "http://127.0.0.1:3000",
  [string]$Ticker = "AAPL",
  [string]$Timeframe = "1D",
  [string]$LineModelId = "cp210_F4_b4_ensemble_mean",
  [string]$Band1dModelId = "tide-1D-ea54dcae654d",
  [string]$Band1wModelId = "tide_s60_q10_q90_param"
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
Write-Host "Product v1 line/band layer candidates"

$v1LinePath = "/api/v1/predictions/line/{0}?days=30" -f $Ticker
$v1Line = Invoke-DemoGet -Name "1D line F4 beta4 ensemble" -Path $v1LinePath
$v1LineCount = if ($v1Line.Ok) { [int]$v1Line.Body.data.rows } else { 0 }
if ($v1Line.Ok -and $v1LineCount -gt 0) {
  Write-Check -State "OK" -Label "1D line F4 beta4 ensemble" -Detail "$LineModelId rows=$v1LineCount"
} else {
  Write-Check -State "FAIL" -Label "1D line F4 beta4 ensemble" -Detail "$($v1Line.StatusCode) $($v1Line.ErrorCode) $($v1Line.ErrorMessage)"
}

$v1Band1dPath = "/api/v1/predictions/band/1d/{0}?days=30&horizon=5" -f $Ticker
$v1Band1d = Invoke-DemoGet -Name "1D band CP153" -Path $v1Band1dPath
$v1Band1dCount = if ($v1Band1d.Ok) { [int]$v1Band1d.Body.data.rows } else { 0 }
if ($v1Band1d.Ok -and $v1Band1dCount -gt 0) {
  Write-Check -State "OK" -Label "1D band CP153" -Detail "$Band1dModelId rows=$v1Band1dCount"
} else {
  Write-Check -State "FAIL" -Label "1D band CP153" -Detail "$($v1Band1d.StatusCode) $($v1Band1d.ErrorCode) $($v1Band1d.ErrorMessage)"
}

$v1Band1wPath = "/api/v1/predictions/band/1w/{0}?days=120&horizon=4" -f $Ticker
$v1Band1w = Invoke-DemoGet -Name "1W band CP178" -Path $v1Band1wPath
$v1Band1wCount = if ($v1Band1w.Ok) { [int]$v1Band1w.Body.data.rows } else { 0 }
if ($v1Band1w.Ok -and $v1Band1wCount -gt 0) {
  Write-Check -State "OK" -Label "1W band CP178" -Detail "$Band1wModelId rows=$v1Band1wCount"
} else {
  Write-Check -State "FAIL" -Label "1W band CP178" -Detail "$($v1Band1w.StatusCode) $($v1Band1w.ErrorCode) $($v1Band1w.ErrorMessage)"
}

Write-Check -State "OK" -Label "1W line" -Detail "deferred, frontend does not call line endpoint"

Write-Host ""
Write-Host "v1 product slots: 1D Line=F4 beta4 ensemble, 1D Band=CP153, 1W Band=CP178, 1W Line=Deferred."
Write-Host "참고: npm run build 후에는 기존 next dev 서버를 재시작해야 frontend-static 404를 피할 수 있습니다."
