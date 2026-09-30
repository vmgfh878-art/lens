# 발행 결과와 현재 서비스 상태를 함께 알림 색으로 바꾼다.
function Get-ServingNotificationColor {
    param(
        [string]$VerifyStatus,
        [bool]$StepFailed,
        [bool]$ProductionFailed,
        [bool]$OtherWarning
    )
    if ($StepFailed) {
        return "danger"
    }
    if ($VerifyStatus -in @("FALLBACK", "R2_DEGRADED", "REMOTE_DEGRADED")) {
        return "warning"
    }
    if ($ProductionFailed -or $VerifyStatus -in @("ERROR", "STALE", "MISMATCH", "LEGACY")) {
        return "danger"
    }
    if ($OtherWarning) {
        return "warning"
    }
    return "good"
}
