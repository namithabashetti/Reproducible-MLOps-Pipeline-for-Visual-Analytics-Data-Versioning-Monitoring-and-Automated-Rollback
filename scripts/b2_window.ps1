# Local credential entry only; no credentials are written to disk or command arguments.
param([ValidateSet('push-verify', 'verify')][string]$Action = 'verify')

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Backblaze B2 - paste replacement credentials'
$form.ClientSize = New-Object System.Drawing.Size(540, 280)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

$instructions = New-Object System.Windows.Forms.Label
$instructions.Text = 'Copy each replacement key, then click its Paste button. Dots confirm entry.'
$instructions.SetBounds(18, 15, 504, 40)
$form.Controls.Add($instructions)

$idLabel = New-Object System.Windows.Forms.Label
$idLabel.Text = 'Replacement key ID'
$idLabel.SetBounds(18, 60, 320, 24)
$form.Controls.Add($idLabel)
$keyId = New-Object System.Windows.Forms.TextBox
$keyId.UseSystemPasswordChar = $true
$keyId.SetBounds(18, 86, 394, 28)
$form.Controls.Add($keyId)
$pasteId = New-Object System.Windows.Forms.Button
$pasteId.Text = 'Paste'
$pasteId.SetBounds(426, 84, 96, 32)
$pasteId.Add_Click({ $keyId.SelectAll(); $keyId.Paste() })
$form.Controls.Add($pasteId)

$secretLabel = New-Object System.Windows.Forms.Label
$secretLabel.Text = 'Replacement application key'
$secretLabel.SetBounds(18, 126, 360, 24)
$form.Controls.Add($secretLabel)
$secret = New-Object System.Windows.Forms.TextBox
$secret.UseSystemPasswordChar = $true
$secret.SetBounds(18, 152, 394, 28)
$form.Controls.Add($secret)
$pasteSecret = New-Object System.Windows.Forms.Button
$pasteSecret.Text = 'Paste'
$pasteSecret.SetBounds(426, 150, 96, 32)
$pasteSecret.Add_Click({ $secret.SelectAll(); $secret.Paste() })
$form.Controls.Add($pasteSecret)

$counts = New-Object System.Windows.Forms.Label
$counts.Text = 'Key ID: 0 characters. Application key: 0 characters.'
$counts.SetBounds(18, 191, 504, 26)
$form.Controls.Add($counts)
$start = New-Object System.Windows.Forms.Button
$start.Text = if ($Action -eq 'verify') { 'Resume verification' } else { 'Upload and verify' }
$start.SetBounds(338, 231, 184, 34)
$start.Enabled = $false
$start.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.Controls.Add($start)
$update = {
    $counts.Text = "Key ID: $($keyId.Text.Length) characters. Application key: $($secret.Text.Length) characters."
    $start.Enabled = $keyId.Text.Trim().Length -gt 0 -and $secret.Text.Trim().Length -gt 0
}
$keyId.Add_TextChanged($update)
$secret.Add_TextChanged($update)

try {
    if ($form.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return }
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $processInfo.WorkingDirectory = $projectRoot
    $processInfo.Arguments = "scripts\b2_session.py $Action --from-environment"
    $processInfo.UseShellExecute = $false
    $processInfo.EnvironmentVariables['AWS_ACCESS_KEY_ID'] = $keyId.Text.Trim()
    $processInfo.EnvironmentVariables['AWS_SECRET_ACCESS_KEY'] = $secret.Text.Trim()
    Write-Host "Starting B2 $Action. Keep this window open."
    $transfer = [System.Diagnostics.Process]::Start($processInfo)
    $processInfo.EnvironmentVariables.Remove('AWS_ACCESS_KEY_ID')
    $processInfo.EnvironmentVariables.Remove('AWS_SECRET_ACCESS_KEY')
    $keyId.Clear()
    $secret.Clear()
    $transfer.WaitForExit()
    if ($transfer.ExitCode -eq 0) { Write-Host 'B2 content verification completed successfully.' }
    else { Write-Host 'B2 verification did not complete. Report the error above without sharing credentials.' }
} finally {
    $keyId.Clear()
    $secret.Clear()
    $form.Dispose()
}
