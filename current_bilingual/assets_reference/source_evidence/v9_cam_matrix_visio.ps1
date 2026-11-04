param([string]$OutputDirectory)
$ErrorActionPreference='Stop'
$app=New-Object -ComObject Visio.InvisibleApp
$app.AlertResponse=7
try{
 $doc=$app.Documents.Add('');$page=$doc.Pages.Item(1)
 $page.PageSheet.CellsU('PageWidth').ResultIU=8.5
 $page.PageSheet.CellsU('PageHeight').ResultIU=10
 $page.Name='GradCAM matrix - real raster evidence'
 $s=$page.Import((Join-Path $OutputDirectory 'v9_cam_matrix.png'))
 $s.CellsU('Width').ResultIU=8.5;$s.CellsU('Height').ResultIU=10
 $s.CellsU('PinX').ResultIU=4.25;$s.CellsU('PinY').ResultIU=5
 $doc.SaveAs((Join-Path $OutputDirectory 'v9_cam_matrix.vsdx'))
 $doc.Close()
}finally{$app.Quit()}