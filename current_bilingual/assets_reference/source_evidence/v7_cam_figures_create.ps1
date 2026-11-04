param([string]$OutputDirectory)
$ErrorActionPreference='Stop'
$base=[IO.Path]::GetFullPath($OutputDirectory)
$app=New-Object -ComObject Visio.InvisibleApp
if($app.Documents.Count -ne 0){throw 'Isolated empty Visio required'}
$app.AlertResponse=7
$navy='35,57,80';$blue='59,122,180';$teal='38,154,157';$coral='190,88,74';$gray='90,101,115'
function Style($s,$line=$navy,$size=13){
 $s.CellsU('LineColor').FormulaU="RGB($line)";$s.CellsU('LineWeight').FormulaU='.7 pt';$s.CellsU('FillForegnd').FormulaU='RGB(255,255,255)';$s.CellsU('FillPattern').FormulaU='1'
 $s.CellsU('Char.Font').FormulaU='FONT("Microsoft YaHei")';$s.CellsU('Char.Size').FormulaU="$size pt";$s.CellsU('Char.Color').FormulaU="RGB($navy)"
 try{$s.CellsU('Char.AsianFont').FormulaU='FONT("Microsoft YaHei")'}catch{}
 $s.CellsU('Para.HorzAlign').FormulaU='1';$s.CellsU('VerticalAlign').FormulaU='1'
 foreach($c in @('LeftMargin','RightMargin','TopMargin','BottomMargin')){$s.CellsU($c).FormulaU='.01 in'}
}
function Rect($x,$y,$w,$h,$line=$navy){$s=$page.DrawRectangle($x-$w/2,$y-$h/2,$x+$w/2,$y+$h/2);Style $s $line;return $s}
function Txt($x,$y,$w,$h,$text,$size=13,$bold=$false,$color=$navy){$s=Rect $x $y $w $h;$s.CellsU('LinePattern').FormulaU='0';$s.CellsU('FillPattern').FormulaU='0';$s.CellsU('Char.Size').FormulaU="$size pt";$s.CellsU('Char.Color').FormulaU="RGB($color)";$s.CellsU('Char.Style').FormulaU=$(if($bold){'1'}else{'0'});$s.Text=$text;return $s}
function Box($x,$y,$w,$h,$text,$size=13,$line=$navy){$s=Rect $x $y $w $h $line;$s.CellsU('Char.Size').FormulaU="$size pt";$s.Text=$text;return $s}
function Ln($x1,$y1,$x2,$y2,$arrow=$false,$color=$gray){$s=$page.DrawLine($x1,$y1,$x2,$y2);Style $s $color;$s.CellsU('EndArrow').FormulaU=$(if($arrow){'4'}else{'0'});return $s}
function Pic($name,$x,$y,$w,$h){$s=$page.Import((Join-Path $base $name));$s.CellsU('Width').ResultIU=$w;$s.CellsU('Height').ResultIU=$h;$s.CellsU('PinX').ResultIU=$x;$s.CellsU('PinY').ResultIU=$y;return $s}
function Page($name,$h){$script:doc=$app.Documents.Add('');$script:page=$doc.Pages.Item(1);$page.Name=$name;$page.PageSheet.CellsU('PageWidth').ResultIU=8.5;$page.PageSheet.CellsU('PageHeight').ResultIU=$h;$s=Rect 4.25 ($h/2) 8.5 $h '255,255,255';$s.SendToBack()}
function Save($name){$out=Join-Path $base ($name+'.vsdx');if(Test-Path -LiteralPath $out){throw "Refuse overwrite $out"};$doc.SaveAs($out);$app.Settings.SetRasterExportResolution(2,260,260,0);$page.Export([IO.Path]::ChangeExtension($out,'.png'));Write-Output "$name : $($page.Shapes.Count) editable objects";$doc.Close()}
function ROI($cx,$cy,$side){
 $s=Rect ($cx+((115/224)-.5)*$side) ($cy+(.5-(58/224))*$side) (102/224*$side) (104/224*$side) $coral
 $s.CellsU('FillPattern').FormulaU='0';$s.CellsU('LineWeight').FormulaU='1.6 pt';return $s
}
try{
 Page '六例分析与选例流程' 4.0
 Txt 4.25 3.70 8 .35 '典型样本分析与重点案例选取流程' 16 $true|Out-Null
 Box 1.35 2.95 2.22 .68 "训练与整体测试完成`n保存最终模型和测试清单" 13 $blue|Out-Null
 Ln 2.48 2.95 2.92 2.95 $true|Out-Null
 Box 4.15 2.95 2.38 .68 "冻结模型与预测记录`n不因热图重新调参或改结果" 13 $blue|Out-Null
 Ln 5.38 2.95 5.82 2.95 $true|Out-Null
 Box 7.00 2.95 2.25 .68 "按真实预测分组`n保留全部测试表现" 13 $blue|Out-Null
 Ln 7 2.59 7 2.28|Out-Null;Ln 2.15 2.28 7 2.28|Out-Null
 Ln 2.15 2.28 2.15 2.04 $true|Out-Null;Ln 6.35 2.28 6.35 2.04 $true|Out-Null
 Box 2.15 1.63 3.6 .76 "正确候选：3例`n图像形态不同，逐例写出真实解释" 14 $teal|Out-Null
 Box 6.35 1.63 3.6 .76 "错误候选：3例`n逐例区分定位、分类和整图输出问题" 14 $coral|Out-Null
 Ln 2.15 1.23 2.15 .95 $true $teal|Out-Null;Ln 6.35 1.23 6.35 .95 $true $coral|Out-Null
 Txt 2.15 .70 3.6 .43 '作者从3例中选1例，作为正确重点案例' 13 $true $teal|Out-Null
 Txt 6.35 .70 3.6 .43 '作者从3例中选1例，作为错误重点案例' 13 $true $coral|Out-Null
 Txt 4.25 .22 8.1 .35 '每例保留：原图、真值与预测、分层热图、对应局部、解释与核查依据。六例均保留。' 12|Out-Null
 Save 'v7_fig10_case_selection'

 Page '同一手图的分层GradCAM' 3.65
 Txt 4.25 3.38 8.1 .35 '同一输入的分层Grad-CAM响应' 15 $true|Out-Null
 Txt 4.25 3.02 8.1 .35 '真实计算演示：ImageNet目标类 nail（钉子），非手势类别' 13 $true $coral|Out-Null
 $xs=@(.91,2.58,4.25,5.92,7.59)
 $labels=@("实际输入`n224 × 224", "layer4 · 主解释`n原始响应7 × 7", "layer1`n原始响应56 × 56", "layer2`n原始响应28 × 28", "layer3`n原始响应14 × 14")
 $files=@('v4_feature_input224.png','v7_cam_layer4_overlay.png','v7_cam_layer1_overlay.png','v7_cam_layer2_overlay.png','v7_cam_layer3_overlay.png')
 for($i=0;$i -lt 5;$i++){Txt $xs[$i] 2.56 1.58 .48 $labels[$i] 12 ($i -eq 1)|Out-Null;Pic $files[$i] $xs[$i] 1.55 1.48 1.48|Out-Null}
 Pic 'v7_cam_colorbar.png' 4.25 .62 2.7 .12|Out-Null
 Txt 2.72 .62 .3 .25 '0' 11|Out-Null;Txt 5.78 .62 .3 .25 '1' 11|Out-Null
 Txt 4.25 .29 8.1 .42 "各图独立归一化：仅表示本图内相对响应，不作跨层强度比较。`n最终层只有7 × 7响应格；插值后的平滑颜色不等于精确指尖定位。" 12|Out-Null
 Save 'v7_fig11_layer_cam'

 Page '照片响应与局部形态对应' 5.25
 Txt 4.25 4.98 8.1 .35 '原始照片、分类输入与局部响应对应关系' 15 $true|Out-Null
 Txt 4.25 4.61 8.1 .34 'ImageNet目标类：nail（钉子），非手势类别；此图展示对应观察方法' 13 $true $coral|Out-Null
 Txt 1.35 4.18 2.3 .3 '原始照片与人工手框' 13 $true|Out-Null
 Txt 4.15 4.18 2.35 .3 '送入网络的手图' 13 $true|Out-Null
 Txt 6.95 4.18 2.35 .3 '同一手图 + layer4响应' 13 $true|Out-Null
 Pic 'v4_feature_input_boxed.png' 1.35 3.12 1.45 1.93|Out-Null
 Pic 'v4_feature_input224.png' 4.15 3.12 1.93 1.93|Out-Null
 Pic 'v7_cam_layer4_overlay.png' 6.95 3.12 1.93 1.93|Out-Null
 ROI 4.15 3.12 1.93|Out-Null;ROI 6.95 3.12 1.93|Out-Null
 Ln 2.16 3.12 2.98 3.12 $true|Out-Null
 Txt 2.57 3.49 .85 .55 "按手框裁图`n等比补边" 11|Out-Null
 Ln 5.17 3.12 5.90 3.12 $true|Out-Null
 Txt 5.55 3.49 .74 .5 '叠加响应' 11|Out-Null
 Ln 3.736 3.137 3.40 2.05 $false $coral|Out-Null;Ln 6.536 3.137 6.15 2.05 $false $coral|Out-Null
 Txt 1.15 1.65 1.95 .6 "红框是人工选定的`n形态观察区，非热区分割" 12 $true $coral|Out-Null
 Pic 'v7_cam_input_roi.png' 3.38 1.45 1.18 1.2|Out-Null
 Pic 'v7_cam_layer4_roi.png' 6.13 1.45 1.18 1.2|Out-Null
 Txt 3.38 .58 3.3 .4 '同一区域：手指轮廓与指间空隙' 12 $true|Out-Null
 Txt 6.13 .58 3.3 .4 '同坐标响应：对照位置与扩散范围' 12 $true|Out-Null
 Txt 4.25 .13 8.1 .25 '本例为宽范围响应，覆盖手指根部、掌部及部分背景；不能据此声称模型已识别V手势。' 12|Out-Null
 Save 'v7_fig12_local_evidence'
}finally{if($null -ne $app){$app.Quit();[void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)}}
